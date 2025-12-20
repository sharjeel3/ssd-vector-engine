# ADR-004: SSD IO Patterns and Optimization

## Status
Proposed

## Context

The SSD Vector Engine is explicitly designed as a **disk-first** system that leverages NVMe SSD performance characteristics for cost-effective vector search at scale. Unlike in-memory systems, our performance is bounded by disk I/O bandwidth and latency. We need to optimize access patterns to maximize throughput while minimizing latency impact.

### Key SSD Characteristics (NVMe)

**Performance Profile:**
- **Sequential Read**: 3-7 GB/s
- **Random Read (4KB)**: 500K-1M IOPS, ~0.05-0.1ms latency
- **Sequential Write**: 2-5 GB/s
- **Random Write (4KB)**: 300K-800K IOPS
- **Queue Depth Impact**: Performance scales with queue depth up to 32-128

**Access Pattern Implications:**
1. Sequential access is 10-100x faster than random access
2. Large block reads (64KB-256KB) are more efficient than 4KB reads
3. Read amplification hurts performance significantly
4. Write amplification affects both performance and endurance

### System Requirements

From initial requirements:
- **Query Latency Target**: p99 < 50ms, p50 < 10ms
- **Throughput Target**: 500+ QPS with 8 threads
- **Dataset Scale**: 10M vectors @ 1536D (58GB raw data)
- **Memory Budget**: 16GB RAM (per ADR-003)
- **Build Time**: < 5 hours for 10M vectors

### I/O Bottlenecks Identified

1. **Graph Navigation**: DiskANN requires 20-50 random reads per query
2. **Vector Loading**: 1536D float32 vector = 6KB per read
3. **Build Phase**: Massive random read/write during graph construction
4. **Cold Start**: First query on cold cache incurs full disk latency

---

## Decision

We will adopt the following SSD I/O optimization strategies:

### 1. **Layout Optimization: Sequential Graph Storage**

**Strategy**: Store graph adjacency lists in a **layout-optimized format** that maximizes spatial locality.

**Implementation:**
```
Graph File Layout:
┌─────────────────────────────────────┐
│ Header (256 bytes)                   │
├─────────────────────────────────────┤
│ Offset Table (8 bytes × N nodes)     │  ← Memory-mapped, stays in RAM
├─────────────────────────────────────┤
│ Node 0 Adjacency (degree + neighbors)│
│ Node 1 Adjacency                     │
│ Node 2 Adjacency                     │
│ ...                                  │  ← Laid out in search order
│ Node N Adjacency                     │
└─────────────────────────────────────┘
```

**Key Decisions:**
- **Graph nodes stored in DFS order** from entry point (medoid)
  - Rationale: Hot nodes (near medoid) are co-located, improving cache efficiency
- **64-byte alignment** for all node records
  - Rationale: Matches SSD page size, reduces read amplification
- **Offset table memory-mapped** for O(1) random access
  - Rationale: 80MB for 10M nodes (8 bytes × 10M) fits in memory budget

---

### 2. **Read Optimization: Large Block Reads with Prefetching**

**Strategy**: Read 64KB blocks instead of individual 4KB pages, prefetch likely-next nodes.

**Techniques:**

#### a) **Adaptive Block Size**
```cpp
// Read entire 64KB block containing target node
constexpr size_t BLOCK_SIZE = 64 * 1024;  // 64KB

// When reading node, read entire block
uint64_t block_offset = (node_offset / BLOCK_SIZE) * BLOCK_SIZE;
read_block(block_offset, BLOCK_SIZE);
```

**Rationale**: 
- 64KB sequential read ≈ same latency as 4KB read (~0.05ms)
- Gets ~10-16 nodes per block (assuming avg degree 32, ~4KB per node)
- Amortizes disk seek over multiple nodes

#### b) **Neighbor Prefetching**
```cpp
// During graph traversal, prefetch next level
for (neighbor in current_node.neighbors) {
    if (!in_cache(neighbor)) {
        async_prefetch(neighbor);  // Non-blocking prefetch
    }
}
```

**Rationale**:
- Graph search is predictable: we'll visit neighbors soon
- Prefetch hides ~50% of disk latency
- Queue depth increases disk utilization

#### c) **Read-Ahead for Sequential Access**
```cpp
// During index load or build
madvise(addr, length, MADV_SEQUENTIAL | MADV_WILLNEED);
```

**Rationale**:
- Tells OS to prefetch aggressively for sequential scans
- Reduces latency during index building and pre-warming

---

### 3. **Write Optimization: Batched Sequential Writes**

**Strategy**: Buffer writes in memory, flush in large sequential batches.

**Techniques:**

#### a) **Build Phase: Memory-Buffered Graph Construction**
```cpp
// During index build:
1. Construct graph entirely in memory (fits in 16GB for 10M vectors)
2. Optimize layout (DFS order from medoid)
3. Write sequentially to disk in single pass
4. Fsync once at end
```

**Rationale**:
- Eliminates random writes during build
- Sequential write is 50-100x faster than random writes
- Reduces write amplification and SSD wear

#### b) **Delta Index: Log-Structured Writes**
```cpp
// New vectors appended sequentially
delta_file.append(vector_data);
delta_file.append(metadata);
// No in-place updates
```

**Rationale**:
- Sequential appends maximize write throughput
- Defers graph updates to rebuild phase
- Simplifies crash recovery (append-only log)

---

### 4. **Cache Strategy: Hot Data in Memory**

**Strategy**: Keep frequently accessed data in memory per ADR-003 memory budget.

**Cache Tiers** (from ADR-003):
1. **Graph Cache (8GB)**: LRU cache of hot graph nodes
   - Target: 70%+ hit rate after warm-up
   - ~2M nodes cached (assuming 4KB per node)
   
2. **Vector Cache (2GB)**: LRU cache of hot vectors
   - Target: 20%+ hit rate
   - ~330K vectors cached (6KB per 1536D vector)

3. **OS Page Cache (remainder)**: Kernel manages remaining data
   - Benefits from sequential reads and prefetching

**Cache Pre-warming**:
```cpp
// At startup, pre-load hot nodes
void prewarm_cache() {
    // Load medoid and its k-hop neighbors
    vector<uint32_t> hot_nodes = bfs_from_medoid(hops=3);
    for (node_id : hot_nodes) {
        graph_cache.load(node_id);
        vector_cache.load(node_id);
    }
}
```

**Rationale**:
- First 1000 queries after cold start are slow (~100ms)
- Pre-warming reduces cold-start latency to ~20ms
- 3-hop BFS from medoid covers ~32^3 = 32K nodes = ~128MB

---

### 5. **Memory-Mapped I/O for Read-Heavy Paths**

**Strategy**: Use `mmap()` for index files, let OS manage paging.

**Components Using mmap:**
- ✅ **Graph offset table**: Always in memory
- ✅ **Vector data**: Partially in memory (OS page cache)
- ✅ **Metadata**: Small enough to stay resident
- ❌ **Delta index**: Use direct I/O (frequently modified)

**Configuration:**
```cpp
// Graph file
int fd = open("graph.bin", O_RDONLY);
void* addr = mmap(NULL, file_size, PROT_READ, MAP_SHARED, fd, 0);

// Advise kernel on access pattern
madvise(addr, file_size, MADV_RANDOM);  // Random access expected
madvise(addr, file_size, MADV_WILLNEED);  // Prefetch aggressively
```

**Benefits:**
- Zero-copy reads (no kernel → userspace copy)
- OS manages page cache automatically
- Simplified code (no manual buffer management)

**Trade-offs:**
- Page faults on first access (~0.05ms per page)
- Less control over eviction policy
- Requires careful error handling (SIGBUS on I/O errors)

---

### 6. **I/O Budget and Monitoring**

**Per-Query I/O Budget:**
- **Target Disk Reads**: 20-50 random reads
  - Graph nodes: 15-40 reads (avg search depth)
  - Vectors: 20-50 reads (same nodes, different file)
- **Target Disk Time**: < 5ms (0.1ms × 50 reads with caching)
- **Target Cache Hits**: 70% graph, 20% vector

**Build Phase I/O Budget:**
- **Sequential Read**: 58GB raw vectors @ 3 GB/s = 20 seconds
- **Random Read/Compute**: Graph construction dominates (4.5 hours)
- **Sequential Write**: 300-500MB graph @ 3 GB/s = 0.2 seconds

**Monitoring Metrics:**
```cpp
struct IOMetrics {
    // Per query
    uint64_t disk_reads;
    uint64_t disk_bytes_read;
    double disk_time_ms;
    
    // Cache
    double graph_cache_hit_rate;
    double vector_cache_hit_rate;
    
    // System
    double disk_utilization;  // % time disk is busy
    uint64_t disk_queue_depth;
};
```

**Alerts:**
- Disk reads > 100 per query → Cache warming needed
- Cache hit rate < 50% → Increase cache size or optimize layout
- Disk utilization > 90% → Add more query nodes

---

### 7. **Direct I/O for Write Path**

**Strategy**: Use `O_DIRECT` for WAL writes to bypass OS cache.

**Rationale:**
- WAL writes must be durable (fsync)
- OS page cache adds no value for write-once data
- Direct I/O eliminates double-buffering (app buffer + page cache)

**Implementation:**
```cpp
// WAL file
int fd = open("wal.log", O_WRONLY | O_DIRECT | O_APPEND);

// Write 4KB-aligned blocks
alignas(4096) char buffer[4096];
write(fd, buffer, 4096);
fsync(fd);  // Ensure durability
```

**Trade-off:**
- Requires 4KB-aligned buffers (more complex)
- No kernel buffering (must manage ourselves)
- Better control over durability guarantees

---

## I/O Access Pattern Summary

### Index Build Phase
| Operation | Pattern | Volume | Strategy |
|-----------|---------|--------|----------|
| Load vectors | Sequential | 58GB | Large block reads, madvise SEQUENTIAL |
| Build graph | Random R/W | In-memory | No disk writes until final flush |
| Write index | Sequential | 300-500MB | Single sequential write, fsync |
| **Total Time** | - | - | **< 5 hours** |

### Query Phase (Warm Cache)
| Operation | Pattern | Volume | Strategy |
|-----------|---------|--------|----------|
| Graph traverse | Random | 15-40 nodes | Cache + prefetch, 64KB blocks |
| Vector read | Random | 20-50 vectors | Cache + prefetch, 64KB blocks |
| **Disk Reads** | - | **5-15 reads** | **< 5ms disk time** |
| **Total Latency** | - | - | **< 10ms p50, < 50ms p99** |

### Write Path (Delta Index)
| Operation | Pattern | Volume | Strategy |
|-----------|---------|--------|----------|
| WAL append | Sequential | ~6KB/write | O_DIRECT, fsync every 10ms or 1MB |
| Delta append | Sequential | ~6KB/write | Buffered, fsync on commit |
| **Throughput** | - | - | **1000+ writes/sec** |

---

## Consequences

### Positive

1. **High Query Throughput**
   - 70% cache hit rate reduces disk I/O to ~10 reads/query
   - Prefetching hides 50% of remaining disk latency
   - Can sustain 500+ QPS with 8 threads

2. **Fast Index Builds**
   - In-memory graph construction eliminates random writes
   - Sequential final write maximizes disk bandwidth
   - < 5 hours for 10M vectors achievable

3. **Predictable Performance**
   - I/O patterns are well-characterized
   - Cache hit rates are measurable and tunable
   - Clear monitoring metrics for operations

4. **SSD Longevity**
   - Minimized write amplification (append-only, batch writes)
   - No random writes in critical path
   - Reduced wear on flash cells

### Negative

1. **Cold Start Penalty**
   - First 1000 queries incur full disk latency (~100ms)
   - Requires explicit cache pre-warming
   - Impact: ~20-30 seconds warm-up time

2. **Memory Pressure**
   - 8GB graph cache is significant memory commitment
   - Less memory available for other processes
   - May need tuning for smaller deployments

3. **Code Complexity**
   - mmap error handling (SIGBUS) is tricky
   - Prefetching logic adds complexity
   - Direct I/O requires aligned buffers

4. **OS Dependency**
   - madvise, O_DIRECT are Linux-specific
   - macOS/Windows require different APIs
   - Portability challenges

---

## Implementation Notes

### Phase 1: Basic I/O (Milestone 1)
- ✅ Memory-mapped graph and vector files
- ✅ Sequential index writing
- ✅ Basic caching (LRU)
- ⬜ Prefetching (optional for M1)

### Phase 2: Optimization (Milestone 2)
- ⬜ Adaptive prefetching based on graph structure
- ⬜ Cache pre-warming on startup
- ⬜ Direct I/O for WAL writes
- ⬜ DFS-ordered graph layout

### Phase 3: Advanced (Milestone 3+)
- ⬜ Tiered caching (SSD cache for flash-optimized systems)
- ⬜ NUMA-aware memory allocation
- ⬜ io_uring for async I/O (Linux 5.1+)
- ⬜ Multi-tier storage (NVMe + SATA SSD)

---

## Validation

### Performance Tests

1. **Cold Cache Query Latency**
   - Metric: p99 < 100ms (first 1000 queries)
   - Method: Clear page cache, run benchmark

2. **Warm Cache Query Latency**
   - Metric: p99 < 50ms (after warm-up)
   - Method: Pre-warm cache, run 10K queries

3. **Cache Hit Rates**
   - Metric: Graph 70%+, Vector 20%+
   - Method: Instrument cache, measure over 10K queries

4. **Build Time**
   - Metric: < 5 hours for 10M vectors
   - Method: Full index build with timing

5. **Disk Utilization**
   - Metric: < 80% average during query load
   - Method: Monitor with `iostat` during load test

### I/O Pattern Validation

```bash
# Monitor I/O during query load
iostat -x 1 60

# Expected output during query phase:
# - rrqm/s < 10 (few read merges, mostly random)
# - r/s: 100-500 (reads per second per thread)
# - rMB/s: 50-200 MB/s
# - avgqu-sz: 4-16 (queue depth)
# - await: 1-5ms (avg read latency)

# Monitor page cache effectiveness
vmtouch graph.bin vectors.bin

# Expected:
# - graph.bin: 20-30% resident (hot nodes)
# - vectors.bin: 5-10% resident (hot vectors)
```

---

## Alternatives Considered

### Alternative 1: Pure In-Memory Index
**Rejected**: Cost prohibitive at scale
- 10M × 1536D × 4 bytes = 58GB RAM for vectors alone
- Graph adds another 8-10GB
- Total: ~70GB RAM per node vs 16GB with disk-first approach

### Alternative 2: Compressed Vectors on Disk
**Deferred to v2**: Adds complexity without clear wins
- Product quantization (PQ) reduces storage 8-16x
- But requires decompression on every read (CPU cost)
- Better suited for memory-constrained deployments

### Alternative 3: Custom SSD Cache Layer (e.g., Open-CAS)
**Rejected**: Adds operational complexity
- Requires kernel module installation
- Harder to debug I/O issues
- OS page cache is "good enough" for v1

### Alternative 4: io_uring for Async I/O
**Deferred to v2**: Linux 5.1+ only
- Excellent performance (lower latency, higher throughput)
- But limited OS support (not on macOS/older Linux)
- Fallback to traditional async I/O adds complexity

---

## Related Decisions

- **ADR-001**: Storage format (defines file layouts)
- **ADR-002**: DiskANN algorithm (determines I/O pattern)
- **ADR-003**: Memory budget (defines cache sizes)
- **COMP-002**: Index storage layer (implements these patterns)
- **COMP-004**: Search engine (consumes optimized I/O)

---

## References

1. **DiskANN Paper**: https://arxiv.org/abs/1907.10310
   - Section 3.2: SSD-Optimized Index Layout
   - Section 4: I/O Optimization Techniques

2. **NVMe Performance Guide**: https://nvmexpress.org/
   - Queue Depth Optimization
   - Block Size Tuning

3. **Linux Kernel Documentation**:
   - mmap: https://man7.org/linux/man-pages/man2/mmap.2.html
   - madvise: https://man7.org/linux/man-pages/man2/madvise.2.html
   - O_DIRECT: https://man7.org/linux/man-pages/man2/open.2.html

4. **Recommended Reading**:
   - "What Every Programmer Should Know About Memory" (Ulrich Drepper)
   - "The Pathologies of Big Data" (Gray et al.)

---

**Decision Date**: 2025-12-20  
**Status**: Proposed  
**Deciders**: Engineering Team  
**Revisit Date**: After Milestone 1 implementation
