# Architecture Decision: Memory Budget and Caching Strategy

## Status
- [x] Proposed
- [ ] Accepted
- [ ] Rejected
- [ ] Deprecated

**Date**: 2025-12-20  
**Deciders**: Engineering Team  
**Related Specs**: ADR-001, ADR-002, COMP-004

---

## Context

The SSD Vector Engine's core value proposition is **SSD-first architecture** that minimizes memory requirements compared to in-memory ANN systems. However, strategic use of RAM for caching is critical for query performance.

### The Memory Challenge

**In-Memory Approach** (HNSW, IVF):
- 10M vectors × 1536D × 4 bytes = **61.4GB** (vectors alone)
- Plus graph structure: ~1.3GB
- Plus metadata: ~80MB
- **Total: ~63GB RAM** per collection

**Our Constraint**:
- Target: 10M vectors per node
- Hardware: Commodity VMs with 16-32GB RAM
- Must fit comfortably within memory budget
- Leave headroom for OS, other processes

### What Needs Memory?

1. **Graph node cache** - Hot nodes accessed frequently
2. **Working sets** - Per-query candidate lists and buffers
3. **Metadata** - ID mappings, filters (small, keep in RAM)
4. **OS page cache** - Managed automatically by OS
5. **Application overhead** - Allocators, buffers, etc.

---

## Decision

We will adopt a **tiered memory strategy** with the following allocation:

### Memory Budget: 16GB RAM per Node

**Rationale**:
- Fits on standard VMs (e.g., AWS m5.xlarge: 16GB, 4 vCPU)
- Leaves ~4GB for OS, monitoring, buffers
- Scales cost-effectively
- Can upgrade to 32GB if needed

---

## Memory Allocation (16GB Node)

### Tier 1: Essential Data (RAM-Resident)

#### 1.1 Metadata Store: 1GB
**Contents**:
- ID mappings (int64): 10M × 8 bytes = 80MB
- Filter indexes (B-trees): ~200MB
- Tombstone bitmaps: 10M bits = 1.25MB
- Metadata values: ~400MB
- Overhead (allocator, fragmentation): ~300MB

**Policy**: Always in RAM, never evicted

**Rationale**:
- Metadata queries must be fast
- Filter evaluation needs quick access
- Relatively small dataset

---

#### 1.2 Delta Index: 2GB
**Contents**:
- Recent writes (buffer): ~1M vectors × 1536D × 4B = 6.1GB in theory
- **Limit**: Cap delta at 100k vectors = 610MB
- In-memory HNSW or flat index: ~800MB
- Metadata for delta vectors: ~100MB
- Working space: ~490MB

**Policy**: RAM-resident until merge

**Rationale**:
- Delta must be fast (no disk)
- Bounded size prevents memory blow-up
- Flush to disk when limit reached

---

### Tier 2: Hot Caches (RAM-Resident, LRU)

#### 2.1 Graph Node Cache: 8GB
**Contents**:
- Cached graph nodes with adjacency lists
- Node structure: 32 neighbors × 4 bytes + overhead = 256 bytes
- Cache capacity: 8GB / 256B = **33M nodes**
- For 10M node graph: ~3.3x over-subscription (good coverage)

**Policy**: LRU eviction, pre-warm on startup

**Eviction Strategy**:
```
if (cache_miss) {
    load_node_from_disk(node_id);
    if (cache_full) {
        evict_lru_node();
    }
    cache.insert(node_id, node_data);
}
```

**Pre-warming**: Load entry points + first few hops on startup

**Expected Hit Rate**: 70-85% (hot nodes stay cached)

**Rationale**:
- Most queries access same hot nodes (power law distribution)
- Large cache reduces disk IO significantly
- 8GB provides good coverage for 10M vectors

---

#### 2.2 Vector Cache: 2GB
**Contents**:
- Recently accessed vectors (for distance computation)
- Vector size: 1536D × 4B = 6KB
- Cache capacity: 2GB / 6KB = **~340k vectors**
- For 10M vectors: ~3.4% coverage

**Policy**: LRU eviction

**Rationale**:
- Vectors accessed less frequently than graph (more random)
- Cache helps but not as critical as graph cache
- Memory better spent on graph cache

**Optimization**: Cache hot query vectors (if query patterns repeat)

---

### Tier 3: Working Memory (RAM-Resident)

#### 3.1 Query Working Sets: 2GB
**Contents**:
- Per-query candidate lists: ~100 entries × 6KB = 600KB
- Distance computation buffers: ~500KB
- Result sets: ~100KB
- **Per query total**: ~1.2MB
- **Concurrent queries**: 2GB / 1.2MB = ~1600 queries

**Policy**: Allocated per query, freed on completion

**Rationale**:
- Queries need scratch space
- Concurrent query capacity: ~1600 (more than enough for v1)

---

### Tier 4: OS and Overhead: 1GB
**Contents**:
- OS kernel: ~500MB
- C++ allocator overhead: ~200MB
- Monitoring agents: ~100MB
- Network buffers: ~100MB
- Reserve: ~100MB

**Policy**: System-managed

**Rationale**: Leave headroom for OS and tools

---

## Memory Budget Summary Table

| Component | Allocation | Policy | Notes |
|-----------|-----------|--------|-------|
| **Metadata Store** | 1GB | RAM-resident | Never evicted |
| **Delta Index** | 2GB | RAM-resident | Bounded size |
| **Graph Node Cache** | 8GB | LRU cache | 70-85% hit rate |
| **Vector Cache** | 2GB | LRU cache | ~3% coverage |
| **Query Working Sets** | 2GB | Allocated on demand | ~1600 concurrent queries |
| **OS + Overhead** | 1GB | System-managed | Buffer |
| **TOTAL** | **16GB** | | Target node size |

---

## Caching Policies

### Graph Node Cache (LRU with Pre-warming)

**Cache Key**: Node ID (uint32)

**Cache Value**:
```cpp
struct CachedNode {
    uint32_t node_id;
    uint32_t degree;           // Number of neighbors
    uint32_t neighbors[32];    // Max 32 neighbors (R=32)
    uint64_t last_access_ts;   // For LRU
};
```

**Pre-warming on Startup**:
```python
def prewarm_graph_cache():
    # Load entry points
    for entry_point in entry_points:
        cache.load(entry_point)
    
    # Load 1-hop neighbors (most accessed)
    for entry in entry_points:
        for neighbor in entry.neighbors:
            cache.load(neighbor)
    
    # Optional: Load 2-hop neighbors if memory permits
```

**Access Pattern**:
```cpp
Node* get_node(uint32_t node_id) {
    if (node = cache.get(node_id)) {
        cache.touch(node_id);  // Update LRU
        return node;
    }
    
    // Cache miss - load from disk
    Node* node = load_from_disk(node_id);
    cache.insert(node_id, node);
    return node;
}
```

**Eviction Policy**: True LRU (least recently accessed)

---

### Vector Cache (LRU, Opportunistic)

**Cache Key**: Vector ID (uint32)

**Cache Value**:
```cpp
struct CachedVector {
    uint32_t vector_id;
    float data[DIMENSION];     // 1536D × 4B = 6KB
    float norm;                // Precomputed for cosine
    uint64_t last_access_ts;
};
```

**Loading Strategy**:
- Load on-demand during distance computation
- Evict when cache full

**Optimization**: Batch-load neighbors during graph traversal

---

### Metadata (No Eviction)

**Structure**:
```cpp
struct MetadataStore {
    // ID mapping
    std::vector<int64_t> external_ids;  // internal -> external
    std::unordered_map<int64_t, uint32_t> id_lookup;  // external -> internal
    
    // Filter indexes
    std::map<std::string, BTreeIndex> filter_indexes;
    
    // Tombstones
    std::bitset<10'000'000> tombstones;
};
```

**Policy**: Always keep entire metadata in RAM

**Rationale**: Small enough to fit, critical for fast filtering

---

## Cache Monitoring and Tuning

### Key Metrics

| Metric | Description | Target | Action if Below |
|--------|-------------|--------|-----------------|
| **graph_cache_hit_rate** | % of node accesses served from cache | > 70% | Increase cache size |
| **vector_cache_hit_rate** | % of vector accesses served from cache | > 20% | Consider increasing cache |
| **cache_memory_usage** | Actual RAM used by caches | < 10GB | Within budget |
| **cache_eviction_rate** | Evictions per second | < 1000/s | Normal churn |
| **prewarm_time** | Cache pre-warming duration on startup | < 30s | Acceptable |

### Tuning Guidelines

**If p99 latency > target**:
1. Check `graph_cache_hit_rate`
   - If < 70%: Increase graph cache (reduce vector cache)
2. Check `disk_reads_per_query`
   - If > 50: Need more caching or better graph layout

**If memory pressure**:
1. Reduce graph cache size
2. Reduce delta index capacity
3. Limit concurrent queries

**If recall drops**:
- Not a caching issue - tune L_search in ADR-002

---

## Scaling Strategies

### Vertical Scaling (More RAM per Node)

#### 32GB Node Configuration

| Component | 16GB | 32GB | Change |
|-----------|------|------|--------|
| Metadata | 1GB | 2GB | 2x vectors (20M) |
| Delta Index | 2GB | 4GB | 2x capacity |
| Graph Cache | 8GB | 18GB | 2.25x cache |
| Vector Cache | 2GB | 4GB | 2x cache |
| Working Sets | 2GB | 3GB | 1.5x concurrency |
| OS + Overhead | 1GB | 1GB | Same |

**Benefits**:
- Support 20M vectors per node
- Higher cache hit rates (80-90%)
- More concurrent queries (~2400)

**Cost**: ~2x instance cost

**Recommendation**: Start with 16GB, upgrade if needed

---

### Horizontal Scaling (More Nodes)

**Sharding Strategy** (future):
- Partition vectors across nodes
- Route queries based on partition
- Each node runs independently

**Not in v1 scope** - deferred to v2

---

## Memory Management Implementation

### Allocator Strategy

**Use jemalloc or tcmalloc** instead of glibc malloc

**Rationale**:
- Better fragmentation handling
- Thread-local caching (lower contention)
- Memory profiling tools

**Configuration**:
```bash
# jemalloc tuning
export MALLOC_CONF="dirty_decay_ms:5000,muzzy_decay_ms:10000"
```

---

### Memory Limits and OOM Protection

**Set hard limits**:
```cpp
// Pseudo-code
struct MemoryLimits {
    size_t graph_cache_max = 8 * GB;
    size_t vector_cache_max = 2 * GB;
    size_t delta_index_max = 2 * GB;
};
```

**OOM Handling**:
```cpp
void* safe_alloc(size_t size) {
    if (current_usage + size > MEMORY_LIMIT) {
        // Trigger cache eviction
        cache.evict_until(size + BUFFER);
        
        if (current_usage + size > HARD_LIMIT) {
            throw OutOfMemoryError();
        }
    }
    return malloc(size);
}
```

---

## Disk + OS Page Cache

### OS Page Cache (Not Managed by Application)

**How it works**:
- OS caches recently read disk blocks
- Transparent to application
- Uses "free" RAM not allocated to applications

**Benefit**:
- Additional ~4GB caching (on 16GB node)
- Helps with vector file reads
- No management overhead

**Interaction with application cache**:
- Application cache = explicit, controlled
- OS page cache = implicit, best-effort
- Together provide multi-tier caching

---

### Memory-Mapped Files

**Usage**: Graph and vector files are mmap'd (per ADR-001)

**Memory Impact**:
- Virtual address space: Yes (large)
- Physical RAM (RSS): Only accessed pages

**Example**:
- mmap 60GB vector file
- Virtual address space: +60GB
- Actual RAM usage: ~2GB (vector cache) + OS page cache

**Benefit**: OS manages paging automatically

---

## Failure Modes and Handling

### Out of Memory (OOM)

**Symptoms**:
- Malloc failures
- Cache eviction storms
- OOM killer invoked

**Prevention**:
1. Hard memory limits (enforce budget)
2. Delta index size cap (bounded buffer)
3. Query concurrency limits (prevent overload)
4. Monitoring and alerting (memory_usage > 90%)

**Recovery**:
- Reject new queries (backpressure)
- Force delta flush (free 2GB)
- Emergency cache clear (last resort)

---

### Cache Thrashing

**Symptoms**:
- Cache hit rate drops < 50%
- Query latency spikes
- High cache eviction rate

**Causes**:
- Cache too small for working set
- Query pattern too random (no locality)

**Mitigation**:
- Increase cache size (vertical scaling)
- Analyze query patterns (logging)
- Improve graph layout (locality)

---

### Memory Leaks

**Detection**:
- Monitor RSS over time (should be stable)
- Use tools: valgrind, AddressSanitizer

**Prevention**:
- Use smart pointers (C++: unique_ptr, shared_ptr)
- RAII patterns
- Regular leak testing in CI

---

## Consequences

### Positive

✅ **Cost-Effective**: 16GB RAM supports 10M vectors  
✅ **Predictable**: Fixed memory budget, no surprises  
✅ **Scalable**: Vertical scaling path (32GB → 20M vectors)  
✅ **Cache-Friendly**: 8GB graph cache gives 70-85% hit rate  
✅ **Bounded**: Delta index and caches have hard limits  
✅ **Monitorable**: Clear metrics for tuning  

### Negative

❌ **Requires Tuning**: Cache sizes need per-dataset tuning  
❌ **Cold Start**: Initial queries slower (cache warming)  
❌ **Limited Concurrency**: ~1600 concurrent queries max  
❌ **Memory Pressure**: No headroom for spikes  
❌ **OS Dependency**: Relies on OS page cache effectiveness  

### Mitigation Strategies

- **Tuning**: Provide good defaults, document tuning process
- **Cold Start**: Pre-warm cache on startup (~30s)
- **Concurrency**: Queue excess queries, return 429
- **Memory Pressure**: Monitor proactively, alert at 85%
- **OS Page Cache**: Test on target OS (Linux preferred)

---

## Alternatives Considered

### Alternative 1: Minimal Memory (8GB)

**Configuration**:
- Graph cache: 4GB
- Vector cache: 1GB
- Other: 3GB

**Pros**: Lower cost per node

**Cons**:
- Low cache hit rate (~50%)
- Higher latency (more disk IO)
- Less concurrent capacity

**Decision**: Rejected - poor performance

---

### Alternative 2: Large Memory (64GB)

**Configuration**:
- Load entire index into RAM
- No caching needed

**Pros**: Best performance (no disk IO)

**Cons**:
- Defeats SSD-first design
- 4x cost
- Not scalable to 100M+ vectors

**Decision**: Rejected - violates core principle

---

### Alternative 3: No Application Cache (Rely on OS)

**Approach**: Just use OS page cache, no application-level cache

**Pros**: Simple implementation

**Cons**:
- No control over eviction policy
- Can't pre-warm specific data
- No visibility into hit rates

**Decision**: Rejected - need explicit control

---

## Open Questions

1. **Huge Pages** (deferred)
   - Should we use Linux huge pages (2MB) for better TLB efficiency?
   - Requires testing and tuning

2. **NUMA Awareness** (deferred)
   - On multi-socket servers, allocate memory on local NUMA node?
   - Likely not relevant for v1 (single-socket VMs)

3. **Memory Compression** (deferred)
   - Can we compress cached data (e.g., with zstd)?
   - Trade CPU for memory capacity?

---

## Implementation Checklist

- [ ] Implement LRU cache for graph nodes (with pre-warming)
- [ ] Implement LRU cache for vectors
- [ ] Add memory usage tracking and limits
- [ ] Integrate jemalloc or tcmalloc
- [ ] Add cache hit rate metrics
- [ ] Implement OOM protection (reject queries)
- [ ] Add cache sizing configuration
- [ ] Write cache tuning guide
- [ ] Test cold start behavior
- [ ] Benchmark with different cache sizes

---

## References

- **LRU Cache Implementation**: https://github.com/lamerman/cpp-lru-cache
- **jemalloc**: http://jemalloc.net/
- **Linux Memory Management**: https://www.kernel.org/doc/html/latest/admin-guide/mm/concepts.html
- **DiskANN Paper** (memory budget discussion): https://arxiv.org/abs/1907.10310

---

**Status**: Proposed - Needs team review and testing  
**Next Steps**: Prototype cache implementation, measure hit rates on real data  
**Implementation**: Blocks COMP-004 (Search Engine), affects all components
