# Architecture Decision: DiskANN Algorithm Selection and Tuning Parameters

## Status
- [x] Proposed
- [ ] Accepted
- [ ] Rejected
- [ ] Deprecated

**Date**: 2025-12-20  
**Deciders**: Engineering Team  
**Related Specs**: COMP-001, COMP-004, ADR-001

---

## Context

The SSD Vector Engine needs an Approximate Nearest Neighbor (ANN) algorithm optimized for SSD-resident indexes. The algorithm must balance:

1. **Search Quality**: High recall with minimal false negatives
2. **Query Latency**: Predictable, sub-100ms p99 response times
3. **Memory Efficiency**: Minimal RAM usage (cache only)
4. **SSD Efficiency**: Minimize random seeks, prefer sequential reads
5. **Build Time**: Reasonable index construction time
6. **Scalability**: Support 10M+ vectors in v1

### Why Not In-Memory ANN?

In-memory algorithms (HNSW, IVF-PQ) require loading entire indexes into RAM:
- **10M vectors × 1536D × 4 bytes** = 61.4GB RAM (vectors only)
- Cost-prohibitive at scale
- Doesn't leverage cheap SSD storage

### Why DiskANN?

DiskANN (Microsoft Research, 2019) is designed for SSD-resident ANN:
- Graph-based navigation (like HNSW)
- Optimized for disk access patterns
- Small memory footprint (cache + working set only)
- Proven at scale (Microsoft Bing, Azure Cognitive Search)

---

## Decision

We will implement **DiskANN-style Vamana graph algorithm** as the core ANN engine with the following configuration.

### Core Algorithm: Vamana Graph

**Key Characteristics**:
- **Graph-based**: Each vector is a node with edges to approximate neighbors
- **Greedy search**: Start at entry point, iteratively move to closer neighbors
- **Pruning strategy**: Robust prune procedure ensures graph quality
- **SSD-optimized**: Graph layout minimizes disk seeks

### Algorithm Overview

```
Build Phase:
1. Initialize random graph
2. For each node in random order:
   a. Find approximate neighbors via graph search
   b. Add edges using robust prune
   c. Update reverse edges
3. Optimize graph layout for sequential disk access

Search Phase:
1. Start at pre-computed entry point
2. Maintain candidate list (priority queue)
3. Iteratively:
   a. Pop closest unvisited node
   b. Load its neighbors from disk
   c. Evaluate distances, update candidates
4. Return top-k candidates
```

---

## Tuning Parameters

### Build Parameters

#### R (Max Degree)
**Definition**: Maximum number of edges per node in the graph

**v1 Value**: `R = 32`

**Rationale**:
- Balances search quality vs disk IO
- Each node access loads ~32 neighbor IDs = 128 bytes
- Fits in single disk IO block (4KB)
- Typical range: 16-64, we choose conservative middle

**Impact**:
- Higher R → better recall, more disk IO, larger index
- Lower R → faster search, worse recall, smaller index

**Tuning**: Monitor recall@10, adjust if < 95%

---

#### L (Build List Size)
**Definition**: Size of candidate list during graph construction

**v1 Value**: `L_build = 100`

**Rationale**:
- Controls build quality (higher = better graph)
- Larger than R to allow selective pruning
- Typical range: 50-200, we choose middle-high
- Microsoft DiskANN uses 100-150

**Impact**:
- Higher L → better graph quality, slower build
- Lower L → faster build, worse recall

**Tuning**: If build time is too long, decrease to 75

---

#### α (Alpha - Pruning Parameter)
**Definition**: Controls aggressive vs conservative pruning

**v1 Value**: `α = 1.2`

**Rationale**:
- Microsoft DiskANN default
- Values: 1.0 (aggressive) to 1.5 (conservative)
- 1.2 balances edge diversity and quality

**Impact**:
- Higher α → more diverse edges, better robustness
- Lower α → more greedy edges, faster search, less robust

**Tuning**: Increase to 1.3-1.4 if recall drops on difficult queries

---

### Search Parameters

#### L (Search List Size)
**Definition**: Size of candidate list during search

**v1 Value**: `L_search = 100` (default), configurable per query

**Rationale**:
- Larger than k (top-k) to allow reranking
- Typical: 2-10x the requested k
- For k=10, L=100 gives 10x buffer

**Impact**:
- Higher L → better recall, more disk reads, higher latency
- Lower L → faster search, lower recall

**Tuning**: Expose as query parameter (e.g., `L_search = 50-200`)

**Example**:
```python
# Fast, lower recall
results = client.query(vector=emb, top_k=10, search_quality="fast")  # L=50

# Default
results = client.query(vector=emb, top_k=10)  # L=100

# High quality
results = client.query(vector=emb, top_k=10, search_quality="high")  # L=200
```

---

#### Beam Width (Optional)
**Definition**: Number of parallel search paths

**v1 Value**: `beam_width = 1` (single-path greedy search)

**Rationale**:
- Single-path is simplest and works well
- Multi-path (beam search) can improve recall slightly
- Deferred to v1.1 or v2.0

---

### Memory Parameters

#### Working Set Size
**Definition**: Amount of RAM used during search

**v1 Value**: `working_set = 512MB` per query thread

**Rationale**:
- Candidate list: ~100 entries × (1536D × 4B + metadata) ≈ 0.6MB
- Prefetch buffer: ~4MB (next few nodes)
- Graph cache: ~507MB (LRU cache of hot nodes)

**Impact**:
- More RAM → better cache hit rate → lower latency
- Less RAM → more disk IO → higher latency

**Tuning**: Monitor cache hit rate, adjust if < 80%

---

#### Graph Cache Size
**Definition**: LRU cache of recently accessed graph nodes

**v1 Value**: `cache_size = 100,000 nodes`

**Rationale**:
- At ~32 neighbors × 4 bytes = 128 bytes per node
- 100k nodes ≈ 12.8MB cache
- Hot nodes stay in cache across queries
- Typical cache hit rate: 60-80%

**Impact**:
- Larger cache → higher hit rate → lower latency
- Smaller cache → more cache misses → more disk IO

**Tuning**: Increase to 500k nodes (64MB) if memory available

---

## Distance Metrics

### Supported Metrics (v1)

#### 1. Cosine Similarity (Primary)
**Formula**: `similarity = (A · B) / (||A|| × ||B||)`

**Use Case**: Normalized embeddings (most common in ML)

**Implementation**: 
- Precompute and store vector norms in metadata
- Distance = 1 - cosine_similarity
- Use SIMD for dot product computation

**Rationale**: Most embedding models output normalized vectors

---

#### 2. L2 (Euclidean) Distance
**Formula**: `distance = sqrt(Σ(A_i - B_i)²)`

**Use Case**: Geometric embeddings

**Implementation**:
- SIMD optimized squared distance
- Optional: skip sqrt for comparison (squared L2)

**Rationale**: Second most common metric

---

#### 3. Inner Product (Negative)
**Formula**: `distance = -(A · B)`

**Use Case**: Unnormalized embeddings, learned similarities

**Implementation**: Negate dot product result

**Rationale**: Some models optimize for inner product

---

### Not Supported (v1)

- Hamming distance (binary vectors) - deferred to v2
- Angular distance - use cosine instead
- Custom distance functions - deferred to v2

---

## Performance Targets

### Latency Targets (Single Query)

| Metric | Target | Measurement |
|--------|--------|-------------|
| **p50** | < 10ms | Median query latency |
| **p95** | < 30ms | 95th percentile |
| **p99** | < 50ms | 99th percentile |
| **p99.9** | < 100ms | 99.9th percentile |

**Conditions**: 
- Single query thread
- Cold cache (worst case)
- 10M vectors, 1536D
- k=10

**Measurement**: Use high-resolution timers, measure end-to-end including disk IO

---

### Throughput Targets

| Metric | Target | Notes |
|--------|--------|-------|
| **Single Thread** | 100-200 QPS | Limited by disk latency |
| **Multi-Thread (8 cores)** | 500-1000 QPS | Parallelism + cache sharing |

**Limiting Factor**: Disk random read IOPS
- NVMe SSD: ~500k IOPS
- Each query: ~20-50 random reads
- Theoretical max: ~10k-25k QPS (parallel)

---

### Recall Targets

**Definition**: Recall@k = (# correct results in top-k) / k

| Metric | Target | Measurement |
|--------|--------|-------------|
| **Recall@10** | ≥ 95% | Compare to brute force |
| **Recall@100** | ≥ 98% | Compare to brute force |

**Measurement**:
- Ground truth: Brute force search on test set
- Test set: 1000 random queries
- Report: Average recall across queries

**Tuning**: If recall < 95%, increase L_search or R

---

### Build Time Targets

| Dataset Size | Target Build Time | Throughput |
|--------------|-------------------|------------|
| 1M vectors | < 30 minutes | ~550 vectors/sec |
| 10M vectors | < 5 hours | ~550 vectors/sec |

**Rationale**:
- Build is offline, doesn't need to be real-time
- ~10 minutes per million vectors is acceptable
- Parallelization can speed up significantly

**Measurement**: Wall-clock time from input to published index

---

## Optimization Strategies

### 1. SIMD Acceleration

**Distance Computation**: Use AVX2/AVX-512 for vector operations

```cpp
// Example: SIMD dot product
float dot_product_simd(const float* a, const float* b, size_t dim) {
    __m256 sum = _mm256_setzero_ps();
    for (size_t i = 0; i < dim; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        sum = _mm256_fmadd_ps(va, vb, sum);  // fused multiply-add
    }
    // Horizontal sum to get final result
    return horizontal_sum(sum);
}
```

**Expected Speedup**: 4-8x vs scalar code

---

### 2. Prefetching

**Strategy**: Prefetch next graph nodes while evaluating current

```cpp
void search_with_prefetch(vector query, int k) {
    priority_queue<Node> candidates;
    
    while (!candidates.empty()) {
        Node current = candidates.pop();
        
        // Prefetch neighbors before access
        for (auto neighbor_id : current.neighbors) {
            __builtin_prefetch(get_node_ptr(neighbor_id));
        }
        
        // Evaluate neighbors (prefetched data arrives)
        for (auto neighbor_id : current.neighbors) {
            evaluate_and_add(neighbor_id, candidates);
        }
    }
}
```

**Expected Impact**: 10-20% latency reduction

---

### 3. Graph Layout Optimization

**Strategy**: Order nodes on disk to maximize sequential access

**Approach**:
- Cluster related nodes together (locality-preserving ordering)
- Use BFS or graph partitioning
- Minimize seeks during search

**Expected Impact**: 20-30% disk IO reduction

---

### 4. Sector-Aligned IO

**Strategy**: Align graph nodes to disk sector boundaries (4KB)

**Rationale**:
- Each disk read fetches entire 4KB sector
- Pack multiple small nodes per sector
- Reduce wasted reads

---

## Error Handling

### Build Failures

| Error | Cause | Action |
|-------|-------|--------|
| **OutOfMemory** | L_build too large | Reduce L_build, add swap space |
| **DiskFull** | Insufficient space | Increase disk, fail gracefully |
| **InvalidInput** | Malformed vectors | Validate input, report bad records |
| **Timeout** | Build too slow | Increase timeout, optimize build |

### Search Failures

| Error | Cause | Action |
|-------|-------|--------|
| **CorruptedIndex** | Bad disk read | Verify checksum, reload index |
| **InvalidQuery** | Wrong dimension | Validate query dimension |
| **TimeoutExceeded** | Slow query | Kill query, return partial results |
| **DiskError** | IO failure | Retry, failover to replica |

---

## Monitoring Metrics

### Key Metrics to Track

| Metric | Description | Target |
|--------|-------------|--------|
| **query_latency_p99** | 99th percentile latency | < 50ms |
| **recall_at_10** | Search quality | ≥ 95% |
| **disk_reads_per_query** | Disk IO efficiency | < 50 reads |
| **cache_hit_rate** | Memory efficiency | > 70% |
| **build_time** | Index build duration | < 5h for 10M |

### Alerting Thresholds

- **Critical**: p99 latency > 100ms
- **Warning**: recall@10 < 95%
- **Warning**: cache hit rate < 60%
- **Info**: build time increases > 20%

---

## Consequences

### Positive

✅ **SSD-Optimized**: Designed for disk-resident indexes  
✅ **Proven at Scale**: Used in production by Microsoft  
✅ **Low Memory**: RAM only for cache, not full index  
✅ **Predictable Latency**: Graph navigation has bounded IO  
✅ **High Recall**: 95%+ recall achievable with tuning  
✅ **Parallelizable**: Multiple queries run concurrently  
✅ **Well-Documented**: Published algorithm with reference implementation  

### Negative

❌ **Complexity**: More complex than simple algorithms (IVF)  
❌ **Build Time**: Graph construction is time-intensive  
❌ **Tuning Required**: Parameters need dataset-specific tuning  
❌ **Cold Start**: First queries slower (cold cache)  
❌ **Disk Dependency**: Requires fast SSD (NVMe recommended)  

### Mitigation Strategies

- **Complexity**: Use reference implementation initially, iterate
- **Build Time**: Parallelize build, run offline
- **Tuning**: Provide good defaults, document tuning guide
- **Cold Start**: Pre-warm cache on startup
- **Disk**: Mandate NVMe SSD for production deployments

---

## Alternatives Considered

### Alternative 1: HNSW (Hierarchical Navigable Small World)

**Algorithm**: Multi-layer graph, in-memory

**Pros**: 
- Excellent recall (98%+)
- Fast queries (< 1ms in memory)
- Well-established (faiss, hnswlib)

**Cons**:
- Requires full index in RAM
- 10M vectors × 1536D = 61GB+ RAM
- Cost-prohibitive at scale

**Decision**: Rejected - doesn't meet SSD-first requirement

---

### Alternative 2: IVF (Inverted File Index)

**Algorithm**: K-means clustering, search only relevant clusters

**Pros**:
- Simpler than graph methods
- Good for SSD (cluster-based access)
- Fast build time

**Cons**:
- Lower recall than graph methods
- Sensitive to data distribution
- Requires tuning nlist (# clusters)

**Decision**: Rejected - recall concerns, less robust

---

### Alternative 3: Product Quantization (PQ)

**Algorithm**: Compress vectors, search compressed space

**Pros**:
- Massive memory savings (8-16x)
- Fast distance computation
- Good for memory-constrained scenarios

**Cons**:
- Lossy compression (recall degradation)
- Complexity of quantization
- Not SSD-optimized (still in-memory)

**Decision**: Consider for v2 as compression layer on top of DiskANN

---

### Alternative 4: LSH (Locality Sensitive Hashing)

**Algorithm**: Hash vectors, search in hash buckets

**Pros**:
- Simple implementation
- Sub-linear search complexity

**Cons**:
- Lower recall than graph methods
- Sensitive to hash function choice
- Not state-of-the-art

**Decision**: Rejected - inferior to modern graph methods

---

## Open Questions

1. **Multi-Vector Search** (deferred)
   - How to search with multiple query vectors?
   - Average, max, or ranked fusion?

2. **Dynamic Updates** (deferred to Milestone 2)
   - Can we update graph without full rebuild?
   - Incremental graph construction?

3. **Quantization** (deferred to v2)
   - Should we add PQ compression?
   - At what scale does it become necessary?

4. **GPU Acceleration** (deferred)
   - Can graph search benefit from GPU?
   - Or is it too IO-bound?

---

## Implementation Checklist

- [ ] Integrate or implement Vamana graph builder
- [ ] Implement greedy graph search with configurable L
- [ ] Add SIMD optimizations for distance computation
- [ ] Implement LRU cache for graph nodes
- [ ] Add prefetching for neighbor nodes
- [ ] Implement all three distance metrics (cosine, L2, inner product)
- [ ] Create benchmark suite (latency, recall, throughput)
- [ ] Tune parameters on test datasets
- [ ] Document tuning guidelines
- [ ] Add monitoring and metrics collection

---

## References

- **DiskANN Paper**: Subramanya et al., "DiskANN: Fast Accurate Billion-point Nearest Neighbor Search on a Single Node", NeurIPS 2019
  - https://arxiv.org/abs/1907.10310
- **Vamana Algorithm**: Jayaram Subramanya et al., "Filtered-DiskANN", 2021
- **Microsoft DiskANN GitHub**: https://github.com/microsoft/DiskANN
- **SIMD Distance Functions**: https://github.com/facebookresearch/faiss (reference implementations)

---

**Status**: Proposed - Needs team review and benchmarking  
**Next Steps**: Prototype on small dataset, measure recall/latency, finalize parameters  
**Implementation**: Blocks COMP-001 (Index Builder), COMP-004 (Search Engine)
