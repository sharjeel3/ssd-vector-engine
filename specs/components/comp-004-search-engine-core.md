# Component: Search Engine Core

## Status
- [x] Draft
- [ ] Review
- [ ] Approved
- [ ] Implemented

**Date**: 2025-12-20  
**Owner**: Engineering Team  
**Milestone**: M1 - Core ANN Engine

---

## Purpose

The Search Engine Core executes approximate nearest neighbor (ANN) queries against DiskANN-style indexes. It implements the Vamana graph search algorithm with SSD-optimized access patterns, leveraging memory caching and SIMD-accelerated distance computation to deliver sub-100ms query latency.

**Core Responsibilities**:
1. Execute ANN graph search (Vamana algorithm)
2. Compute distances using SIMD-optimized functions
3. Manage graph and vector caches (per ADR-003)
4. Rank and return top-k results
5. Track query performance metrics
6. Handle concurrent queries efficiently
7. Support multiple distance metrics (cosine, L2, inner product)

---

## Interface

### Public API

```cpp
namespace ssd_vector {
namespace search {

// ============================================================================
// Search Query and Results
// ============================================================================

struct SearchQuery {
    // Query vector
    Vector query_vector;
    
    // Search parameters
    uint32_t top_k = 10;                    // Number of results to return
    uint32_t L = 100;                       // Search list size (from ADR-002)
    
    // Distance metric (must match index)
    enum class Metric {
        COSINE,
        L2,
        INNER_PRODUCT
    } metric = Metric::COSINE;
    
    // Optional: timeout
    std::chrono::milliseconds timeout = std::chrono::milliseconds(100);
};

struct SearchResult {
    IDMapper::ExternalID id;                // External ID
    float distance;                         // Distance to query
    uint32_t internal_id;                   // Internal ID (for debugging)
};

struct SearchResponse {
    std::vector<SearchResult> results;      // Top-k results, sorted by distance
    
    // Query statistics
    struct Stats {
        uint64_t nodes_visited;             // Graph nodes explored
        uint64_t distance_computations;     // Distance calculations performed
        uint64_t cache_hits;                // Cache hits (graph + vector)
        uint64_t cache_misses;              // Cache misses
        double query_time_ms;               // Total query time
        double distance_compute_time_ms;    // Time spent computing distances
        double graph_traversal_time_ms;     // Time spent traversing graph
    } stats;
    
    bool timed_out = false;                 // Whether query timed out
};

// ============================================================================
// Search Engine
// ============================================================================

class SearchEngine {
public:
    struct Config {
        // Search parameters
        uint32_t default_L = 100;           // Default search list size
        uint32_t max_L = 200;               // Maximum allowed L
        
        // Cache configuration (from ADR-003)
        size_t graph_cache_size_mb = 8192;  // 8GB graph cache
        size_t vector_cache_size_mb = 2048; // 2GB vector cache
        
        // Performance
        uint32_t num_threads = 1;           // Concurrent query threads
        bool prefetch_neighbors = true;     // Prefetch graph neighbors
        bool use_simd = true;               // Use SIMD for distances
        
        // Monitoring
        bool collect_stats = true;          // Collect per-query stats
        bool enable_metrics = true;         // Enable metrics collection
    };
    
    // Constructor: load index from directory
    explicit SearchEngine(
        const std::string& index_dir,
        const Config& config = Config()
    );
    
    // Execute single query
    SearchResponse search(const SearchQuery& query);
    
    // Execute batch of queries (parallel)
    std::vector<SearchResponse> search_batch(
        const std::vector<SearchQuery>& queries
    );
    
    // Cache management
    void prewarm_cache();                   // Pre-load hot nodes
    void clear_cache();                     // Clear all caches
    
    struct CacheStats {
        size_t graph_cache_size_mb;
        size_t vector_cache_size_mb;
        double graph_cache_hit_rate;
        double vector_cache_hit_rate;
        uint64_t graph_cache_evictions;
        uint64_t vector_cache_evictions;
    };
    
    CacheStats get_cache_stats() const;
    
    // Index info
    uint64_t vector_count() const;
    uint32_t dimension() const;
    std::string metric() const;
    
    ~SearchEngine();

private:
    Config config_;
    std::unique_ptr<storage::IndexStorage::Index> index_;
    std::unique_ptr<GraphCache> graph_cache_;
    std::unique_ptr<VectorCache> vector_cache_;
    uint32_t entry_point_;                  // Medoid node ID
    
    // Core search algorithm
    SearchResponse search_internal(const SearchQuery& query);
    
    // Distance computation
    float compute_distance(
        const float* vec_a,
        const float* vec_b,
        uint32_t dimension,
        SearchQuery::Metric metric
    );
    
    // Cache access
    const storage::GraphFile::Node* get_graph_node(uint32_t node_id);
    const float* get_vector(uint32_t node_id);
};

// ============================================================================
// Graph Cache (LRU)
// ============================================================================

class GraphCache {
public:
    struct CachedNode {
        uint32_t node_id;
        uint32_t degree;
        std::vector<uint32_t> neighbors;
        uint64_t last_access_time;
    };
    
    explicit GraphCache(size_t capacity_mb);
    
    // Get node (loads from disk if not cached)
    const CachedNode* get(
        uint32_t node_id,
        storage::GraphFile* graph_file
    );
    
    // Pre-warm cache with specific nodes
    void prewarm(
        const std::vector<uint32_t>& node_ids,
        storage::GraphFile* graph_file
    );
    
    // Statistics
    double hit_rate() const;
    uint64_t hits() const { return hits_; }
    uint64_t misses() const { return misses_; }
    uint64_t evictions() const { return evictions_; }
    size_t size_bytes() const;
    
    // Clear cache
    void clear();
    
private:
    size_t capacity_bytes_;
    std::unordered_map<uint32_t, std::shared_ptr<CachedNode>> cache_;
    std::list<uint32_t> lru_list_;
    std::unordered_map<uint32_t, std::list<uint32_t>::iterator> lru_map_;
    
    uint64_t hits_ = 0;
    uint64_t misses_ = 0;
    uint64_t evictions_ = 0;
    
    void evict_lru();
    void touch(uint32_t node_id);
};

// ============================================================================
// Vector Cache (LRU)
// ============================================================================

class VectorCache {
public:
    struct CachedVector {
        uint32_t vector_id;
        std::vector<float> data;
        float norm;                         // Pre-computed norm for cosine
        uint64_t last_access_time;
    };
    
    explicit VectorCache(size_t capacity_mb, uint32_t dimension);
    
    // Get vector (loads from disk if not cached)
    const float* get(
        uint32_t vector_id,
        storage::VectorFile* vector_file
    );
    
    // Pre-warm cache
    void prewarm(
        const std::vector<uint32_t>& vector_ids,
        storage::VectorFile* vector_file
    );
    
    // Statistics
    double hit_rate() const;
    uint64_t hits() const { return hits_; }
    uint64_t misses() const { return misses_; }
    
    // Clear cache
    void clear();
    
private:
    size_t capacity_bytes_;
    uint32_t dimension_;
    std::unordered_map<uint32_t, std::shared_ptr<CachedVector>> cache_;
    std::list<uint32_t> lru_list_;
    std::unordered_map<uint32_t, std::list<uint32_t>::iterator> lru_map_;
    
    uint64_t hits_ = 0;
    uint64_t misses_ = 0;
    
    void evict_lru();
    void touch(uint32_t vector_id);
};

// ============================================================================
// Distance Functions (SIMD Optimized)
// ============================================================================

class DistanceComputer {
public:
    // Function pointer type for distance computation
    using DistanceFunc = float (*)(const float*, const float*, uint32_t);
    
    // Get distance function for metric
    static DistanceFunc get_distance_function(
        SearchQuery::Metric metric,
        bool use_simd = true
    );
    
    // Cosine distance (1 - cosine similarity)
    static float cosine_distance(
        const float* a, 
        const float* b, 
        uint32_t dim
    );
    
    static float cosine_distance_simd(
        const float* a, 
        const float* b, 
        uint32_t dim
    );
    
    // L2 (Euclidean) distance
    static float l2_distance(
        const float* a, 
        const float* b, 
        uint32_t dim
    );
    
    static float l2_distance_simd(
        const float* a, 
        const float* b, 
        uint32_t dim
    );
    
    // Inner product (negative for distance)
    static float inner_product_distance(
        const float* a, 
        const float* b, 
        uint32_t dim
    );
    
    static float inner_product_distance_simd(
        const float* a, 
        const float* b, 
        uint32_t dim
    );
};

} // namespace search
} // namespace ssd_vector
```

---

## Implementation Details

### Core Search Algorithm (Vamana)

```cpp
SearchResponse SearchEngine::search_internal(const SearchQuery& query) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    SearchResponse response;
    response.stats = {};
    
    // Normalize query vector if needed
    Vector normalized_query = query.query_vector;
    if (query.metric == SearchQuery::Metric::COSINE) {
        normalized_query.normalize_l2();
    }
    
    // Initialize candidate set with entry point
    struct Candidate {
        uint32_t node_id;
        float distance;
        
        bool operator<(const Candidate& other) const {
            return distance > other.distance;  // Max heap
        }
    };
    
    std::priority_queue<Candidate> candidates;
    std::unordered_set<uint32_t> visited;
    
    // Get distance function
    auto distance_func = DistanceComputer::get_distance_function(
        query.metric,
        config_.use_simd
    );
    
    // Start from entry point (medoid)
    const float* entry_vector = get_vector(entry_point_);
    float entry_dist = distance_func(
        normalized_query.data(),
        entry_vector,
        dimension()
    );
    
    candidates.push({entry_point_, entry_dist});
    visited.insert(entry_point_);
    response.stats.distance_computations++;
    
    // Keep track of best candidates
    std::vector<Candidate> best_candidates;
    best_candidates.push_back({entry_point_, entry_dist});
    
    // Greedy graph search
    uint32_t iterations = 0;
    const uint32_t max_iterations = query.L;
    
    while (!candidates.empty() && iterations < max_iterations) {
        // Check timeout
        auto now = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            now - start_time
        );
        if (elapsed > query.timeout) {
            response.timed_out = true;
            break;
        }
        
        // Get closest unvisited candidate
        Candidate current = candidates.top();
        candidates.pop();
        iterations++;
        
        // Get neighbors from graph
        const auto* node = get_graph_node(current.node_id);
        response.stats.nodes_visited++;
        
        // Prefetch next level neighbors if enabled
        if (config_.prefetch_neighbors && node->degree > 0) {
            for (uint32_t i = 0; i < std::min(4u, node->degree); i++) {
                uint32_t neighbor_id = node->neighbors[i];
                __builtin_prefetch(get_graph_node(neighbor_id));
                __builtin_prefetch(get_vector(neighbor_id));
            }
        }
        
        // Explore neighbors
        for (uint32_t i = 0; i < node->degree; i++) {
            uint32_t neighbor_id = node->neighbors[i];
            
            // Skip if already visited
            if (visited.count(neighbor_id)) {
                continue;
            }
            visited.insert(neighbor_id);
            
            // Compute distance
            const float* neighbor_vector = get_vector(neighbor_id);
            float dist = distance_func(
                normalized_query.data(),
                neighbor_vector,
                dimension()
            );
            response.stats.distance_computations++;
            
            // Add to candidates
            candidates.push({neighbor_id, dist});
            best_candidates.push_back({neighbor_id, dist});
        }
    }
    
    // Sort by distance and return top-k
    std::sort(
        best_candidates.begin(),
        best_candidates.end(),
        [](const Candidate& a, const Candidate& b) {
            return a.distance < b.distance;
        }
    );
    
    // Convert to search results
    size_t k = std::min(static_cast<size_t>(query.top_k), best_candidates.size());
    response.results.reserve(k);
    
    for (size_t i = 0; i < k; i++) {
        const auto& candidate = best_candidates[i];
        
        SearchResult result;
        result.internal_id = candidate.node_id;
        result.id = index_->metadata->get_external_id(candidate.node_id);
        result.distance = candidate.distance;
        
        response.results.push_back(result);
    }
    
    // Compute timing stats
    auto end_time = std::chrono::high_resolution_clock::now();
    response.stats.query_time_ms = std::chrono::duration<double, std::milli>(
        end_time - start_time
    ).count();
    
    // Cache stats
    response.stats.cache_hits = graph_cache_->hits() + vector_cache_->hits();
    response.stats.cache_misses = graph_cache_->misses() + vector_cache_->misses();
    
    return response;
}
```

---

### Graph Cache Implementation

```cpp
GraphCache::GraphCache(size_t capacity_mb) 
    : capacity_bytes_(capacity_mb * 1024 * 1024) {
}

const GraphCache::CachedNode* GraphCache::get(
    uint32_t node_id,
    storage::GraphFile* graph_file
) {
    // Check if in cache
    auto it = cache_.find(node_id);
    if (it != cache_.end()) {
        hits_++;
        touch(node_id);
        return it->second.get();
    }
    
    // Cache miss - load from disk
    misses_++;
    
    // Load node from storage
    const auto* disk_node = graph_file->get_node(node_id);
    
    // Create cached copy
    auto cached = std::make_shared<CachedNode>();
    cached->node_id = node_id;
    cached->degree = disk_node->degree;
    cached->neighbors.assign(
        disk_node->neighbors,
        disk_node->neighbors + disk_node->degree
    );
    cached->last_access_time = std::chrono::steady_clock::now()
        .time_since_epoch().count();
    
    // Check if we need to evict
    size_t node_size = sizeof(CachedNode) + 
                       cached->neighbors.size() * sizeof(uint32_t);
    
    while (size_bytes() + node_size > capacity_bytes_ && !cache_.empty()) {
        evict_lru();
    }
    
    // Insert into cache
    cache_[node_id] = cached;
    lru_list_.push_front(node_id);
    lru_map_[node_id] = lru_list_.begin();
    
    return cached.get();
}

void GraphCache::evict_lru() {
    if (lru_list_.empty()) return;
    
    // Remove least recently used
    uint32_t evict_id = lru_list_.back();
    lru_list_.pop_back();
    lru_map_.erase(evict_id);
    cache_.erase(evict_id);
    evictions_++;
}

void GraphCache::touch(uint32_t node_id) {
    auto it = lru_map_.find(node_id);
    if (it != lru_map_.end()) {
        // Move to front of LRU list
        lru_list_.erase(it->second);
        lru_list_.push_front(node_id);
        lru_map_[node_id] = lru_list_.begin();
    }
}

void GraphCache::prewarm(
    const std::vector<uint32_t>& node_ids,
    storage::GraphFile* graph_file
) {
    for (uint32_t node_id : node_ids) {
        get(node_id, graph_file);
    }
}

double GraphCache::hit_rate() const {
    uint64_t total = hits_ + misses_;
    return total > 0 ? static_cast<double>(hits_) / total : 0.0;
}

size_t GraphCache::size_bytes() const {
    size_t total = 0;
    for (const auto& [id, node] : cache_) {
        total += sizeof(CachedNode) + node->neighbors.size() * sizeof(uint32_t);
    }
    return total;
}
```

---

### Vector Cache Implementation

```cpp
VectorCache::VectorCache(size_t capacity_mb, uint32_t dimension)
    : capacity_bytes_(capacity_mb * 1024 * 1024),
      dimension_(dimension) {
}

const float* VectorCache::get(
    uint32_t vector_id,
    storage::VectorFile* vector_file
) {
    // Check if in cache
    auto it = cache_.find(vector_id);
    if (it != cache_.end()) {
        hits_++;
        touch(vector_id);
        return it->second->data.data();
    }
    
    // Cache miss - load from disk
    misses_++;
    
    // Load vector from storage
    const float* disk_vector = vector_file->get_vector(vector_id);
    
    // Create cached copy
    auto cached = std::make_shared<CachedVector>();
    cached->vector_id = vector_id;
    cached->data.assign(disk_vector, disk_vector + dimension_);
    
    // Pre-compute norm for cosine distance
    cached->norm = VectorNormalizer::compute_norm_l2(
        cached->data.data(),
        dimension_
    );
    
    cached->last_access_time = std::chrono::steady_clock::now()
        .time_since_epoch().count();
    
    // Check if we need to evict
    size_t vector_size = sizeof(CachedVector) + dimension_ * sizeof(float);
    
    while (size_bytes() + vector_size > capacity_bytes_ && !cache_.empty()) {
        evict_lru();
    }
    
    // Insert into cache
    cache_[vector_id] = cached;
    lru_list_.push_front(vector_id);
    lru_map_[vector_id] = lru_list_.begin();
    
    return cached->data.data();
}

void VectorCache::evict_lru() {
    if (lru_list_.empty()) return;
    
    uint32_t evict_id = lru_list_.back();
    lru_list_.pop_back();
    lru_map_.erase(evict_id);
    cache_.erase(evict_id);
}

void VectorCache::touch(uint32_t vector_id) {
    auto it = lru_map_.find(vector_id);
    if (it != lru_map_.end()) {
        lru_list_.erase(it->second);
        lru_list_.push_front(vector_id);
        lru_map_[vector_id] = lru_list_.begin();
    }
}

double VectorCache::hit_rate() const {
    uint64_t total = hits_ + misses_;
    return total > 0 ? static_cast<double>(hits_) / total : 0.0;
}
```

---

### Distance Functions (SIMD Optimized)

#### Cosine Distance

```cpp
float DistanceComputer::cosine_distance(
    const float* a,
    const float* b,
    uint32_t dim
) {
    float dot = 0.0f;
    float norm_a = 0.0f;
    float norm_b = 0.0f;
    
    for (uint32_t i = 0; i < dim; i++) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    
    float similarity = dot / (std::sqrt(norm_a) * std::sqrt(norm_b));
    return 1.0f - similarity;  // Convert to distance
}

#ifdef __AVX2__
float DistanceComputer::cosine_distance_simd(
    const float* a,
    const float* b,
    uint32_t dim
) {
    __m256 sum_dot = _mm256_setzero_ps();
    __m256 sum_a = _mm256_setzero_ps();
    __m256 sum_b = _mm256_setzero_ps();
    
    uint32_t i = 0;
    for (; i + 8 <= dim; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        
        // Dot product
        sum_dot = _mm256_fmadd_ps(va, vb, sum_dot);
        
        // Norms
        sum_a = _mm256_fmadd_ps(va, va, sum_a);
        sum_b = _mm256_fmadd_ps(vb, vb, sum_b);
    }
    
    // Horizontal sum
    float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
    
    alignas(32) float temp_dot[8];
    alignas(32) float temp_a[8];
    alignas(32) float temp_b[8];
    
    _mm256_store_ps(temp_dot, sum_dot);
    _mm256_store_ps(temp_a, sum_a);
    _mm256_store_ps(temp_b, sum_b);
    
    for (int j = 0; j < 8; j++) {
        dot += temp_dot[j];
        norm_a += temp_a[j];
        norm_b += temp_b[j];
    }
    
    // Handle remaining elements
    for (; i < dim; i++) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    
    float similarity = dot / (std::sqrt(norm_a) * std::sqrt(norm_b));
    return 1.0f - similarity;
}
#endif
```

#### L2 Distance

```cpp
float DistanceComputer::l2_distance(
    const float* a,
    const float* b,
    uint32_t dim
) {
    float sum = 0.0f;
    
    for (uint32_t i = 0; i < dim; i++) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    
    return std::sqrt(sum);
}

#ifdef __AVX2__
float DistanceComputer::l2_distance_simd(
    const float* a,
    const float* b,
    uint32_t dim
) {
    __m256 sum_vec = _mm256_setzero_ps();
    
    uint32_t i = 0;
    for (; i + 8 <= dim; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 diff = _mm256_sub_ps(va, vb);
        sum_vec = _mm256_fmadd_ps(diff, diff, sum_vec);  // diff^2 + sum
    }
    
    // Horizontal sum
    alignas(32) float temp[8];
    _mm256_store_ps(temp, sum_vec);
    
    float sum = 0.0f;
    for (int j = 0; j < 8; j++) {
        sum += temp[j];
    }
    
    // Handle remaining elements
    for (; i < dim; i++) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    
    return std::sqrt(sum);
}
#endif
```

#### Inner Product Distance

```cpp
float DistanceComputer::inner_product_distance(
    const float* a,
    const float* b,
    uint32_t dim
) {
    float dot = 0.0f;
    
    for (uint32_t i = 0; i < dim; i++) {
        dot += a[i] * b[i];
    }
    
    return -dot;  // Negate for distance (higher is worse)
}

#ifdef __AVX2__
float DistanceComputer::inner_product_distance_simd(
    const float* a,
    const float* b,
    uint32_t dim
) {
    __m256 sum_vec = _mm256_setzero_ps();
    
    uint32_t i = 0;
    for (; i + 8 <= dim; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        sum_vec = _mm256_fmadd_ps(va, vb, sum_vec);
    }
    
    // Horizontal sum
    alignas(32) float temp[8];
    _mm256_store_ps(temp, sum_vec);
    
    float sum = 0.0f;
    for (int j = 0; j < 8; j++) {
        sum += temp[j];
    }
    
    // Handle remaining elements
    for (; i < dim; i++) {
        sum += a[i] * b[i];
    }
    
    return -sum;
}
#endif
```

---

### Batch Search (Parallel)

```cpp
std::vector<SearchResponse> SearchEngine::search_batch(
    const std::vector<SearchQuery>& queries
) {
    std::vector<SearchResponse> responses(queries.size());
    
    // Execute queries in parallel
    #pragma omp parallel for num_threads(config_.num_threads)
    for (size_t i = 0; i < queries.size(); i++) {
        responses[i] = search(queries[i]);
    }
    
    return responses;
}
```

---

### Cache Pre-warming

```cpp
void SearchEngine::prewarm_cache() {
    // Pre-warm with entry point and its neighbors
    std::vector<uint32_t> prewarm_nodes;
    prewarm_nodes.push_back(entry_point_);
    
    // Get entry point's neighbors
    const auto* entry_node = index_->graph->get_node(entry_point_);
    for (uint32_t i = 0; i < entry_node->degree; i++) {
        prewarm_nodes.push_back(entry_node->neighbors[i]);
    }
    
    // Pre-warm graph cache
    graph_cache_->prewarm(prewarm_nodes, index_->graph.get());
    
    // Pre-warm vector cache
    vector_cache_->prewarm(prewarm_nodes, index_->vectors.get());
    
    LOG_INFO("Cache pre-warmed with " << prewarm_nodes.size() << " nodes");
}
```

---

## Performance Requirements

### Latency Targets

| Metric | Target | Conditions |
|--------|--------|------------|
| **p50** | < 10ms | Single query, warm cache |
| **p95** | < 30ms | Single query, warm cache |
| **p99** | < 50ms | Single query, warm cache |
| **p99.9** | < 100ms | Single query, cold cache |

**Measurement**: End-to-end from query submission to results

---

### Throughput Targets

| Configuration | Target QPS | Notes |
|---------------|------------|-------|
| **Single thread** | 100-200 | Sequential queries |
| **4 threads** | 300-500 | Parallel queries |
| **8 threads** | 500-1000 | Parallel queries |

**Bottleneck**: Disk random read IOPS (~20-50 reads per query)

---

### Recall Targets

| Metric | Target | Notes |
|--------|--------|-------|
| **Recall@10** | ≥ 95% | L=100, R=32 |
| **Recall@100** | ≥ 98% | L=100, R=32 |

**Measurement**: Compared to brute-force ground truth on test set

---

### Cache Performance

| Metric | Target | Notes |
|--------|--------|-------|
| **Graph cache hit rate** | > 70% | After warm-up |
| **Vector cache hit rate** | > 20% | Opportunistic |
| **Cache warm-up time** | < 30s | Pre-loading hot nodes |

---

### Resource Usage

| Resource | Target | Notes |
|----------|--------|-------|
| **Memory per query** | ~1.2MB | Working set |
| **CPU per query** | ~5-20ms | Single core |
| **Disk reads per query** | 20-50 | Random reads |

---

## Error Handling

### Search Exceptions

```cpp
class SearchException : public std::exception {
public:
    enum class ErrorCode {
        INVALID_QUERY,
        DIMENSION_MISMATCH,
        TIMEOUT,
        INDEX_CORRUPTED,
        OUT_OF_MEMORY
    };
    
    SearchException(ErrorCode code, const std::string& message)
        : code_(code), message_(message) {}
    
    ErrorCode code() const { return code_; }
    const char* what() const noexcept override { return message_.c_str(); }
    
private:
    ErrorCode code_;
    std::string message_;
};
```

### Error Recovery

| Error | Detection | Recovery |
|-------|-----------|----------|
| **Invalid query** | Dimension check | Return error immediately |
| **Timeout** | Time check in loop | Return partial results |
| **Corrupted node** | Validation | Skip node, continue search |
| **Cache full** | Memory check | Evict LRU, continue |
| **Disk read error** | IO exception | Retry once, then fail |

---

## Testing Strategy

### Unit Tests

```cpp
TEST(SearchEngine, SingleQuery) {
    auto engine = create_test_engine();
    
    Vector query = create_random_vector(1536);
    SearchQuery sq;
    sq.query_vector = query;
    sq.top_k = 10;
    
    auto response = engine->search(sq);
    
    EXPECT_EQ(response.results.size(), 10);
    EXPECT_FALSE(response.timed_out);
    EXPECT_GT(response.stats.nodes_visited, 0);
}

TEST(SearchEngine, ResultsAreSorted) {
    auto engine = create_test_engine();
    auto response = engine->search(create_test_query());
    
    // Verify results are sorted by distance
    for (size_t i = 1; i < response.results.size(); i++) {
        EXPECT_LE(response.results[i-1].distance, 
                  response.results[i].distance);
    }
}

TEST(SearchEngine, DimensionMismatch) {
    auto engine = create_test_engine();  // Expects 1536D
    
    Vector wrong_dim = create_random_vector(768);
    SearchQuery sq;
    sq.query_vector = wrong_dim;
    
    EXPECT_THROW(engine->search(sq), SearchException);
}

TEST(GraphCache, LRUEviction) {
    GraphCache cache(1);  // 1MB cache
    
    // Fill cache beyond capacity
    for (uint32_t i = 0; i < 10000; i++) {
        cache.get(i, graph_file.get());
    }
    
    EXPECT_GT(cache.evictions(), 0);
    EXPECT_LE(cache.size_bytes(), 1024 * 1024);
}

TEST(VectorCache, HitRate) {
    VectorCache cache(100, 1536);  // 100MB cache
    
    // Access same vectors multiple times
    for (int iter = 0; iter < 10; iter++) {
        for (uint32_t i = 0; i < 100; i++) {
            cache.get(i, vector_file.get());
        }
    }
    
    // Should have high hit rate (90%+)
    EXPECT_GT(cache.hit_rate(), 0.9);
}

TEST(DistanceComputer, CosineDistance) {
    float a[] = {1.0f, 0.0f, 0.0f};
    float b[] = {0.0f, 1.0f, 0.0f};
    
    float dist = DistanceComputer::cosine_distance(a, b, 3);
    
    EXPECT_NEAR(dist, 1.0f, 1e-6f);  // Orthogonal vectors
}

TEST(DistanceComputer, SIMDMatchesScalar) {
    auto a = create_random_vector(1536);
    auto b = create_random_vector(1536);
    
    float scalar = DistanceComputer::cosine_distance(
        a.data(), b.data(), 1536
    );
    
    float simd = DistanceComputer::cosine_distance_simd(
        a.data(), b.data(), 1536
    );
    
    EXPECT_NEAR(scalar, simd, 1e-5f);
}
```

---

### Integration Tests

```cpp
TEST(SearchEngine, RecallVsBruteForce) {
    auto engine = create_test_engine();
    auto test_queries = load_test_queries(100);
    auto ground_truth = compute_brute_force_ground_truth(test_queries);
    
    // Run ANN search
    double total_recall = 0.0;
    for (size_t i = 0; i < test_queries.size(); i++) {
        auto response = engine->search(test_queries[i]);
        double recall = compute_recall(response.results, ground_truth[i], 10);
        total_recall += recall;
    }
    
    double avg_recall = total_recall / test_queries.size();
    EXPECT_GE(avg_recall, 0.95);  // 95% recall target
}

TEST(SearchEngine, ConcurrentQueries) {
    auto engine = create_test_engine();
    
    // Launch 100 concurrent queries
    std::vector<std::future<SearchResponse>> futures;
    for (int i = 0; i < 100; i++) {
        futures.push_back(std::async(std::launch::async, [&]() {
            return engine->search(create_random_query());
        }));
    }
    
    // Collect results
    for (auto& f : futures) {
        auto response = f.get();
        EXPECT_GT(response.results.size(), 0);
    }
}

TEST(SearchEngine, CachePrewarming) {
    auto engine = create_test_engine();
    
    // Clear cache
    engine->clear_cache();
    
    // First query (cold cache)
    auto cold_response = engine->search(create_test_query());
    double cold_time = cold_response.stats.query_time_ms;
    
    // Pre-warm cache
    engine->prewarm_cache();
    
    // Second query (warm cache)
    auto warm_response = engine->search(create_test_query());
    double warm_time = warm_response.stats.query_time_ms;
    
    // Warm should be faster
    EXPECT_LT(warm_time, cold_time * 0.8);
}
```

---

### Performance Tests

```cpp
BENCHMARK(SearchEngine, QueryLatency) {
    auto engine = create_benchmark_engine();
    auto queries = create_benchmark_queries(1000);
    
    auto start = high_resolution_clock::now();
    for (const auto& query : queries) {
        auto response = engine->search(query);
        benchmark::DoNotOptimize(response);
    }
    auto end = high_resolution_clock::now();
    
    auto avg_ms = duration_cast<milliseconds>(end - start).count() / 1000.0;
    
    // Target: < 10ms average
    EXPECT_LT(avg_ms, 10.0);
}

BENCHMARK(SearchEngine, Throughput) {
    auto engine = create_benchmark_engine();
    
    std::atomic<uint64_t> query_count{0};
    auto start = high_resolution_clock::now();
    
    // Run for 10 seconds with 8 threads
    #pragma omp parallel num_threads(8)
    {
        while (duration_cast<seconds>(
            high_resolution_clock::now() - start
        ).count() < 10) {
            auto query = create_random_query();
            auto response = engine->search(query);
            query_count++;
        }
    }
    
    double qps = query_count.load() / 10.0;
    
    // Target: > 500 QPS with 8 threads
    EXPECT_GT(qps, 500.0);
}

BENCHMARK(DistanceComputer, SIMDSpeedup) {
    auto a = create_random_vector(1536);
    auto b = create_random_vector(1536);
    
    // Scalar
    auto start = high_resolution_clock::now();
    for (int i = 0; i < 1000000; i++) {
        float dist = DistanceComputer::cosine_distance(a.data(), b.data(), 1536);
        benchmark::DoNotOptimize(dist);
    }
    auto scalar_time = duration_cast<nanoseconds>(
        high_resolution_clock::now() - start
    ).count();
    
    // SIMD
    start = high_resolution_clock::now();
    for (int i = 0; i < 1000000; i++) {
        float dist = DistanceComputer::cosine_distance_simd(a.data(), b.data(), 1536);
        benchmark::DoNotOptimize(dist);
    }
    auto simd_time = duration_cast<nanoseconds>(
        high_resolution_clock::now() - start
    ).count();
    
    double speedup = static_cast<double>(scalar_time) / simd_time;
    
    // Target: 2x speedup with AVX2
    EXPECT_GT(speedup, 1.8);
}
```

---

## Dependencies

### Internal Dependencies
- **ADR-001**: Storage format (file access)
- **ADR-002**: DiskANN algorithm (search implementation)
- **ADR-003**: Memory budget (cache sizing)
- **COMP-002**: Storage layer (loading indexes)
- **COMP-003**: Vector format (vector operations, ID mapping)

### External Dependencies
- **C++ Standard Library**: STL containers, algorithms
- **AVX2/AVX-512**: SIMD distance computation (optional but recommended)
- **OpenMP**: Parallel batch queries (optional)
- **Threading**: std::thread for concurrent queries

---

## Configuration

```cpp
struct SearchEngineConfig {
    // Algorithm parameters
    uint32_t default_L = 100;           // Default search list size
    uint32_t max_L = 200;               // Maximum L (prevent abuse)
    uint32_t default_top_k = 10;        // Default k
    
    // Cache configuration
    size_t graph_cache_mb = 8192;       // 8GB
    size_t vector_cache_mb = 2048;      // 2GB
    bool prewarm_on_load = true;        // Auto pre-warm
    
    // Performance
    uint32_t num_query_threads = 4;     // Concurrent query threads
    bool use_simd = true;               // Enable SIMD
    bool prefetch_neighbors = true;     // Prefetch optimization
    
    // Timeouts
    std::chrono::milliseconds default_timeout = std::chrono::milliseconds(100);
    std::chrono::milliseconds max_timeout = std::chrono::milliseconds(1000);
    
    // Monitoring
    bool collect_stats = true;
    bool enable_metrics = true;
    uint32_t metrics_report_interval_sec = 60;
};
```

---

## Monitoring and Metrics

### Key Metrics

```cpp
struct SearchMetrics {
    // Latency (ms)
    Histogram query_latency_p50;
    Histogram query_latency_p95;
    Histogram query_latency_p99;
    
    // Throughput
    Counter queries_per_second;
    Counter total_queries;
    
    // Cache
    Gauge graph_cache_hit_rate;
    Gauge vector_cache_hit_rate;
    Counter cache_evictions;
    
    // Algorithm
    Histogram nodes_visited_per_query;
    Histogram distance_computations_per_query;
    
    // Errors
    Counter timeout_count;
    Counter error_count;
};
```

### Logging

```
[INFO] Query executed: id=query_123, latency=8.5ms, nodes_visited=42, top_k=10
[INFO] Cache stats: graph_hit_rate=78.5%, vector_hit_rate=23.2%
[WARN] Query timeout: id=query_456, elapsed=105ms
[ERROR] Corrupted graph node: node_id=12345
```

---

## Future Enhancements

1. **Filtered Search** (Milestone 3)
   - Apply metadata filters during search
   - Pre-filtering vs post-filtering strategies

2. **Query Result Caching** (v1.1)
   - Cache results for repeated queries
   - Invalidation on index updates

3. **GPU Acceleration** (v2)
   - GPU-based distance computation
   - Batch query processing on GPU

4. **Approximate Filtering** (v2)
   - Bloom filters for early filtering
   - Reduce distance computations

5. **Multi-Vector Queries** (v2)
   - Query with multiple vectors
   - Aggregation strategies (avg, max, fusion)

---

## Open Questions

1. **Adaptive L**: Should L adjust based on query difficulty? (Deferred to v1.1)
2. **Early Termination**: Stop search when confident? (Deferred)
3. **Query Scheduler**: Priority queuing for queries? (Deferred to v2)

---

## References

- DiskANN Paper: https://arxiv.org/abs/1907.10310
- Vamana Algorithm: Section 3 of DiskANN paper
- Intel Intrinsics Guide: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/
- LRU Cache Design: https://en.wikipedia.org/wiki/Cache_replacement_policies#LRU

---

**Status**: Draft - Ready for review  
**Next Steps**: Team review, prototype search algorithm  
**Blocks**: None (completes Milestone 1 specs)  
**Blocked By**: COMP-001 (needs built indexes), COMP-002 (needs storage layer), COMP-003 (needs vector utilities)
