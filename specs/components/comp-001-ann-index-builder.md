# Component: ANN Index Builder

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

The ANN Index Builder is responsible for constructing DiskANN-style Vamana graph indexes from input vector datasets. It takes raw vector data and produces an optimized, disk-resident index structure that enables efficient nearest-neighbor search.

**Core Responsibilities**:
1. Ingest vector data from various input formats
2. Build Vamana graph structure using DiskANN algorithm
3. Optimize graph layout for sequential disk access
4. Serialize index to storage format (per ADR-001)
5. Validate index correctness and quality
6. Generate index metadata and statistics

---

## Interface

### Public API

```cpp
namespace ssd_vector {

class IndexBuilder {
public:
    // Configuration for index building
    struct Config {
        // Build parameters (from ADR-002)
        uint32_t R = 32;          // Max degree (edges per node)
        uint32_t L_build = 100;   // Build list size
        float alpha = 1.2f;       // Pruning parameter
        
        // Distance metric
        enum Metric {
            COSINE,
            L2,
            INNER_PRODUCT
        } metric = COSINE;
        
        // Resource limits
        size_t max_memory_gb = 8;     // Max RAM during build
        uint32_t num_threads = 4;      // Parallel build threads
        
        // Input/output
        std::string input_path;        // Input vectors file
        std::string output_dir;        // Output index directory
        
        // Optimization
        bool optimize_layout = true;   // Graph layout optimization
        uint32_t random_seed = 42;     // Deterministic builds
    };
    
    // Constructor
    explicit IndexBuilder(const Config& config);
    
    // Build index from input vectors
    // Returns: Index statistics (build time, node count, etc.)
    struct BuildStats {
        uint64_t vector_count;
        uint32_t dimension;
        double build_time_seconds;
        double avg_degree;
        size_t index_size_bytes;
        std::string index_version;
    };
    
    BuildStats build();
    
    // Build from in-memory vectors (for testing)
    BuildStats build_from_memory(
        const float* vectors,
        uint64_t count,
        uint32_t dimension
    );
    
    // Validate built index (check graph connectivity, etc.)
    bool validate_index(const std::string& index_dir);
    
    // Get progress information during build
    struct Progress {
        uint64_t vectors_processed;
        uint64_t total_vectors;
        double percent_complete;
        double estimated_time_remaining_seconds;
    };
    
    Progress get_progress() const;
    
private:
    Config config_;
    std::atomic<uint64_t> progress_;
    
    // Internal build stages
    void load_vectors();
    void initialize_random_graph();
    void build_vamana_graph();
    void optimize_graph_layout();
    void serialize_index();
    void compute_checksums();
};

} // namespace ssd_vector
```

---

### Input Format

#### Vector Input File Format

**Option 1: Binary Format** (Recommended for performance)
```
┌─────────────────────────────────────┐
│ Header (64 bytes)                   │
├─────────────────────────────────────┤
│ Magic: "VECS" (4 bytes)             │
│ Version: 1 (4 bytes)                │
│ Count: N (8 bytes)                  │
│ Dimension: D (4 bytes)              │
│ Data Type: f32 (4 bytes)            │
│ Reserved (40 bytes)                 │
├─────────────────────────────────────┤
│ Vector 0: [f32 × D]                 │
│ Vector 1: [f32 × D]                 │
│ ...                                 │
│ Vector N-1: [f32 × D]               │
└─────────────────────────────────────┘
```

**Option 2: FVECS Format** (Compatible with ANN benchmarks)
```
For each vector:
  dimension (4 bytes, int32)
  components (dimension × 4 bytes, f32)
```

**Option 3: NPY Format** (NumPy compatibility)
- Standard NumPy `.npy` file with shape `(N, D)` and dtype `float32`

---

### Output Format

Generated files in `output_dir/`:
- `manifest.json` - Index metadata
- `vectors.bin` - Vector data (per ADR-001)
- `graph.bin` - Graph structure (per ADR-001)
- `metadata.bin` - ID mappings (per ADR-001)
- `checksums.sha256` - Integrity checksums

See ADR-001 for detailed format specifications.

---

## Algorithm Implementation

### Build Process Overview

```
┌──────────────────────────────────────────┐
│ 1. Load & Validate Input                 │ 
│    - Parse input file                    │
│    - Validate dimensions                 │
│    - Allocate memory                     │
└────────────────┬─────────────────────────┘
                 │
┌────────────────▼─────────────────────────┐
│ 2. Initialize Random Graph               │
│    - Create nodes (one per vector)       │
│    - Random edges (for connectivity)     │
└────────────────┬─────────────────────────┘
                 │
┌────────────────▼─────────────────────────┐
│ 3. Vamana Graph Construction             │
│    - For each node (random order):       │
│      a. Search for neighbors             │
│      b. Robust prune to R edges          │
│      c. Update reverse edges             │
└────────────────┬─────────────────────────┘
                 │
┌────────────────▼─────────────────────────┐
│ 4. Optimize Graph Layout                 │
│    - Reorder nodes for locality          │
│    - Update all edge references          │
└────────────────┬─────────────────────────┘
                 │
┌────────────────▼─────────────────────────┐
│ 5. Serialize to Disk                     │
│    - Write vectors.bin                   │
│    - Write graph.bin                     │
│    - Write metadata.bin                  │
│    - Generate manifest.json              │
└────────────────┬─────────────────────────┘
                 │
┌────────────────▼─────────────────────────┐
│ 6. Compute Checksums & Validate          │
│    - SHA-256 for each file               │
│    - Basic connectivity checks           │
└──────────────────────────────────────────┘
```

---

### Stage 1: Load & Validate Input

**Steps**:
1. Open and parse input file header
2. Validate vector count and dimension
3. Check available memory vs required memory
4. Memory-map or stream vectors

**Validations**:
- File size matches expected size: `header_size + (count × dimension × 4)`
- Dimension is reasonable: `64 ≤ dimension ≤ 4096`
- No NaN or Inf values in vectors
- Vectors are non-zero (for cosine similarity)

**Memory Management**:
```cpp
size_t required_memory = 
    vector_count * dimension * sizeof(float) +  // Vectors
    vector_count * R * sizeof(uint32_t) +       // Graph
    overhead;
    
if (required_memory > config_.max_memory_gb * GB) {
    // Use streaming or external memory approach
    use_streaming_build();
} else {
    // Load everything into RAM
    use_inmemory_build();
}
```

---

### Stage 2: Initialize Random Graph

**Purpose**: Create initial connectivity for graph construction

**Algorithm**:
```cpp
void initialize_random_graph() {
    std::mt19937 rng(config_.random_seed);  // Deterministic
    
    for (uint32_t i = 0; i < vector_count; i++) {
        // Add random edges (at least R/2)
        std::uniform_int_distribution<uint32_t> dist(0, vector_count - 1);
        
        for (uint32_t j = 0; j < R / 2; j++) {
            uint32_t neighbor = dist(rng);
            if (neighbor != i) {
                graph[i].add_edge(neighbor);
            }
        }
    }
}
```

**Properties**:
- Each node has at least R/2 random edges
- Ensures initial graph connectivity
- Deterministic (fixed seed)

---

### Stage 3: Vamana Graph Construction

**Core Algorithm** (from ADR-002):

```cpp
void build_vamana_graph() {
    // Process nodes in random order (better convergence)
    std::vector<uint32_t> order = random_permutation(vector_count);
    
    #pragma omp parallel for num_threads(config_.num_threads)
    for (uint32_t idx = 0; idx < vector_count; idx++) {
        uint32_t node_id = order[idx];
        
        // 1. Search for approximate neighbors
        auto candidates = greedy_search(
            vectors[node_id],
            config_.L_build,
            entry_point  // Start from medoid
        );
        
        // 2. Robust prune to keep best R neighbors
        auto pruned = robust_prune(
            node_id,
            candidates,
            config_.R,
            config_.alpha
        );
        
        // 3. Update node's edges
        graph[node_id].set_edges(pruned);
        
        // 4. Update reverse edges (for bidirectionality)
        for (uint32_t neighbor : pruned) {
            graph[neighbor].try_add_reverse_edge(node_id, config_.R);
        }
        
        // Progress tracking
        progress_++;
    }
}
```

**Greedy Search**:
```cpp
std::vector<uint32_t> greedy_search(
    const float* query,
    uint32_t L,
    uint32_t start_node
) {
    std::priority_queue<NodeDist> candidates;  // Min-heap by distance
    std::unordered_set<uint32_t> visited;
    
    candidates.push({start_node, distance(query, vectors[start_node])});
    visited.insert(start_node);
    
    while (!candidates.empty() && visited.size() < L) {
        auto current = candidates.top();
        candidates.pop();
        
        // Explore neighbors
        for (uint32_t neighbor : graph[current.id].edges) {
            if (visited.count(neighbor)) continue;
            
            float dist = distance(query, vectors[neighbor]);
            candidates.push({neighbor, dist});
            visited.insert(neighbor);
        }
    }
    
    // Return top L candidates
    return get_top_k(candidates, L);
}
```

**Robust Prune** (RobustPrune from DiskANN paper):
```cpp
std::vector<uint32_t> robust_prune(
    uint32_t node_id,
    const std::vector<uint32_t>& candidates,
    uint32_t R,
    float alpha
) {
    std::vector<uint32_t> result;
    std::vector<bool> selected(candidates.size(), false);
    
    // Sort candidates by distance to node_id
    auto sorted = sort_by_distance(node_id, candidates);
    
    for (size_t i = 0; i < sorted.size() && result.size() < R; i++) {
        uint32_t candidate = sorted[i].id;
        
        // Check if candidate is diverse enough
        bool diverse = true;
        for (uint32_t existing : result) {
            float dist_to_existing = distance(
                vectors[candidate],
                vectors[existing]
            );
            float dist_to_center = sorted[i].distance;
            
            // Alpha controls pruning aggressiveness
            if (dist_to_existing < alpha * dist_to_center) {
                diverse = false;
                break;
            }
        }
        
        if (diverse) {
            result.push_back(candidate);
        }
    }
    
    return result;
}
```

**Parallelization**:
- Process multiple nodes concurrently
- Thread-safe graph updates (lock per node)
- Load balancing via OpenMP dynamic scheduling

---

### Stage 4: Optimize Graph Layout

**Purpose**: Reorder nodes on disk to improve spatial locality during search

**Algorithm**: BFS-based ordering

```cpp
std::vector<uint32_t> optimize_layout() {
    std::vector<uint32_t> new_order;
    std::vector<bool> visited(vector_count, false);
    std::queue<uint32_t> queue;
    
    // Start from medoid (entry point)
    queue.push(medoid_id);
    visited[medoid_id] = true;
    
    // BFS traversal
    while (!queue.empty()) {
        uint32_t node = queue.front();
        queue.pop();
        
        new_order.push_back(node);
        
        // Add unvisited neighbors
        for (uint32_t neighbor : graph[node].edges) {
            if (!visited[neighbor]) {
                visited[neighbor] = true;
                queue.push(neighbor);
            }
        }
    }
    
    // Handle any disconnected components
    for (uint32_t i = 0; i < vector_count; i++) {
        if (!visited[i]) {
            new_order.push_back(i);
        }
    }
    
    return new_order;
}

void apply_layout_optimization(const std::vector<uint32_t>& new_order) {
    // Create mapping: old_id -> new_id
    std::vector<uint32_t> mapping(vector_count);
    for (uint32_t i = 0; i < vector_count; i++) {
        mapping[new_order[i]] = i;
    }
    
    // Reorder vectors and graph
    reorder_vectors(new_order);
    update_graph_edges(mapping);
}
```

**Expected Benefit**: 20-30% reduction in disk seeks during search

---

### Stage 5: Serialize to Disk

**Implementation**:
```cpp
void serialize_index() {
    // 1. Write vectors.bin
    VectorFile::write(
        output_dir + "/vectors.bin",
        vectors,
        vector_count,
        dimension,
        config_.metric
    );
    
    // 2. Write graph.bin
    GraphFile::write(
        output_dir + "/graph.bin",
        graph,
        vector_count,
        R
    );
    
    // 3. Write metadata.bin
    MetadataFile::write(
        output_dir + "/metadata.bin",
        id_mapping,
        vector_count
    );
    
    // 4. Write manifest.json
    ManifestFile::write(
        output_dir + "/manifest.json",
        create_manifest()
    );
}
```

**Manifest Generation**:
```cpp
json create_manifest() {
    return {
        {"version", "1.0.0"},
        {"format_version", 1},
        {"created_at", current_timestamp()},
        {"vector_count", vector_count},
        {"dimension", dimension},
        {"metric", metric_to_string(config_.metric)},
        {"build_parameters", {
            {"R", config_.R},
            {"L", config_.L_build},
            {"alpha", config_.alpha}
        }},
        {"files", {
            {"vectors", "vectors.bin"},
            {"graph", "graph.bin"},
            {"metadata", "metadata.bin"}
        }}
    };
}
```

---

### Stage 6: Compute Checksums

**Implementation**:
```cpp
void compute_checksums() {
    std::ofstream checksum_file(output_dir + "/checksums.sha256");
    
    for (const auto& filename : {"vectors.bin", "graph.bin", "metadata.bin"}) {
        std::string hash = compute_sha256(output_dir + "/" + filename);
        checksum_file << hash << "  " << filename << "\n";
    }
}
```

---

## Performance Requirements

### Build Time Targets

| Dataset Size | Max Build Time | Throughput |
|--------------|----------------|------------|
| 100k vectors | < 3 minutes | ~550 vec/s |
| 1M vectors | < 30 minutes | ~550 vec/s |
| 10M vectors | < 5 hours | ~550 vec/s |

**Conditions**: Single-node, 8GB RAM, 4 CPU cores, NVMe SSD

---

### Memory Requirements

| Phase | Memory Usage |
|-------|--------------|
| **Load Vectors** | `N × D × 4 bytes` |
| **Graph Construction** | `+ N × R × 4 bytes` |
| **Layout Optimization** | `+ N × 4 bytes` (temp) |
| **Total Peak** | ~1.5x raw data size |

**Example**: 1M vectors × 1536D = 6.1GB + 128MB graph ≈ **6.3GB peak**

---

### Disk Space Requirements

**Output Index Size**:
- Vectors: `N × D × 4 bytes`
- Graph: `N × R × 4 bytes` (R=32 → 128 bytes per node)
- Metadata: `N × 8 bytes`
- **Total**: ~2-2.5x raw vector data

**Temporary Space**: ~1.5x final index size during build

---

### Parallelization

**Multi-threading**:
- Graph construction: Parallel across nodes (OpenMP)
- Distance computation: SIMD within threads
- Serialization: Single-threaded (IO-bound)

**Scalability**: Near-linear speedup up to 8 cores

---

## Quality Metrics

### Graph Quality Metrics

Computed and logged after build:

```cpp
struct GraphQuality {
    double avg_degree;        // Should be close to R
    double min_degree;        // Should be > 0
    double max_degree;        // Should be ≤ R
    double avg_distance;      // Average edge distance
    bool is_connected;        // Graph connectivity
    uint32_t num_components;  // Connected components (should be 1)
};
```

**Validation Rules**:
- ✅ Graph is connected (single component)
- ✅ Every node has at least 1 edge
- ✅ Average degree ≥ R * 0.8 (allow some variance)
- ✅ No self-loops

---

### Determinism Validation

**Test**: Build same input twice with same seed, compare outputs

```cpp
bool validate_determinism(
    const std::string& index1,
    const std::string& index2
) {
    // Compare checksums
    auto checksums1 = read_checksums(index1 + "/checksums.sha256");
    auto checksums2 = read_checksums(index2 + "/checksums.sha256");
    
    return checksums1 == checksums2;  // Must match exactly
}
```

**Requirement**: 100% deterministic builds (byte-for-byte identical)

---

## Dependencies

### Internal Dependencies

- **ADR-001**: Storage format specification
- **ADR-002**: DiskANN algorithm and parameters
- **ADR-003**: Memory budget (build memory limits)

### External Dependencies

- **C++ Standard Library**: STL containers, algorithms
- **OpenMP**: Parallel graph construction
- **SIMD Libraries**: AVX2/AVX-512 for distance computation
  - Consider: `xsimd` or hand-written intrinsics
- **JSON Library**: Manifest generation
  - Recommended: `nlohmann/json`
- **SHA-256 Library**: Checksum computation
  - Consider: OpenSSL, or standalone implementation

### Optional Dependencies

- **Progress Bar**: `indicators` library (for CLI progress display)
- **Memory Profiling**: `jemalloc` or `tcmalloc`

---

## Testing Strategy

### Unit Tests

```cpp
TEST(IndexBuilder, LoadValidVectors) {
    // Test vector loading from various formats
}

TEST(IndexBuilder, InitializeRandomGraph) {
    // Verify random graph properties
}

TEST(IndexBuilder, RobustPrune) {
    // Test pruning algorithm correctness
}

TEST(IndexBuilder, GraphConnectivity) {
    // Ensure built graph is connected
}

TEST(IndexBuilder, Determinism) {
    // Build twice, verify identical output
}

TEST(IndexBuilder, SerializationFormat) {
    // Verify output matches ADR-001 format
}
```

---

### Integration Tests

```cpp
TEST(IndexBuilder, SmallDataset) {
    // Build index for 1000 vectors, verify correctness
}

TEST(IndexBuilder, MediumDataset) {
    // Build index for 100k vectors, check build time
}

TEST(IndexBuilder, VariousDimensions) {
    // Test dimensions: 128, 384, 768, 1536, 2048
}

TEST(IndexBuilder, AllMetrics) {
    // Build with cosine, L2, inner product
}
```

---

### Performance Tests

```cpp
BENCHMARK(IndexBuilder, Build1M) {
    // Measure build time for 1M vectors
    // Target: < 30 minutes
}

BENCHMARK(IndexBuilder, MemoryUsage) {
    // Monitor peak memory during build
    // Target: < 8GB for 1M vectors
}

BENCHMARK(IndexBuilder, Parallelization) {
    // Test speedup with 1, 2, 4, 8 threads
}
```

---

### Validation Tests

**Recall Validation**:
```cpp
TEST(IndexBuilder, RecallQuality) {
    // Build index
    auto index = IndexBuilder(config).build();
    
    // Load index and search
    auto searcher = IndexSearcher(index);
    
    // Compare to brute force
    double recall = measure_recall(
        searcher,
        test_queries,
        ground_truth
    );
    
    EXPECT_GE(recall, 0.90);  // At least 90% recall
}
```

**Corruption Detection**:
```cpp
TEST(IndexBuilder, CorruptionDetection) {
    // Build index
    auto index = build_test_index();
    
    // Corrupt a file
    corrupt_file(index + "/graph.bin");
    
    // Verify checksum fails
    EXPECT_FALSE(verify_checksums(index));
}
```

---

## Error Handling

### Build Errors

| Error | Condition | Recovery |
|-------|-----------|----------|
| **InvalidInput** | Malformed input file | Report error, exit cleanly |
| **OutOfMemory** | Insufficient RAM | Suggest reducing batch size or using streaming |
| **DiskFull** | Insufficient disk space | Report required space, exit |
| **CorruptInput** | NaN/Inf in vectors | Skip bad vectors, log warning |
| **Timeout** | Build exceeds time limit | Save partial progress, allow resume |

### Error Reporting

```cpp
class BuildException : public std::exception {
public:
    enum class ErrorCode {
        INVALID_INPUT,
        OUT_OF_MEMORY,
        DISK_FULL,
        CORRUPT_INPUT,
        TIMEOUT
    };
    
    BuildException(ErrorCode code, const std::string& message)
        : code_(code), message_(message) {}
    
    ErrorCode code() const { return code_; }
    const char* what() const noexcept override { return message_.c_str(); }
    
private:
    ErrorCode code_;
    std::string message_;
};
```

---

## Configuration and Tuning

### Tuning Parameters

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| **R** | 32 | 16-64 | Higher = better recall, larger index |
| **L_build** | 100 | 50-200 | Higher = better graph, slower build |
| **alpha** | 1.2 | 1.0-1.5 | Higher = more diverse edges |
| **num_threads** | 4 | 1-16 | More = faster build (diminishing returns) |
| **max_memory_gb** | 8 | 4-32 | Limits peak memory usage |

### Configuration File Format

```json
{
  "build_parameters": {
    "R": 32,
    "L_build": 100,
    "alpha": 1.2
  },
  "metric": "cosine",
  "resources": {
    "max_memory_gb": 8,
    "num_threads": 4
  },
  "optimization": {
    "optimize_layout": true,
    "random_seed": 42
  }
}
```

---

## Monitoring and Logging

### Build Progress Logging

```
[INFO] Starting index build...
[INFO] Input: 1000000 vectors, dimension: 1536
[INFO] Memory estimate: 6.3 GB
[INFO] Stage 1/6: Loading vectors... 100% (30s)
[INFO] Stage 2/6: Initializing graph... 100% (5s)
[INFO] Stage 3/6: Building Vamana graph... 45% (ETA: 15m)
[INFO] Stage 4/6: Optimizing layout... 100% (2m)
[INFO] Stage 5/6: Serializing to disk... 100% (1m)
[INFO] Stage 6/6: Computing checksums... 100% (10s)
[INFO] Build complete! Total time: 18m 30s
[INFO] Index size: 6.8 GB
[INFO] Average degree: 31.8
```

### Metrics to Log

- Stage progress (percentage)
- ETA (estimated time to completion)
- Memory usage (current/peak)
- Disk usage
- Graph statistics (avg/min/max degree)
- Build time per stage
- Final index size

---

## Implementation Notes

### Entry Point Selection

**Medoid Selection**: Choose entry point as vector closest to dataset centroid

```cpp
uint32_t select_medoid(const std::vector<Vector>& vectors) {
    // Compute centroid
    Vector centroid = compute_centroid(vectors);
    
    // Find closest vector to centroid
    uint32_t medoid = 0;
    float min_dist = std::numeric_limits<float>::max();
    
    for (uint32_t i = 0; i < vectors.size(); i++) {
        float dist = distance(vectors[i], centroid);
        if (dist < min_dist) {
            min_dist = dist;
            medoid = i;
        }
    }
    
    return medoid;
}
```

Store medoid ID in manifest.json for use during search.

---

### Distance Function Optimization

**Use SIMD**: Implement AVX2 optimized distance functions

```cpp
// Pseudo-code for SIMD cosine similarity
float cosine_similarity_avx2(const float* a, const float* b, size_t dim) {
    __m256 sum = _mm256_setzero_ps();
    for (size_t i = 0; i < dim; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        sum = _mm256_fmadd_ps(va, vb, sum);
    }
    return horizontal_sum(sum);
}
```

**Fallback**: Provide scalar implementation for non-AVX2 systems

---

### Crash Recovery

**Checkpoint Strategy** (optional for long builds):
```cpp
void save_checkpoint(const std::string& checkpoint_file) {
    // Save current state
    serialize_partial_graph(checkpoint_file);
    save_progress(checkpoint_file + ".progress");
}

bool load_checkpoint(const std::string& checkpoint_file) {
    if (file_exists(checkpoint_file)) {
        deserialize_partial_graph(checkpoint_file);
        restore_progress(checkpoint_file + ".progress");
        return true;
    }
    return false;
}
```

**Not in v1 scope** - deferred to v1.1

---

## Future Enhancements

1. **Incremental Build** (v1.1)
   - Add new vectors to existing index without full rebuild
   
2. **Distributed Build** (v2)
   - Partition dataset across nodes, merge graphs

3. **Compression** (v2)
   - Quantize vectors during build (PQ, OPQ)

4. **GPU Acceleration** (v2)
   - Use GPU for distance computations

5. **Streaming Build** (v1.1)
   - Build with data larger than RAM

---

## Open Questions

1. **Max vector dimension**: Should we enforce a hard limit? (Proposed: 4096)
2. **Input validation**: How strict? Skip bad vectors or fail fast?
3. **Resume support**: Priority for v1 or defer?

---

## References

- DiskANN Paper: https://arxiv.org/abs/1907.10310
- Vamana Algorithm: Section 3 of DiskANN paper
- ANN Benchmarks: http://ann-benchmarks.com/
- SIMD Distance Functions: https://github.com/facebookresearch/faiss

---

**Status**: Draft - Ready for review  
**Next Steps**: Team review, implementation planning  
**Blocks**: COMP-002 (needs index format), COMP-004 (needs built indexes to search)
