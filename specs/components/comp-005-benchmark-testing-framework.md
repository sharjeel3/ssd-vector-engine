# Component: Benchmark and Testing Framework

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

The Benchmark and Testing Framework provides standardized evaluation tools for measuring the quality and performance of the SSD Vector Engine. It enables reproducible testing, performance regression detection, and comparison against ground truth and competing systems.

**Core Responsibilities**:
1. Load standard vector datasets (SIFT, GIST, Deep1B, etc.)
2. Compute ground truth using brute-force search
3. Measure recall accuracy against ground truth
4. Measure query latency (p50, p95, p99)
5. Measure throughput (QPS)
6. Generate performance reports
7. Integrate with CI/CD for regression testing
8. Compare against other ANN systems

---

## Interface

### Public API

```cpp
namespace ssd_vector {
namespace benchmark {

// ============================================================================
// Dataset Loading
// ============================================================================

struct Dataset {
    std::string name;                       // Dataset name (e.g., "sift1m")
    uint32_t dimension;                     // Vector dimension
    uint64_t train_count;                   // Training vectors count
    uint64_t test_count;                    // Test queries count
    
    std::vector<Vector> train_vectors;      // Vectors to build index
    std::vector<Vector> test_queries;       // Query vectors
    std::vector<std::vector<IDMapper::ExternalID>> ground_truth;  // GT neighbors
    
    DistanceMetric metric;                  // Distance metric
};

enum class DistanceMetric {
    COSINE,
    L2,
    INNER_PRODUCT
};

class DatasetLoader {
public:
    // Load standard benchmarking datasets
    static Dataset load_sift1m(const std::string& data_dir);
    static Dataset load_sift10m(const std::string& data_dir);
    static Dataset load_gist1m(const std::string& data_dir);
    static Dataset load_deep1m(const std::string& data_dir);
    static Dataset load_deep10m(const std::string& data_dir);
    
    // Load custom dataset from various formats
    static Dataset load_fvecs(
        const std::string& base_file,
        const std::string& query_file,
        const std::string& gt_file = ""
    );
    
    static Dataset load_hdf5(
        const std::string& file_path,
        const std::string& dataset_name
    );
    
    static Dataset load_numpy(
        const std::string& train_file,
        const std::string& test_file,
        const std::string& gt_file = ""
    );
    
    // Generate synthetic dataset
    static Dataset generate_random(
        uint32_t dimension,
        uint64_t train_count,
        uint64_t test_count,
        DistanceMetric metric = DistanceMetric::COSINE
    );
    
private:
    static std::vector<float> read_fvecs_file(const std::string& path);
    static std::vector<int32_t> read_ivecs_file(const std::string& path);
};

// ============================================================================
// Ground Truth Computation
// ============================================================================

class GroundTruthComputer {
public:
    struct Config {
        uint32_t k = 100;                   // Number of neighbors to compute
        DistanceMetric metric = DistanceMetric::COSINE;
        uint32_t num_threads = 8;           // Parallel threads
        bool use_simd = true;               // SIMD optimization
    };
    
    explicit GroundTruthComputer(const Config& config = Config());
    
    // Compute ground truth for query set
    std::vector<std::vector<IDMapper::ExternalID>> compute(
        const std::vector<Vector>& train_vectors,
        const std::vector<Vector>& test_queries
    );
    
    // Compute single query ground truth
    std::vector<IDMapper::ExternalID> compute_single(
        const std::vector<Vector>& train_vectors,
        const Vector& query
    );
    
private:
    Config config_;
};

// ============================================================================
// Recall Metrics
// ============================================================================

class RecallEvaluator {
public:
    // Compute recall@k
    static double compute_recall_at_k(
        const std::vector<IDMapper::ExternalID>& predicted,
        const std::vector<IDMapper::ExternalID>& ground_truth,
        uint32_t k
    );
    
    // Compute recall for batch of queries
    static std::vector<double> compute_batch_recall(
        const std::vector<std::vector<IDMapper::ExternalID>>& predicted_batch,
        const std::vector<std::vector<IDMapper::ExternalID>>& ground_truth_batch,
        uint32_t k
    );
    
    // Compute average recall across queries
    static double compute_average_recall(
        const std::vector<std::vector<IDMapper::ExternalID>>& predicted_batch,
        const std::vector<std::vector<IDMapper::ExternalID>>& ground_truth_batch,
        uint32_t k
    );
    
    // Compute recall curve (recall@1, recall@5, recall@10, ...)
    static std::map<uint32_t, double> compute_recall_curve(
        const std::vector<std::vector<IDMapper::ExternalID>>& predicted_batch,
        const std::vector<std::vector<IDMapper::ExternalID>>& ground_truth_batch,
        const std::vector<uint32_t>& k_values = {1, 5, 10, 20, 50, 100}
    );
};

// ============================================================================
// Performance Benchmark
// ============================================================================

struct BenchmarkConfig {
    // Dataset
    std::string dataset_name;
    std::string dataset_path;
    
    // Index build parameters
    IndexBuilder::Config build_config;
    
    // Search parameters
    std::vector<uint32_t> L_values = {50, 75, 100, 150, 200};  // Test multiple L
    uint32_t top_k = 10;
    
    // Performance measurement
    uint32_t warmup_queries = 100;          // Warm-up before timing
    uint32_t measured_queries = 1000;       // Queries to measure
    bool measure_throughput = true;         // QPS measurement
    uint32_t throughput_threads = 8;        // Threads for QPS test
    
    // Output
    std::string output_dir = "benchmark_results";
    bool save_detailed_results = true;
    bool generate_plots = true;
};

struct BenchmarkResult {
    std::string dataset_name;
    uint32_t dimension;
    uint64_t vector_count;
    
    // Build metrics
    double build_time_seconds;
    size_t index_size_mb;
    
    // Per-L results
    struct LResult {
        uint32_t L;
        
        // Recall
        double recall_at_1;
        double recall_at_5;
        double recall_at_10;
        double recall_at_100;
        
        // Latency (ms)
        double latency_p50;
        double latency_p95;
        double latency_p99;
        double latency_p999;
        double latency_avg;
        
        // Throughput
        double qps;
        
        // Algorithm stats
        double avg_nodes_visited;
        double avg_distance_computations;
        double graph_cache_hit_rate;
        double vector_cache_hit_rate;
    };
    
    std::vector<LResult> l_results;
    
    // Export to JSON
    std::string to_json() const;
    
    // Export to CSV
    void to_csv(const std::string& filepath) const;
    
    // Print summary
    void print_summary() const;
};

class BenchmarkRunner {
public:
    explicit BenchmarkRunner(const BenchmarkConfig& config);
    
    // Run full benchmark
    BenchmarkResult run();
    
    // Run individual phases
    void build_index();
    BenchmarkResult::LResult evaluate_search_with_L(uint32_t L);
    
    // Generate reports
    void generate_html_report(const BenchmarkResult& result);
    void generate_plots(const BenchmarkResult& result);
    
private:
    BenchmarkConfig config_;
    Dataset dataset_;
    std::unique_ptr<IndexBuilder> builder_;
    std::unique_ptr<SearchEngine> engine_;
    
    // Measure latency distribution
    std::vector<double> measure_latencies(
        const std::vector<SearchQuery>& queries
    );
    
    // Measure throughput
    double measure_throughput(
        const std::vector<SearchQuery>& queries,
        uint32_t num_threads,
        uint32_t duration_seconds = 10
    );
};

// ============================================================================
// Comparative Benchmarking
// ============================================================================

struct ComparisonConfig {
    std::string dataset_name;
    std::string dataset_path;
    
    // Systems to compare
    std::vector<std::string> systems = {
        "ssd-vector",
        "faiss-ivfpq",
        "hnswlib",
        "annoy"
    };
    
    // Test parameters
    std::vector<uint32_t> recall_targets = {90, 95, 99};  // Target recall %
    uint32_t top_k = 10;
};

struct ComparisonResult {
    std::string dataset_name;
    
    struct SystemResult {
        std::string system_name;
        double build_time_seconds;
        size_t index_size_mb;
        
        // At target recall levels
        struct RecallPoint {
            double target_recall;
            double actual_recall;
            double latency_p99_ms;
            double qps;
            std::string parameters;     // e.g., "L=100" or "ef=150"
        };
        
        std::vector<RecallPoint> recall_points;
    };
    
    std::vector<SystemResult> system_results;
    
    // Export
    void to_markdown_table(const std::string& filepath) const;
    void to_json(const std::string& filepath) const;
    void generate_comparison_plots(const std::string& output_dir) const;
};

class ComparisonBenchmark {
public:
    explicit ComparisonBenchmark(const ComparisonConfig& config);
    
    ComparisonResult run();
    
private:
    ComparisonConfig config_;
    
    // Run SSD Vector Engine
    ComparisonResult::SystemResult benchmark_ssd_vector();
    
    // Run FAISS
    ComparisonResult::SystemResult benchmark_faiss();
    
    // Run HNSWlib
    ComparisonResult::SystemResult benchmark_hnswlib();
    
    // Run Annoy
    ComparisonResult::SystemResult benchmark_annoy();
};

// ============================================================================
// Continuous Integration Testing
// ============================================================================

class RegressionTest {
public:
    struct Baseline {
        std::string dataset_name;
        uint32_t L;
        double min_recall;                  // Minimum acceptable recall
        double max_latency_p99;             // Maximum acceptable latency
    };
    
    explicit RegressionTest(const std::vector<Baseline>& baselines);
    
    // Run regression tests
    bool run_all_tests();
    
    // Check single test
    bool check_regression(
        const std::string& dataset_name,
        const BenchmarkResult::LResult& result
    );
    
private:
    std::vector<Baseline> baselines_;
};

// ============================================================================
// Utilities
// ============================================================================

class BenchmarkUtils {
public:
    // Latency statistics
    struct LatencyStats {
        double p50;
        double p95;
        double p99;
        double p999;
        double mean;
        double std_dev;
        double min;
        double max;
    };
    
    static LatencyStats compute_latency_stats(
        const std::vector<double>& latencies
    );
    
    // Format results
    static std::string format_latency(double ms);
    static std::string format_throughput(double qps);
    static std::string format_size(size_t bytes);
    static std::string format_recall(double recall);
    
    // Plotting (generates matplotlib Python scripts)
    static void plot_recall_vs_latency(
        const std::vector<BenchmarkResult::LResult>& results,
        const std::string& output_path
    );
    
    static void plot_recall_vs_qps(
        const std::vector<BenchmarkResult::LResult>& results,
        const std::string& output_path
    );
    
    static void plot_system_comparison(
        const ComparisonResult& comparison,
        const std::string& output_path
    );
};

} // namespace benchmark
} // namespace ssd_vector
```

---

## Implementation Details

### Dataset Loading (FVECS Format)

```cpp
Dataset DatasetLoader::load_fvecs(
    const std::string& base_file,
    const std::string& query_file,
    const std::string& gt_file
) {
    Dataset dataset;
    
    // Load base vectors
    auto base_data = read_fvecs_file(base_file);
    uint32_t dimension = *reinterpret_cast<uint32_t*>(base_data.data());
    uint64_t base_count = base_data.size() / (dimension + 1);
    
    dataset.dimension = dimension;
    dataset.train_count = base_count;
    dataset.train_vectors.reserve(base_count);
    
    for (uint64_t i = 0; i < base_count; i++) {
        size_t offset = i * (dimension + 1) + 1;  // Skip dimension header
        std::vector<float> vec_data(
            base_data.begin() + offset,
            base_data.begin() + offset + dimension
        );
        dataset.train_vectors.emplace_back(Vector(std::move(vec_data)));
    }
    
    // Load query vectors
    auto query_data = read_fvecs_file(query_file);
    uint64_t query_count = query_data.size() / (dimension + 1);
    dataset.test_count = query_count;
    dataset.test_queries.reserve(query_count);
    
    for (uint64_t i = 0; i < query_count; i++) {
        size_t offset = i * (dimension + 1) + 1;
        std::vector<float> vec_data(
            query_data.begin() + offset,
            query_data.begin() + offset + dimension
        );
        dataset.test_queries.emplace_back(Vector(std::move(vec_data)));
    }
    
    // Load ground truth if provided
    if (!gt_file.empty()) {
        auto gt_data = read_ivecs_file(gt_file);
        uint32_t k = *reinterpret_cast<uint32_t*>(gt_data.data());
        
        dataset.ground_truth.reserve(query_count);
        for (uint64_t i = 0; i < query_count; i++) {
            std::vector<IDMapper::ExternalID> neighbors;
            neighbors.reserve(k);
            
            size_t offset = i * (k + 1) + 1;
            for (uint32_t j = 0; j < k; j++) {
                neighbors.push_back(gt_data[offset + j]);
            }
            dataset.ground_truth.push_back(neighbors);
        }
    }
    
    return dataset;
}

std::vector<float> DatasetLoader::read_fvecs_file(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open file: " + path);
    }
    
    // Get file size
    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    // Read all data
    std::vector<float> data(file_size / sizeof(float));
    file.read(reinterpret_cast<char*>(data.data()), file_size);
    
    return data;
}
```

---

### Ground Truth Computation

```cpp
std::vector<std::vector<IDMapper::ExternalID>> GroundTruthComputer::compute(
    const std::vector<Vector>& train_vectors,
    const std::vector<Vector>& test_queries
) {
    std::vector<std::vector<IDMapper::ExternalID>> ground_truth(
        test_queries.size()
    );
    
    // Get distance function
    auto distance_func = [this](const float* a, const float* b, uint32_t dim) {
        switch (config_.metric) {
            case DistanceMetric::COSINE:
                return config_.use_simd
                    ? DistanceComputer::cosine_distance_simd(a, b, dim)
                    : DistanceComputer::cosine_distance(a, b, dim);
            case DistanceMetric::L2:
                return config_.use_simd
                    ? DistanceComputer::l2_distance_simd(a, b, dim)
                    : DistanceComputer::l2_distance(a, b, dim);
            case DistanceMetric::INNER_PRODUCT:
                return config_.use_simd
                    ? DistanceComputer::inner_product_distance_simd(a, b, dim)
                    : DistanceComputer::inner_product_distance(a, b, dim);
        }
    };
    
    // Parallel brute-force search
    #pragma omp parallel for num_threads(config_.num_threads)
    for (size_t q = 0; q < test_queries.size(); q++) {
        const auto& query = test_queries[q];
        
        // Compute distances to all vectors
        struct Neighbor {
            uint64_t id;
            float distance;
            
            bool operator<(const Neighbor& other) const {
                return distance > other.distance;  // Max heap
            }
        };
        
        std::priority_queue<Neighbor> heap;
        
        for (uint64_t i = 0; i < train_vectors.size(); i++) {
            float dist = distance_func(
                query.data(),
                train_vectors[i].data(),
                query.dimension()
            );
            
            if (heap.size() < config_.k) {
                heap.push({i, dist});
            } else if (dist < heap.top().distance) {
                heap.pop();
                heap.push({i, dist});
            }
        }
        
        // Extract neighbors in sorted order
        std::vector<Neighbor> neighbors;
        while (!heap.empty()) {
            neighbors.push_back(heap.top());
            heap.pop();
        }
        std::reverse(neighbors.begin(), neighbors.end());
        
        // Convert to external IDs
        std::vector<IDMapper::ExternalID> neighbor_ids;
        neighbor_ids.reserve(neighbors.size());
        for (const auto& n : neighbors) {
            neighbor_ids.push_back(n.id);
        }
        
        ground_truth[q] = neighbor_ids;
        
        // Progress reporting
        if (q % 100 == 0) {
            LOG_INFO("Ground truth progress: " << q << "/" << test_queries.size());
        }
    }
    
    return ground_truth;
}
```

---

### Recall Computation

```cpp
double RecallEvaluator::compute_recall_at_k(
    const std::vector<IDMapper::ExternalID>& predicted,
    const std::vector<IDMapper::ExternalID>& ground_truth,
    uint32_t k
) {
    // Take top-k from both
    std::unordered_set<IDMapper::ExternalID> gt_set(
        ground_truth.begin(),
        ground_truth.begin() + std::min(k, static_cast<uint32_t>(ground_truth.size()))
    );
    
    uint32_t hits = 0;
    uint32_t check_count = std::min(k, static_cast<uint32_t>(predicted.size()));
    
    for (uint32_t i = 0; i < check_count; i++) {
        if (gt_set.count(predicted[i])) {
            hits++;
        }
    }
    
    return static_cast<double>(hits) / std::min(k, static_cast<uint32_t>(gt_set.size()));
}

double RecallEvaluator::compute_average_recall(
    const std::vector<std::vector<IDMapper::ExternalID>>& predicted_batch,
    const std::vector<std::vector<IDMapper::ExternalID>>& ground_truth_batch,
    uint32_t k
) {
    double total_recall = 0.0;
    
    for (size_t i = 0; i < predicted_batch.size(); i++) {
        total_recall += compute_recall_at_k(
            predicted_batch[i],
            ground_truth_batch[i],
            k
        );
    }
    
    return total_recall / predicted_batch.size();
}

std::map<uint32_t, double> RecallEvaluator::compute_recall_curve(
    const std::vector<std::vector<IDMapper::ExternalID>>& predicted_batch,
    const std::vector<std::vector<IDMapper::ExternalID>>& ground_truth_batch,
    const std::vector<uint32_t>& k_values
) {
    std::map<uint32_t, double> curve;
    
    for (uint32_t k : k_values) {
        curve[k] = compute_average_recall(predicted_batch, ground_truth_batch, k);
    }
    
    return curve;
}
```

---

### Benchmark Runner

```cpp
BenchmarkResult BenchmarkRunner::run() {
    BenchmarkResult result;
    result.dataset_name = config_.dataset_name;
    result.dimension = dataset_.dimension;
    result.vector_count = dataset_.train_count;
    
    // Phase 1: Build index
    LOG_INFO("Building index...");
    auto build_start = std::chrono::high_resolution_clock::now();
    
    build_index();
    
    auto build_end = std::chrono::high_resolution_clock::now();
    result.build_time_seconds = std::chrono::duration<double>(
        build_end - build_start
    ).count();
    
    // Get index size
    namespace fs = std::filesystem;
    size_t index_size = 0;
    for (const auto& entry : fs::directory_iterator(config_.output_dir + "/index")) {
        if (entry.is_regular_file()) {
            index_size += entry.file_size();
        }
    }
    result.index_size_mb = index_size / (1024 * 1024);
    
    LOG_INFO("Index built in " << result.build_time_seconds << "s, size: " 
             << result.index_size_mb << " MB");
    
    // Phase 2: Load index for searching
    LOG_INFO("Loading index...");
    engine_ = std::make_unique<SearchEngine>(
        config_.output_dir + "/index"
    );
    
    // Phase 3: Evaluate for each L value
    for (uint32_t L : config_.L_values) {
        LOG_INFO("Evaluating with L=" << L);
        auto l_result = evaluate_search_with_L(L);
        result.l_results.push_back(l_result);
    }
    
    // Phase 4: Generate reports
    if (config_.save_detailed_results) {
        result.to_csv(config_.output_dir + "/results.csv");
        result.to_json(config_.output_dir + "/results.json");
    }
    
    if (config_.generate_plots) {
        generate_plots(result);
    }
    
    result.print_summary();
    
    return result;
}

BenchmarkResult::LResult BenchmarkRunner::evaluate_search_with_L(uint32_t L) {
    BenchmarkResult::LResult result;
    result.L = L;
    
    // Prepare queries
    std::vector<SearchQuery> queries;
    queries.reserve(dataset_.test_count);
    
    for (const auto& test_query : dataset_.test_queries) {
        SearchQuery sq;
        sq.query_vector = test_query;
        sq.top_k = config_.top_k;
        sq.L = L;
        queries.push_back(sq);
    }
    
    // Warm-up
    LOG_INFO("Warming up with " << config_.warmup_queries << " queries...");
    for (uint32_t i = 0; i < config_.warmup_queries && i < queries.size(); i++) {
        engine_->search(queries[i]);
    }
    
    // Measure latencies
    LOG_INFO("Measuring latencies for " << config_.measured_queries << " queries...");
    std::vector<SearchQuery> measured_queries(
        queries.begin(),
        queries.begin() + std::min(config_.measured_queries, 
                                   static_cast<uint32_t>(queries.size()))
    );
    
    auto latencies = measure_latencies(measured_queries);
    auto latency_stats = BenchmarkUtils::compute_latency_stats(latencies);
    
    result.latency_p50 = latency_stats.p50;
    result.latency_p95 = latency_stats.p95;
    result.latency_p99 = latency_stats.p99;
    result.latency_p999 = latency_stats.p999;
    result.latency_avg = latency_stats.mean;
    
    // Compute recall
    LOG_INFO("Computing recall...");
    std::vector<std::vector<IDMapper::ExternalID>> predicted_results;
    predicted_results.reserve(measured_queries.size());
    
    for (const auto& query : measured_queries) {
        auto response = engine_->search(query);
        
        std::vector<IDMapper::ExternalID> ids;
        for (const auto& r : response.results) {
            ids.push_back(r.id);
        }
        predicted_results.push_back(ids);
    }
    
    // Get corresponding ground truth
    std::vector<std::vector<IDMapper::ExternalID>> relevant_gt(
        dataset_.ground_truth.begin(),
        dataset_.ground_truth.begin() + measured_queries.size()
    );
    
    result.recall_at_1 = RecallEvaluator::compute_average_recall(
        predicted_results, relevant_gt, 1
    );
    result.recall_at_5 = RecallEvaluator::compute_average_recall(
        predicted_results, relevant_gt, 5
    );
    result.recall_at_10 = RecallEvaluator::compute_average_recall(
        predicted_results, relevant_gt, 10
    );
    result.recall_at_100 = RecallEvaluator::compute_average_recall(
        predicted_results, relevant_gt, 100
    );
    
    // Measure throughput
    if (config_.measure_throughput) {
        LOG_INFO("Measuring throughput with " << config_.throughput_threads 
                 << " threads...");
        result.qps = measure_throughput(
            measured_queries,
            config_.throughput_threads
        );
    }
    
    // Get algorithm stats (average from last batch)
    auto sample_response = engine_->search(measured_queries[0]);
    result.avg_nodes_visited = sample_response.stats.nodes_visited;
    result.avg_distance_computations = sample_response.stats.distance_computations;
    
    auto cache_stats = engine_->get_cache_stats();
    result.graph_cache_hit_rate = cache_stats.graph_cache_hit_rate;
    result.vector_cache_hit_rate = cache_stats.vector_cache_hit_rate;
    
    return result;
}

std::vector<double> BenchmarkRunner::measure_latencies(
    const std::vector<SearchQuery>& queries
) {
    std::vector<double> latencies;
    latencies.reserve(queries.size());
    
    for (const auto& query : queries) {
        auto start = std::chrono::high_resolution_clock::now();
        
        auto response = engine_->search(query);
        
        auto end = std::chrono::high_resolution_clock::now();
        double latency_ms = std::chrono::duration<double, std::milli>(
            end - start
        ).count();
        
        latencies.push_back(latency_ms);
    }
    
    return latencies;
}

double BenchmarkRunner::measure_throughput(
    const std::vector<SearchQuery>& queries,
    uint32_t num_threads,
    uint32_t duration_seconds
) {
    std::atomic<uint64_t> query_count{0};
    std::atomic<bool> should_stop{false};
    
    // Launch timer thread
    std::thread timer([&]() {
        std::this_thread::sleep_for(std::chrono::seconds(duration_seconds));
        should_stop.store(true);
    });
    
    // Launch worker threads
    std::vector<std::thread> workers;
    for (uint32_t i = 0; i < num_threads; i++) {
        workers.emplace_back([&]() {
            size_t query_idx = 0;
            
            while (!should_stop.load()) {
                const auto& query = queries[query_idx % queries.size()];
                engine_->search(query);
                query_count++;
                query_idx++;
            }
        });
    }
    
    // Wait for completion
    timer.join();
    for (auto& worker : workers) {
        worker.join();
    }
    
    double qps = static_cast<double>(query_count.load()) / duration_seconds;
    return qps;
}
```

---

### Result Export

```cpp
std::string BenchmarkResult::to_json() const {
    std::ostringstream ss;
    ss << "{\n";
    ss << "  \"dataset\": \"" << dataset_name << "\",\n";
    ss << "  \"dimension\": " << dimension << ",\n";
    ss << "  \"vector_count\": " << vector_count << ",\n";
    ss << "  \"build_time_seconds\": " << build_time_seconds << ",\n";
    ss << "  \"index_size_mb\": " << index_size_mb << ",\n";
    ss << "  \"results\": [\n";
    
    for (size_t i = 0; i < l_results.size(); i++) {
        const auto& r = l_results[i];
        ss << "    {\n";
        ss << "      \"L\": " << r.L << ",\n";
        ss << "      \"recall_at_1\": " << r.recall_at_1 << ",\n";
        ss << "      \"recall_at_5\": " << r.recall_at_5 << ",\n";
        ss << "      \"recall_at_10\": " << r.recall_at_10 << ",\n";
        ss << "      \"recall_at_100\": " << r.recall_at_100 << ",\n";
        ss << "      \"latency_p50\": " << r.latency_p50 << ",\n";
        ss << "      \"latency_p95\": " << r.latency_p95 << ",\n";
        ss << "      \"latency_p99\": " << r.latency_p99 << ",\n";
        ss << "      \"latency_p999\": " << r.latency_p999 << ",\n";
        ss << "      \"qps\": " << r.qps << ",\n";
        ss << "      \"avg_nodes_visited\": " << r.avg_nodes_visited << ",\n";
        ss << "      \"graph_cache_hit_rate\": " << r.graph_cache_hit_rate << "\n";
        ss << "    }";
        if (i < l_results.size() - 1) ss << ",";
        ss << "\n";
    }
    
    ss << "  ]\n";
    ss << "}\n";
    
    return ss.str();
}

void BenchmarkResult::print_summary() const {
    std::cout << "\n=== Benchmark Results Summary ===\n";
    std::cout << "Dataset: " << dataset_name << "\n";
    std::cout << "Vectors: " << vector_count << " x " << dimension << "D\n";
    std::cout << "Build Time: " << build_time_seconds << "s\n";
    std::cout << "Index Size: " << index_size_mb << " MB\n\n";
    
    std::cout << std::setw(6) << "L"
              << std::setw(12) << "Recall@10"
              << std::setw(12) << "p50 (ms)"
              << std::setw(12) << "p99 (ms)"
              << std::setw(12) << "QPS"
              << std::setw(12) << "Nodes"
              << "\n";
    std::cout << std::string(72, '-') << "\n";
    
    for (const auto& r : l_results) {
        std::cout << std::setw(6) << r.L
                  << std::setw(12) << std::fixed << std::setprecision(4) 
                  << r.recall_at_10
                  << std::setw(12) << std::fixed << std::setprecision(2) 
                  << r.latency_p50
                  << std::setw(12) << std::fixed << std::setprecision(2) 
                  << r.latency_p99
                  << std::setw(12) << std::fixed << std::setprecision(0) 
                  << r.qps
                  << std::setw(12) << std::fixed << std::setprecision(0) 
                  << r.avg_nodes_visited
                  << "\n";
    }
    
    std::cout << "\n";
}
```

---

### Plotting Utilities

```cpp
void BenchmarkUtils::plot_recall_vs_latency(
    const std::vector<BenchmarkResult::LResult>& results,
    const std::string& output_path
) {
    // Generate Python matplotlib script
    std::ostringstream script;
    
    script << "import matplotlib.pyplot as plt\n";
    script << "import numpy as np\n\n";
    
    script << "# Data\n";
    script << "recall = [";
    for (size_t i = 0; i < results.size(); i++) {
        script << results[i].recall_at_10;
        if (i < results.size() - 1) script << ", ";
    }
    script << "]\n";
    
    script << "latency_p99 = [";
    for (size_t i = 0; i < results.size(); i++) {
        script << results[i].latency_p99;
        if (i < results.size() - 1) script << ", ";
    }
    script << "]\n";
    
    script << "L_values = [";
    for (size_t i = 0; i < results.size(); i++) {
        script << results[i].L;
        if (i < results.size() - 1) script << ", ";
    }
    script << "]\n\n";
    
    script << "# Plot\n";
    script << "plt.figure(figsize=(10, 6))\n";
    script << "plt.plot(latency_p99, recall, 'o-', linewidth=2, markersize=8)\n";
    script << "for i, L in enumerate(L_values):\n";
    script << "    plt.annotate(f'L={L}', (latency_p99[i], recall[i]), \n";
    script << "                 xytext=(5, 5), textcoords='offset points')\n";
    script << "plt.xlabel('p99 Latency (ms)', fontsize=12)\n";
    script << "plt.ylabel('Recall@10', fontsize=12)\n";
    script << "plt.title('Recall vs Latency Trade-off', fontsize=14)\n";
    script << "plt.grid(True, alpha=0.3)\n";
    script << "plt.savefig('" << output_path << "', dpi=300, bbox_inches='tight')\n";
    script << "print('Plot saved to " << output_path << "')\n";
    
    // Save script and execute
    std::string script_path = output_path + ".py";
    std::ofstream file(script_path);
    file << script.str();
    file.close();
    
    // Execute Python script
    std::string cmd = "python3 " + script_path;
    std::system(cmd.c_str());
}
```

---

## Testing Strategy

### Unit Tests

```cpp
TEST(DatasetLoader, LoadFVECS) {
    auto dataset = DatasetLoader::load_fvecs(
        "testdata/sift_base.fvecs",
        "testdata/sift_query.fvecs",
        "testdata/sift_groundtruth.ivecs"
    );
    
    EXPECT_EQ(dataset.dimension, 128);
    EXPECT_GT(dataset.train_count, 0);
    EXPECT_GT(dataset.test_count, 0);
    EXPECT_EQ(dataset.ground_truth.size(), dataset.test_count);
}

TEST(GroundTruthComputer, Correctness) {
    // Simple 2D test
    std::vector<Vector> train = {
        Vector({1.0f, 0.0f}),
        Vector({0.0f, 1.0f}),
        Vector({-1.0f, 0.0f}),
        Vector({0.0f, -1.0f})
    };
    
    std::vector<Vector> queries = {
        Vector({0.9f, 0.1f})  // Closest to (1, 0)
    };
    
    GroundTruthComputer::Config config;
    config.k = 2;
    config.metric = DistanceMetric::L2;
    
    GroundTruthComputer computer(config);
    auto gt = computer.compute(train, queries);
    
    EXPECT_EQ(gt[0][0], 0);  // First neighbor is index 0
}

TEST(RecallEvaluator, PerfectRecall) {
    std::vector<IDMapper::ExternalID> predicted = {1, 2, 3, 4, 5};
    std::vector<IDMapper::ExternalID> ground_truth = {1, 2, 3, 4, 5};
    
    double recall = RecallEvaluator::compute_recall_at_k(
        predicted, ground_truth, 5
    );
    
    EXPECT_DOUBLE_EQ(recall, 1.0);
}

TEST(RecallEvaluator, PartialRecall) {
    std::vector<IDMapper::ExternalID> predicted = {1, 2, 6, 7, 8};
    std::vector<IDMapper::ExternalID> ground_truth = {1, 2, 3, 4, 5};
    
    double recall = RecallEvaluator::compute_recall_at_k(
        predicted, ground_truth, 5
    );
    
    EXPECT_DOUBLE_EQ(recall, 0.4);  // 2 out of 5
}

TEST(BenchmarkUtils, LatencyStatsComputation) {
    std::vector<double> latencies = {1.0, 2.0, 3.0, 4.0, 100.0};
    
    auto stats = BenchmarkUtils::compute_latency_stats(latencies);
    
    EXPECT_DOUBLE_EQ(stats.p50, 3.0);
    EXPECT_DOUBLE_EQ(stats.min, 1.0);
    EXPECT_DOUBLE_EQ(stats.max, 100.0);
}
```

---

### Integration Tests

```cpp
TEST(BenchmarkRunner, EndToEnd) {
    // Create small synthetic dataset
    auto dataset = DatasetLoader::generate_random(
        128,    // dimension
        10000,  // train vectors
        100     // test queries
    );
    
    // Save to disk
    save_dataset_as_fvecs(dataset, "/tmp/test_dataset");
    
    // Run benchmark
    BenchmarkConfig config;
    config.dataset_name = "synthetic";
    config.dataset_path = "/tmp/test_dataset";
    config.L_values = {50, 100};
    config.measured_queries = 100;
    config.generate_plots = false;
    
    BenchmarkRunner runner(config);
    auto result = runner.run();
    
    // Verify results
    EXPECT_GT(result.build_time_seconds, 0);
    EXPECT_GT(result.index_size_mb, 0);
    EXPECT_EQ(result.l_results.size(), 2);
    
    for (const auto& lr : result.l_results) {
        EXPECT_GT(lr.recall_at_10, 0.0);
        EXPECT_LT(lr.recall_at_10, 1.01);  // Allow for rounding
        EXPECT_GT(lr.latency_p99, 0.0);
        EXPECT_GT(lr.qps, 0.0);
    }
}

TEST(RegressionTest, DetectsRegression) {
    std::vector<RegressionTest::Baseline> baselines = {
        {"sift1m", 100, 0.95, 50.0}  // min recall 95%, max p99 50ms
    };
    
    RegressionTest test(baselines);
    
    // Good result
    BenchmarkResult::LResult good;
    good.L = 100;
    good.recall_at_10 = 0.96;
    good.latency_p99 = 45.0;
    
    EXPECT_TRUE(test.check_regression("sift1m", good));
    
    // Bad recall
    BenchmarkResult::LResult bad_recall;
    bad_recall.L = 100;
    bad_recall.recall_at_10 = 0.90;
    bad_recall.latency_p99 = 40.0;
    
    EXPECT_FALSE(test.check_regression("sift1m", bad_recall));
    
    // Bad latency
    BenchmarkResult::LResult bad_latency;
    bad_latency.L = 100;
    bad_latency.recall_at_10 = 0.96;
    bad_latency.latency_p99 = 60.0;
    
    EXPECT_FALSE(test.check_regression("sift1m", bad_latency));
}
```

---

## Standard Datasets

### Recommended Benchmarks

| Dataset | Dimension | Train Size | Test Size | Metric | Source |
|---------|-----------|------------|-----------|--------|--------|
| **SIFT1M** | 128 | 1M | 10k | L2 | http://corpus-texmex.irisa.fr/ |
| **SIFT10M** | 128 | 10M | 10k | L2 | http://corpus-texmex.irisa.fr/ |
| **GIST1M** | 960 | 1M | 1k | L2 | http://corpus-texmex.irisa.fr/ |
| **Deep1B** | 96 | 1B | 10k | L2 | https://yadi.sk/d/11eDCm7Dsn9GA |
| **GloVe-100** | 100 | 1.2M | 10k | Cosine | http://nlp.stanford.edu/projects/glove/ |
| **OpenAI Embeddings** | 1536 | Custom | Custom | Cosine | Custom generation |

---

## CI/CD Integration

### GitHub Actions Workflow

```yaml
name: Performance Regression Tests

on:
  pull_request:
    branches: [main]
  schedule:
    - cron: '0 0 * * 0'  # Weekly on Sunday

jobs:
  benchmark:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Download SIFT1M
      run: |
        wget http://corpus-texmex.irisa.fr/sift.tar.gz
        tar -xzf sift.tar.gz
    
    - name: Build
      run: |
        mkdir build && cd build
        cmake .. -DCMAKE_BUILD_TYPE=Release
        make -j$(nproc)
    
    - name: Run Benchmark
      run: |
        ./build/benchmark_runner \
          --dataset sift1m \
          --dataset-path ./sift \
          --output benchmark_results
    
    - name: Check Regression
      run: |
        ./build/regression_test \
          --baseline baselines/sift1m.json \
          --results benchmark_results/results.json
    
    - name: Upload Results
      uses: actions/upload-artifact@v3
      with:
        name: benchmark-results
        path: benchmark_results/
```

---

## Performance Baselines

### Expected Results (SIFT1M)

| L | Recall@10 | p99 Latency | QPS (8 threads) | Notes |
|---|-----------|-------------|-----------------|-------|
| 50 | 85-90% | 20-30ms | 800-1000 | Fast, lower recall |
| 75 | 90-93% | 30-40ms | 600-800 | Balanced |
| 100 | 93-96% | 40-50ms | 400-600 | Target config |
| 150 | 96-98% | 60-80ms | 300-400 | High recall |
| 200 | 97-99% | 80-100ms | 200-300 | Maximum recall |

**Build Time**: ~5-10 minutes for 1M vectors  
**Index Size**: ~200-300 MB for SIFT1M

---

## Dependencies

### Internal Dependencies
- **COMP-001**: Index builder (for building test indexes)
- **COMP-004**: Search engine (for query execution)
- **COMP-003**: Vector format (for dataset loading)

### External Dependencies
- **C++ Standard Library**: File I/O, containers
- **OpenMP**: Parallel ground truth computation (optional)
- **Python 3.7+**: Plotting utilities (optional)
- **matplotlib**: Plot generation (optional)

---

## Configuration

```cpp
struct BenchmarkConfig {
    // Dataset
    std::string dataset_name = "sift1m";
    std::string dataset_path = "./datasets/sift";
    
    // Build parameters
    uint32_t R = 32;
    uint32_t L_build = 100;
    float alpha = 1.2f;
    
    // Search parameters
    std::vector<uint32_t> L_values = {50, 75, 100, 150, 200};
    uint32_t top_k = 10;
    
    // Measurement
    uint32_t warmup_queries = 100;
    uint32_t measured_queries = 1000;
    uint32_t throughput_duration_sec = 10;
    uint32_t throughput_threads = 8;
    
    // Output
    std::string output_dir = "benchmark_results";
    bool save_json = true;
    bool save_csv = true;
    bool generate_plots = true;
    bool save_html_report = true;
};
```

---

## Future Enhancements

1. **More Datasets** (v1.1)
   - Add support for HDF5, Parquet formats
   - Integrate with ANN Benchmarks suite

2. **Profiling Integration** (v1.1)
   - CPU profiling with perf/vtune
   - Memory profiling with valgrind
   - Flame graph generation

3. **A/B Testing** (v2)
   - Compare two builds side-by-side
   - Statistical significance testing

4. **Cloud Benchmarking** (v2)
   - Run benchmarks on cloud VMs
   - Test on different instance types

5. **Interactive Dashboard** (v2)
   - Web-based result visualization
   - Historical trend tracking

---

## Open Questions

1. **Dataset Hosting**: Self-host datasets or download on-demand? (Download on-demand for CI)
2. **Ground Truth Caching**: Cache computed ground truth? (Yes, cache to disk)
3. **Benchmark Frequency**: How often to run full benchmarks in CI? (Weekly + on-demand)

---

## References

- ANN Benchmarks: http://ann-benchmarks.com/
- SIFT Dataset: http://corpus-texmex.irisa.fr/
- Google Benchmark: https://github.com/google/benchmark
- DiskANN Evaluation: Section 4 of DiskANN paper

---

**Status**: Draft - Ready for review  
**Next Steps**: Team review, implement dataset loaders  
**Completes**: Milestone 1 - Core ANN Engine  
**Blocked By**: COMP-001 (index builder), COMP-004 (search engine)
