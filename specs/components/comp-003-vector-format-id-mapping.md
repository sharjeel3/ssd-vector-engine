# Component: Vector Format and ID Mapping

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

The Vector Format and ID Mapping component defines the data model for vectors and their identifiers throughout the system. It handles the bidirectional mapping between user-provided external IDs and internal system IDs, manages vector normalization for different distance metrics, and provides efficient vector representation utilities.

**Core Responsibilities**:
1. Define vector data structures and format
2. Manage bidirectional ID mapping (external ↔ internal)
3. Normalize vectors for distance metrics (cosine, L2, inner product)
4. Validate vector data quality (no NaN/Inf, non-zero)
5. Provide efficient vector comparison utilities
6. Handle vector serialization/deserialization
7. Support multiple ID types (int64, string in future)

---

## Interface

### Public API

```cpp
namespace ssd_vector {

// ============================================================================
// Vector Representation
// ============================================================================

class Vector {
public:
    // Constructors
    Vector() = default;
    Vector(uint32_t dimension);
    Vector(const float* data, uint32_t dimension);
    Vector(std::vector<float> data);
    
    // Copy/move semantics
    Vector(const Vector&) = default;
    Vector(Vector&&) noexcept = default;
    Vector& operator=(const Vector&) = default;
    Vector& operator=(Vector&&) noexcept = default;
    
    // Access
    uint32_t dimension() const { return dimension_; }
    const float* data() const { return data_.data(); }
    float* data() { return data_.data(); }
    
    float operator[](size_t i) const { return data_[i]; }
    float& operator[](size_t i) { return data_[i]; }
    
    // Normalization (in-place)
    void normalize_l2();      // For cosine similarity
    void normalize_max();     // For numerical stability
    
    // Properties
    float norm_l2() const;    // Euclidean norm
    float norm_max() const;   // Max absolute value
    bool is_normalized() const;
    bool is_valid() const;    // No NaN/Inf
    bool is_zero() const;
    
    // Serialization
    void serialize(std::ostream& out) const;
    static Vector deserialize(std::istream& in);
    
private:
    uint32_t dimension_ = 0;
    std::vector<float> data_;
};

// ============================================================================
// ID Mapping (External ↔ Internal)
// ============================================================================

class IDMapper {
public:
    using InternalID = uint32_t;
    using ExternalID = int64_t;
    
    // Constructor
    explicit IDMapper(uint64_t capacity = 0);
    
    // Add new mapping (returns internal ID)
    InternalID add_vector(ExternalID external_id);
    
    // Lookup operations
    InternalID get_internal_id(ExternalID external_id) const;
    ExternalID get_external_id(InternalID internal_id) const;
    
    // Check existence
    bool has_external_id(ExternalID external_id) const;
    bool has_internal_id(InternalID internal_id) const;
    
    // Bulk operations
    std::vector<InternalID> get_internal_ids(
        const std::vector<ExternalID>& external_ids
    ) const;
    
    std::vector<ExternalID> get_external_ids(
        const std::vector<InternalID>& internal_ids
    ) const;
    
    // Statistics
    uint64_t size() const { return next_internal_id_; }
    uint64_t capacity() const { return external_to_internal_.bucket_count(); }
    
    // Serialization (for metadata.bin)
    void serialize(std::ostream& out) const;
    static IDMapper deserialize(std::istream& in);
    
    // Clear all mappings
    void clear();
    
    // Reserve capacity (optimization)
    void reserve(uint64_t capacity);
    
private:
    // Bidirectional mapping
    std::unordered_map<ExternalID, InternalID> external_to_internal_;
    std::vector<ExternalID> internal_to_external_;
    InternalID next_internal_id_ = 0;
};

// ============================================================================
// Vector Normalization Utilities
// ============================================================================

class VectorNormalizer {
public:
    enum class NormType {
        NONE,           // No normalization
        L2,             // L2 normalization (for cosine)
        MAX_ABS         // Divide by max absolute value
    };
    
    // Normalize single vector (in-place)
    static void normalize(float* data, uint32_t dimension, NormType type);
    
    // Normalize batch of vectors (in-place)
    static void normalize_batch(
        float* data, 
        uint64_t count,
        uint32_t dimension,
        NormType type
    );
    
    // Compute norm
    static float compute_norm_l2(const float* data, uint32_t dimension);
    static float compute_norm_max(const float* data, uint32_t dimension);
    
    // Check if normalized
    static bool is_normalized_l2(const float* data, uint32_t dimension, float epsilon = 1e-6f);
};

// ============================================================================
// Vector Validation
// ============================================================================

class VectorValidator {
public:
    struct ValidationResult {
        bool valid = true;
        std::string error_message;
        
        // Specific issues
        bool has_nan = false;
        bool has_inf = false;
        bool is_zero = false;
        bool dimension_mismatch = false;
        
        operator bool() const { return valid; }
    };
    
    // Validate single vector
    static ValidationResult validate(
        const float* data,
        uint32_t dimension,
        uint32_t expected_dimension
    );
    
    // Validate batch of vectors
    struct BatchValidationResult {
        uint64_t total_count = 0;
        uint64_t valid_count = 0;
        uint64_t invalid_count = 0;
        std::vector<uint64_t> invalid_indices;
        std::map<std::string, uint64_t> error_counts;
    };
    
    static BatchValidationResult validate_batch(
        const float* data,
        uint64_t count,
        uint32_t dimension
    );
    
private:
    static bool has_nan(const float* data, uint32_t dimension);
    static bool has_inf(const float* data, uint32_t dimension);
    static bool is_zero_vector(const float* data, uint32_t dimension);
};

// ============================================================================
// Vector I/O Utilities
// ============================================================================

class VectorIO {
public:
    // Load vectors from various formats
    struct LoadedVectors {
        std::vector<float> data;        // Flattened: [v0d0, v0d1, ..., v1d0, v1d1, ...]
        uint64_t count;
        uint32_t dimension;
    };
    
    // Load from binary format (our format)
    static LoadedVectors load_binary(const std::string& filepath);
    
    // Load from FVECS format (ANN benchmarks)
    static LoadedVectors load_fvecs(const std::string& filepath);
    
    // Load from NumPy .npy format
    static LoadedVectors load_npy(const std::string& filepath);
    
    // Save to binary format
    static void save_binary(
        const std::string& filepath,
        const float* data,
        uint64_t count,
        uint32_t dimension
    );
    
    // Save to FVECS format (for benchmarking)
    static void save_fvecs(
        const std::string& filepath,
        const float* data,
        uint64_t count,
        uint32_t dimension
    );
};

// ============================================================================
// Vector Batch (Collection of Vectors with IDs)
// ============================================================================

class VectorBatch {
public:
    struct VectorWithID {
        IDMapper::ExternalID id;
        Vector vector;
        std::map<std::string, std::string> metadata;  // Optional metadata
    };
    
    VectorBatch() = default;
    explicit VectorBatch(uint32_t expected_dimension);
    
    // Add vector
    void add(const VectorWithID& vec);
    void add(IDMapper::ExternalID id, const Vector& vec);
    
    // Batch operations
    void reserve(size_t capacity);
    size_t size() const { return vectors_.size(); }
    bool empty() const { return vectors_.empty(); }
    void clear();
    
    // Access
    const VectorWithID& operator[](size_t i) const { return vectors_[i]; }
    VectorWithID& operator[](size_t i) { return vectors_[i]; }
    
    const std::vector<VectorWithID>& vectors() const { return vectors_; }
    
    // Properties
    uint32_t dimension() const { return dimension_; }
    
    // Validation
    VectorValidator::BatchValidationResult validate() const;
    
    // Normalization (apply to all vectors)
    void normalize(VectorNormalizer::NormType type);
    
private:
    uint32_t dimension_ = 0;
    std::vector<VectorWithID> vectors_;
};

} // namespace ssd_vector
```

---

## Implementation Details

### Vector Class Implementation

#### Constructor and Access

```cpp
Vector::Vector(const float* data, uint32_t dimension)
    : dimension_(dimension), data_(data, data + dimension) {
    if (!is_valid()) {
        throw std::invalid_argument("Vector contains NaN or Inf values");
    }
}

Vector::Vector(std::vector<float> data)
    : dimension_(static_cast<uint32_t>(data.size())), 
      data_(std::move(data)) {
    if (!is_valid()) {
        throw std::invalid_argument("Vector contains NaN or Inf values");
    }
}
```

#### L2 Normalization

```cpp
void Vector::normalize_l2() {
    float norm = norm_l2();
    
    if (norm < 1e-10f) {
        throw std::runtime_error("Cannot normalize zero vector");
    }
    
    float inv_norm = 1.0f / norm;
    for (float& val : data_) {
        val *= inv_norm;
    }
}

float Vector::norm_l2() const {
    float sum = 0.0f;
    
    // SIMD optimization possible here
    for (float val : data_) {
        sum += val * val;
    }
    
    return std::sqrt(sum);
}

bool Vector::is_normalized() const {
    float norm = norm_l2();
    return std::abs(norm - 1.0f) < 1e-6f;
}
```

#### Validation

```cpp
bool Vector::is_valid() const {
    for (float val : data_) {
        if (!std::isfinite(val)) {
            return false;
        }
    }
    return true;
}

bool Vector::is_zero() const {
    for (float val : data_) {
        if (std::abs(val) > 1e-10f) {
            return false;
        }
    }
    return true;
}
```

---

### ID Mapper Implementation

#### Adding Mappings

```cpp
IDMapper::InternalID IDMapper::add_vector(ExternalID external_id) {
    // Check if already exists
    auto it = external_to_internal_.find(external_id);
    if (it != external_to_internal_.end()) {
        throw std::invalid_argument(
            "External ID already exists: " + std::to_string(external_id)
        );
    }
    
    // Assign new internal ID
    InternalID internal_id = next_internal_id_++;
    
    // Create bidirectional mapping
    external_to_internal_[external_id] = internal_id;
    internal_to_external_.push_back(external_id);
    
    return internal_id;
}
```

#### Lookup Operations

```cpp
IDMapper::InternalID IDMapper::get_internal_id(ExternalID external_id) const {
    auto it = external_to_internal_.find(external_id);
    if (it == external_to_internal_.end()) {
        throw std::out_of_range(
            "External ID not found: " + std::to_string(external_id)
        );
    }
    return it->second;
}

IDMapper::ExternalID IDMapper::get_external_id(InternalID internal_id) const {
    if (internal_id >= internal_to_external_.size()) {
        throw std::out_of_range(
            "Internal ID out of range: " + std::to_string(internal_id)
        );
    }
    return internal_to_external_[internal_id];
}

bool IDMapper::has_external_id(ExternalID external_id) const {
    return external_to_internal_.find(external_id) != external_to_internal_.end();
}
```

#### Bulk Operations

```cpp
std::vector<IDMapper::InternalID> IDMapper::get_internal_ids(
    const std::vector<ExternalID>& external_ids
) const {
    std::vector<InternalID> result;
    result.reserve(external_ids.size());
    
    for (ExternalID ext_id : external_ids) {
        result.push_back(get_internal_id(ext_id));
    }
    
    return result;
}
```

#### Serialization

```cpp
void IDMapper::serialize(std::ostream& out) const {
    // Write count
    uint64_t count = internal_to_external_.size();
    out.write(reinterpret_cast<const char*>(&count), sizeof(count));
    
    // Write ID mappings (internal order)
    out.write(
        reinterpret_cast<const char*>(internal_to_external_.data()),
        count * sizeof(ExternalID)
    );
}

IDMapper IDMapper::deserialize(std::istream& in) {
    IDMapper mapper;
    
    // Read count
    uint64_t count;
    in.read(reinterpret_cast<char*>(&count), sizeof(count));
    
    // Read ID mappings
    mapper.internal_to_external_.resize(count);
    in.read(
        reinterpret_cast<char*>(mapper.internal_to_external_.data()),
        count * sizeof(ExternalID)
    );
    
    // Rebuild reverse mapping
    mapper.external_to_internal_.reserve(count);
    for (uint32_t i = 0; i < count; i++) {
        mapper.external_to_internal_[mapper.internal_to_external_[i]] = i;
    }
    
    mapper.next_internal_id_ = static_cast<InternalID>(count);
    
    return mapper;
}
```

---

### Vector Normalizer Implementation

#### L2 Normalization (SIMD Optimized)

```cpp
void VectorNormalizer::normalize(float* data, uint32_t dimension, NormType type) {
    switch (type) {
        case NormType::NONE:
            return;
            
        case NormType::L2: {
            float norm = compute_norm_l2(data, dimension);
            if (norm < 1e-10f) {
                throw std::runtime_error("Cannot normalize zero vector");
            }
            
            float inv_norm = 1.0f / norm;
            
            // SIMD optimization
            #ifdef __AVX2__
            normalize_l2_avx2(data, dimension, inv_norm);
            #else
            for (uint32_t i = 0; i < dimension; i++) {
                data[i] *= inv_norm;
            }
            #endif
            break;
        }
        
        case NormType::MAX_ABS: {
            float max_val = compute_norm_max(data, dimension);
            if (max_val < 1e-10f) {
                throw std::runtime_error("Cannot normalize zero vector");
            }
            
            float inv_max = 1.0f / max_val;
            for (uint32_t i = 0; i < dimension; i++) {
                data[i] *= inv_max;
            }
            break;
        }
    }
}

float VectorNormalizer::compute_norm_l2(const float* data, uint32_t dimension) {
    float sum = 0.0f;
    
    #ifdef __AVX2__
    // AVX2 SIMD implementation
    __m256 sum_vec = _mm256_setzero_ps();
    
    uint32_t i = 0;
    for (; i + 8 <= dimension; i += 8) {
        __m256 v = _mm256_loadu_ps(data + i);
        sum_vec = _mm256_fmadd_ps(v, v, sum_vec);  // v*v + sum
    }
    
    // Horizontal sum
    float temp[8];
    _mm256_storeu_ps(temp, sum_vec);
    for (int j = 0; j < 8; j++) {
        sum += temp[j];
    }
    
    // Handle remaining elements
    for (; i < dimension; i++) {
        sum += data[i] * data[i];
    }
    #else
    // Scalar fallback
    for (uint32_t i = 0; i < dimension; i++) {
        sum += data[i] * data[i];
    }
    #endif
    
    return std::sqrt(sum);
}
```

#### Batch Normalization

```cpp
void VectorNormalizer::normalize_batch(
    float* data,
    uint64_t count,
    uint32_t dimension,
    NormType type
) {
    // Parallelize across vectors
    #pragma omp parallel for
    for (uint64_t i = 0; i < count; i++) {
        normalize(data + i * dimension, dimension, type);
    }
}
```

---

### Vector Validator Implementation

#### Single Vector Validation

```cpp
VectorValidator::ValidationResult VectorValidator::validate(
    const float* data,
    uint32_t dimension,
    uint32_t expected_dimension
) {
    ValidationResult result;
    
    // Check dimension
    if (dimension != expected_dimension) {
        result.valid = false;
        result.dimension_mismatch = true;
        result.error_message = "Dimension mismatch: expected " + 
            std::to_string(expected_dimension) + ", got " + 
            std::to_string(dimension);
        return result;
    }
    
    // Check for NaN
    if (has_nan(data, dimension)) {
        result.valid = false;
        result.has_nan = true;
        result.error_message = "Vector contains NaN values";
        return result;
    }
    
    // Check for Inf
    if (has_inf(data, dimension)) {
        result.valid = false;
        result.has_inf = true;
        result.error_message = "Vector contains Inf values";
        return result;
    }
    
    // Check for zero vector (warning, not error)
    if (is_zero_vector(data, dimension)) {
        result.is_zero = true;
        // Note: not marking as invalid, but flagging
    }
    
    return result;
}

bool VectorValidator::has_nan(const float* data, uint32_t dimension) {
    for (uint32_t i = 0; i < dimension; i++) {
        if (std::isnan(data[i])) {
            return true;
        }
    }
    return false;
}

bool VectorValidator::has_inf(const float* data, uint32_t dimension) {
    for (uint32_t i = 0; i < dimension; i++) {
        if (std::isinf(data[i])) {
            return true;
        }
    }
    return false;
}

bool VectorValidator::is_zero_vector(const float* data, uint32_t dimension) {
    for (uint32_t i = 0; i < dimension; i++) {
        if (std::abs(data[i]) > 1e-10f) {
            return false;
        }
    }
    return true;
}
```

#### Batch Validation

```cpp
VectorValidator::BatchValidationResult VectorValidator::validate_batch(
    const float* data,
    uint64_t count,
    uint32_t dimension
) {
    BatchValidationResult result;
    result.total_count = count;
    
    for (uint64_t i = 0; i < count; i++) {
        const float* vec = data + i * dimension;
        auto validation = validate(vec, dimension, dimension);
        
        if (validation.valid) {
            result.valid_count++;
        } else {
            result.invalid_count++;
            result.invalid_indices.push_back(i);
            result.error_counts[validation.error_message]++;
        }
    }
    
    return result;
}
```

---

### Vector I/O Implementation

#### Load FVECS Format

```cpp
VectorIO::LoadedVectors VectorIO::load_fvecs(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open file: " + filepath);
    }
    
    LoadedVectors result;
    std::vector<float> all_data;
    
    // Read first vector to get dimension
    int32_t first_dim;
    file.read(reinterpret_cast<char*>(&first_dim), sizeof(int32_t));
    if (!file) {
        throw std::runtime_error("Failed to read dimension");
    }
    
    result.dimension = static_cast<uint32_t>(first_dim);
    file.seekg(0);  // Reset to beginning
    
    // Read all vectors
    while (true) {
        int32_t dim;
        file.read(reinterpret_cast<char*>(&dim), sizeof(int32_t));
        if (!file) break;
        
        if (dim != static_cast<int32_t>(result.dimension)) {
            throw std::runtime_error("Inconsistent dimensions in FVECS file");
        }
        
        std::vector<float> vec(dim);
        file.read(reinterpret_cast<char*>(vec.data()), dim * sizeof(float));
        if (!file) {
            throw std::runtime_error("Failed to read vector data");
        }
        
        all_data.insert(all_data.end(), vec.begin(), vec.end());
        result.count++;
    }
    
    result.data = std::move(all_data);
    return result;
}
```

#### Load Binary Format

```cpp
VectorIO::LoadedVectors VectorIO::load_binary(const std::string& filepath) {
    // Use VectorFile from storage layer
    auto file = storage::VectorFile::open(filepath);
    
    LoadedVectors result;
    result.count = file->count();
    result.dimension = file->dimension();
    
    // Copy all vector data
    size_t total_floats = result.count * result.dimension;
    result.data.resize(total_floats);
    
    for (uint64_t i = 0; i < result.count; i++) {
        const float* vec = file->get_vector(i);
        std::copy(
            vec, 
            vec + result.dimension,
            result.data.begin() + i * result.dimension
        );
    }
    
    return result;
}
```

---

### Vector Batch Implementation

```cpp
void VectorBatch::add(IDMapper::ExternalID id, const Vector& vec) {
    if (dimension_ == 0) {
        dimension_ = vec.dimension();
    } else if (vec.dimension() != dimension_) {
        throw std::invalid_argument(
            "Vector dimension mismatch: expected " + 
            std::to_string(dimension_) + ", got " + 
            std::to_string(vec.dimension())
        );
    }
    
    vectors_.push_back({id, vec, {}});
}

VectorValidator::BatchValidationResult VectorBatch::validate() const {
    VectorValidator::BatchValidationResult result;
    result.total_count = vectors_.size();
    
    for (size_t i = 0; i < vectors_.size(); i++) {
        const auto& vec = vectors_[i].vector;
        auto validation = VectorValidator::validate(
            vec.data(),
            vec.dimension(),
            dimension_
        );
        
        if (validation.valid) {
            result.valid_count++;
        } else {
            result.invalid_count++;
            result.invalid_indices.push_back(i);
            result.error_counts[validation.error_message]++;
        }
    }
    
    return result;
}

void VectorBatch::normalize(VectorNormalizer::NormType type) {
    for (auto& vec_with_id : vectors_) {
        VectorNormalizer::normalize(
            vec_with_id.vector.data(),
            vec_with_id.vector.dimension(),
            type
        );
    }
}
```

---

## Data Model Specifications

### ID Format Constraints

#### Internal IDs
- **Type**: `uint32_t` (32-bit unsigned integer)
- **Range**: 0 to 4,294,967,295 (4B vectors)
- **Properties**:
  - Dense: Assigned sequentially (0, 1, 2, ...)
  - Immutable: Never change once assigned
  - Compact: Used as array indices

#### External IDs (v1)
- **Type**: `int64_t` (64-bit signed integer)
- **Range**: -9,223,372,036,854,775,808 to 9,223,372,036,854,775,807
- **Properties**:
  - User-provided
  - Can be negative (though not recommended)
  - Must be unique per collection
  - Sparse: Arbitrary values allowed

#### Future: String IDs (v2)
- **Type**: UTF-8 string
- **Max length**: 256 bytes
- **Properties**:
  - Arbitrary strings (UUIDs, URLs, etc.)
  - Stored with offset table for efficient lookup

---

### Vector Format Constraints

#### Dimension Constraints
- **Minimum**: 64
- **Maximum**: 4096 (v1)
- **Typical**: 128, 384, 768, 1536, 2048
- **Alignment**: Prefer multiples of 8 for SIMD efficiency

#### Data Type
- **v1**: `float32` (IEEE 754 single precision)
- **Future**: `float16` (half precision for compression)

#### Value Constraints
- **No NaN**: All values must be finite
- **No Inf**: All values must be finite
- **Non-zero**: At least one non-zero value (for normalization)
- **Range**: Typically [-1, 1] after normalization

---

### Memory Layout

#### Flattened Storage
Vectors stored in row-major order:
```
[v0d0, v0d1, ..., v0dN, v1d0, v1d1, ..., v1dN, ...]
```

**Access pattern**:
```cpp
float* get_vector(float* data, uint64_t index, uint32_t dimension) {
    return data + (index * dimension);
}
```

#### Alignment
- Vectors aligned to 64-byte boundaries (cache line)
- SIMD operations require 32-byte alignment (AVX2)

---

## Performance Requirements

### ID Mapping Performance

| Operation | Target | Notes |
|-----------|--------|-------|
| **Add mapping** | O(1) amortized | Hash table insert |
| **Lookup external→internal** | O(1) average | Hash table lookup |
| **Lookup internal→external** | O(1) | Direct array access |
| **Bulk conversion** | < 1μs per ID | 1M IDs in ~1 second |
| **Memory overhead** | 12 bytes per ID | Hash table + vector |

### Normalization Performance

| Operation | Target | Notes |
|-----------|--------|-------|
| **Single vector (scalar)** | ~100 ns | For D=1536 |
| **Single vector (SIMD)** | ~50 ns | 2x speedup with AVX2 |
| **Batch (1000 vectors)** | < 100 μs | Parallelized |

### Validation Performance

| Operation | Target | Notes |
|-----------|--------|-------|
| **Single vector validation** | < 50 ns | Simple checks |
| **Batch validation (1M)** | < 100 ms | Parallel scan |

---

## Error Handling

### ID Mapping Errors

```cpp
class IDMappingException : public std::exception {
public:
    enum class ErrorCode {
        DUPLICATE_EXTERNAL_ID,
        EXTERNAL_ID_NOT_FOUND,
        INTERNAL_ID_OUT_OF_RANGE
    };
    
    IDMappingException(ErrorCode code, const std::string& message)
        : code_(code), message_(message) {}
    
    ErrorCode code() const { return code_; }
    const char* what() const noexcept override { return message_.c_str(); }
    
private:
    ErrorCode code_;
    std::string message_;
};
```

### Vector Validation Errors

```cpp
class VectorValidationException : public std::exception {
public:
    enum class ErrorCode {
        INVALID_DIMENSION,
        CONTAINS_NAN,
        CONTAINS_INF,
        ZERO_VECTOR,
        DIMENSION_MISMATCH
    };
    
    VectorValidationException(ErrorCode code, const std::string& message)
        : code_(code), message_(message) {}
    
    ErrorCode code() const { return code_; }
    const char* what() const noexcept override { return message_.c_str(); }
    
private:
    ErrorCode code_;
    std::string message_;
};
```

---

## Testing Strategy

### Unit Tests

```cpp
TEST(Vector, Construction) {
    std::vector<float> data = {1.0f, 2.0f, 3.0f};
    Vector v(data);
    
    EXPECT_EQ(v.dimension(), 3);
    EXPECT_FLOAT_EQ(v[0], 1.0f);
    EXPECT_FLOAT_EQ(v[1], 2.0f);
    EXPECT_FLOAT_EQ(v[2], 3.0f);
}

TEST(Vector, L2Normalization) {
    Vector v({3.0f, 4.0f, 0.0f});
    v.normalize_l2();
    
    EXPECT_FLOAT_EQ(v[0], 0.6f);
    EXPECT_FLOAT_EQ(v[1], 0.8f);
    EXPECT_FLOAT_EQ(v[2], 0.0f);
    EXPECT_TRUE(v.is_normalized());
}

TEST(Vector, Validation) {
    // Valid vector
    Vector v1({1.0f, 2.0f, 3.0f});
    EXPECT_TRUE(v1.is_valid());
    
    // NaN vector
    Vector v2({1.0f, NAN, 3.0f});
    EXPECT_FALSE(v2.is_valid());
    
    // Inf vector
    Vector v3({1.0f, INFINITY, 3.0f});
    EXPECT_FALSE(v3.is_valid());
}

TEST(IDMapper, AddAndLookup) {
    IDMapper mapper;
    
    // Add mappings
    auto id0 = mapper.add_vector(1000);
    auto id1 = mapper.add_vector(2000);
    auto id2 = mapper.add_vector(3000);
    
    EXPECT_EQ(id0, 0u);
    EXPECT_EQ(id1, 1u);
    EXPECT_EQ(id2, 2u);
    
    // Lookup external -> internal
    EXPECT_EQ(mapper.get_internal_id(1000), 0u);
    EXPECT_EQ(mapper.get_internal_id(2000), 1u);
    
    // Lookup internal -> external
    EXPECT_EQ(mapper.get_external_id(0), 1000);
    EXPECT_EQ(mapper.get_external_id(1), 2000);
}

TEST(IDMapper, DuplicateExternalID) {
    IDMapper mapper;
    mapper.add_vector(1000);
    
    EXPECT_THROW(mapper.add_vector(1000), IDMappingException);
}

TEST(VectorNormalizer, L2Norm) {
    float data[] = {3.0f, 4.0f, 0.0f};
    VectorNormalizer::normalize(data, 3, VectorNormalizer::NormType::L2);
    
    EXPECT_FLOAT_EQ(data[0], 0.6f);
    EXPECT_FLOAT_EQ(data[1], 0.8f);
    EXPECT_FLOAT_EQ(data[2], 0.0f);
    
    float norm = VectorNormalizer::compute_norm_l2(data, 3);
    EXPECT_NEAR(norm, 1.0f, 1e-6f);
}

TEST(VectorValidator, DetectNaN) {
    float data[] = {1.0f, NAN, 3.0f};
    auto result = VectorValidator::validate(data, 3, 3);
    
    EXPECT_FALSE(result.valid);
    EXPECT_TRUE(result.has_nan);
}

TEST(VectorBatch, AddAndValidate) {
    VectorBatch batch(3);
    
    batch.add(1, Vector({1.0f, 2.0f, 3.0f}));
    batch.add(2, Vector({4.0f, 5.0f, 6.0f}));
    
    EXPECT_EQ(batch.size(), 2);
    
    auto validation = batch.validate();
    EXPECT_EQ(validation.valid_count, 2);
    EXPECT_EQ(validation.invalid_count, 0);
}
```

### Integration Tests

```cpp
TEST(VectorIO, LoadAndSaveFVECS) {
    // Create test vectors
    std::vector<float> data(1000 * 128);
    generate_random_vectors(data.data(), 1000, 128);
    
    // Save as FVECS
    VectorIO::save_fvecs("test.fvecs", data.data(), 1000, 128);
    
    // Load back
    auto loaded = VectorIO::load_fvecs("test.fvecs");
    
    EXPECT_EQ(loaded.count, 1000);
    EXPECT_EQ(loaded.dimension, 128);
    
    // Verify data
    for (size_t i = 0; i < data.size(); i++) {
        EXPECT_FLOAT_EQ(loaded.data[i], data[i]);
    }
}

TEST(IDMapper, Serialization) {
    IDMapper mapper;
    for (int64_t i = 0; i < 1000; i++) {
        mapper.add_vector(i * 100);
    }
    
    // Serialize
    std::ostringstream out;
    mapper.serialize(out);
    
    // Deserialize
    std::istringstream in(out.str());
    auto loaded = IDMapper::deserialize(in);
    
    // Verify
    EXPECT_EQ(loaded.size(), 1000);
    for (int64_t i = 0; i < 1000; i++) {
        EXPECT_EQ(loaded.get_external_id(i), i * 100);
    }
}
```

### Performance Tests

```cpp
BENCHMARK(VectorNormalizer, L2NormSingle) {
    float data[1536];
    generate_random_vector(data, 1536);
    
    for (int i = 0; i < 1000000; i++) {
        VectorNormalizer::normalize(data, 1536, VectorNormalizer::NormType::L2);
    }
}

BENCHMARK(IDMapper, LookupPerformance) {
    IDMapper mapper;
    for (int64_t i = 0; i < 1000000; i++) {
        mapper.add_vector(i);
    }
    
    // Benchmark lookups
    auto start = high_resolution_clock::now();
    for (int i = 0; i < 1000000; i++) {
        auto internal = mapper.get_internal_id(i);
        benchmark::DoNotOptimize(internal);
    }
    auto end = high_resolution_clock::now();
    
    // Target: < 1μs per lookup
}
```

---

## Dependencies

### Internal Dependencies
- **ADR-001**: Storage format (for serialization)
- **COMP-002**: Storage layer (for VectorIO)

### External Dependencies
- **C++ Standard Library**: STL containers, algorithms
- **Optional: Intel SIMD Intrinsics**: AVX2 for normalization
- **Optional: OpenMP**: Parallel batch operations

---

## Configuration

```cpp
struct VectorFormatConfig {
    // Dimension constraints
    uint32_t min_dimension = 64;
    uint32_t max_dimension = 4096;
    
    // Validation options
    bool allow_zero_vectors = false;
    bool strict_validation = true;    // Fail on invalid vectors
    
    // Normalization
    VectorNormalizer::NormType default_norm = VectorNormalizer::NormType::L2;
    bool auto_normalize = false;       // Auto-normalize on add
    
    // ID constraints
    bool allow_negative_ids = true;
    int64_t max_external_id = INT64_MAX;
};
```

---

## Future Enhancements

1. **String IDs** (v2)
   - Support UUID and arbitrary string IDs
   - Efficient string storage and lookup

2. **Float16 Support** (v1.1)
   - Half-precision vectors for 2x compression
   - Transparent conversion for distance computation

3. **Quantization** (v2)
   - Product Quantization (PQ)
   - Scalar Quantization (SQ)

4. **Sparse Vectors** (v2)
   - Support for sparse vector representation
   - Efficient sparse distance computation

---

## Open Questions

1. **Negative External IDs**: Allow or disallow? (Proposed: allow but document)
2. **Zero Vectors**: Hard error or warning? (Proposed: warning, allow)
3. **ID Reuse**: Allow reusing deleted IDs? (Proposed: no, for simplicity)

---

## References

- IEEE 754 Floating Point: https://en.wikipedia.org/wiki/IEEE_754
- SIMD Intrinsics Guide: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/
- FVECS Format: http://corpus-texmex.irisa.fr/

---

**Status**: Draft - Ready for review  
**Next Steps**: Team review, implement core classes  
**Blocks**: COMP-001 (needs ID mapping), COMP-004 (needs vector utilities)  
**Blocked By**: None (foundational component)
