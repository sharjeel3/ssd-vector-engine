# Component: Index Storage Layer

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

The Index Storage Layer provides low-level file I/O abstractions for persisting and loading DiskANN indexes. It implements the storage format specified in ADR-001, handling binary serialization, memory mapping, checksum verification, and format validation.

**Core Responsibilities**:
1. Serialize vectors, graph, and metadata to binary files
2. Deserialize and memory-map index files
3. Validate file format and data integrity
4. Compute and verify SHA-256 checksums
5. Handle manifest.json generation and parsing
6. Provide efficient zero-copy access via mmap
7. Abstract platform differences (Linux, macOS, Windows)

---

## Interface

### Public API

```cpp
namespace ssd_vector {
namespace storage {

// ============================================================================
// Vector File Operations
// ============================================================================

class VectorFile {
public:
    struct Header {
        uint64_t magic;          // 0x5644415441 ("VDATA")
        uint32_t format_version; // 1
        uint64_t vector_count;
        uint32_t dimension;
        uint32_t data_type;      // 0=f32, 1=f16
        uint32_t alignment;      // Byte alignment (64)
        uint8_t reserved[224];
    };
    
    // Write vectors to file
    static void write(
        const std::string& filepath,
        const float* vectors,
        uint64_t count,
        uint32_t dimension
    );
    
    // Memory-map vectors for reading
    static std::unique_ptr<VectorFile> open(
        const std::string& filepath
    );
    
    // Access methods
    uint64_t count() const { return header_.vector_count; }
    uint32_t dimension() const { return header_.dimension; }
    
    // Get pointer to vector i (zero-copy)
    const float* get_vector(uint64_t i) const {
        assert(i < header_.vector_count);
        return data_ + (i * header_.dimension);
    }
    
    // Validate file format
    bool validate() const;
    
    // Close and unmap
    void close();
    
    ~VectorFile();

private:
    VectorFile() = default;
    
    Header header_;
    const float* data_;      // Memory-mapped data
    void* mmap_ptr_;         // mmap handle
    size_t mmap_size_;
    int fd_;                 // File descriptor
};

// ============================================================================
// Graph File Operations
// ============================================================================

class GraphFile {
public:
    struct Header {
        uint64_t magic;          // 0x4752415048 ("GRAPH")
        uint32_t format_version; // 1
        uint64_t node_count;
        uint32_t max_degree;     // R parameter
        uint32_t avg_degree;
        uint8_t reserved[228];
    };
    
    struct Node {
        uint32_t degree;         // Actual number of edges
        uint32_t neighbors[0];   // Variable-length array
        
        const uint32_t* begin() const { return neighbors; }
        const uint32_t* end() const { return neighbors + degree; }
    };
    
    // Write graph to file
    static void write(
        const std::string& filepath,
        const std::vector<std::vector<uint32_t>>& adjacency_lists
    );
    
    // Memory-map graph for reading
    static std::unique_ptr<GraphFile> open(
        const std::string& filepath
    );
    
    // Access methods
    uint64_t node_count() const { return header_.node_count; }
    uint32_t max_degree() const { return header_.max_degree; }
    
    // Get node by ID (zero-copy access via offset table)
    const Node* get_node(uint32_t node_id) const {
        assert(node_id < header_.node_count);
        size_t offset = offset_table_[node_id];
        return reinterpret_cast<const Node*>(data_ + offset);
    }
    
    // Validate file format
    bool validate() const;
    
    // Close and unmap
    void close();
    
    ~GraphFile();

private:
    GraphFile() = default;
    
    Header header_;
    const uint64_t* offset_table_;  // Offsets to adjacency lists
    const uint8_t* data_;           // Memory-mapped data
    void* mmap_ptr_;
    size_t mmap_size_;
    int fd_;
};

// ============================================================================
// Metadata File Operations
// ============================================================================

class MetadataFile {
public:
    struct Header {
        uint64_t magic;          // 0x4d45544144 ("METAD")
        uint32_t format_version; // 1
        uint64_t entry_count;
        uint32_t id_type;        // 0=int64, 1=string
        uint8_t reserved[232];
    };
    
    // Write metadata to file
    static void write(
        const std::string& filepath,
        const std::vector<int64_t>& external_ids
    );
    
    // Memory-map metadata for reading
    static std::unique_ptr<MetadataFile> open(
        const std::string& filepath
    );
    
    // Access methods
    uint64_t entry_count() const { return header_.entry_count; }
    
    // Get external ID for internal ID
    int64_t get_external_id(uint32_t internal_id) const {
        assert(internal_id < header_.entry_count);
        return id_data_[internal_id];
    }
    
    // Validate file format
    bool validate() const;
    
    // Close and unmap
    void close();
    
    ~MetadataFile();

private:
    MetadataFile() = default;
    
    Header header_;
    const int64_t* id_data_;  // Memory-mapped ID array
    void* mmap_ptr_;
    size_t mmap_size_;
    int fd_;
};

// ============================================================================
// Manifest Operations (JSON)
// ============================================================================

class ManifestFile {
public:
    struct Manifest {
        std::string version;
        uint32_t format_version;
        std::string created_at;
        uint64_t vector_count;
        uint32_t dimension;
        std::string metric;
        
        struct BuildParams {
            uint32_t R;
            uint32_t L;
            float alpha;
        } build_parameters;
        
        struct Files {
            std::string vectors;
            std::string graph;
            std::string metadata;
        } files;
        
        std::map<std::string, std::string> checksums;
    };
    
    // Write manifest
    static void write(
        const std::string& filepath,
        const Manifest& manifest
    );
    
    // Read manifest
    static Manifest read(const std::string& filepath);
    
    // Validate manifest contents
    static bool validate(const Manifest& manifest);
};

// ============================================================================
// Checksum Operations
// ============================================================================

class ChecksumFile {
public:
    // Compute SHA-256 checksum of file
    static std::string compute_sha256(const std::string& filepath);
    
    // Write checksums file
    static void write(
        const std::string& filepath,
        const std::map<std::string, std::string>& checksums
    );
    
    // Read and parse checksums file
    static std::map<std::string, std::string> read(
        const std::string& filepath
    );
    
    // Verify file against expected checksum
    static bool verify(
        const std::string& filepath,
        const std::string& expected_checksum
    );
};

// ============================================================================
// Complete Index Operations (High-Level API)
// ============================================================================

class IndexStorage {
public:
    struct Index {
        std::unique_ptr<VectorFile> vectors;
        std::unique_ptr<GraphFile> graph;
        std::unique_ptr<MetadataFile> metadata;
        ManifestFile::Manifest manifest;
    };
    
    // Load complete index from directory
    static std::unique_ptr<Index> load(
        const std::string& index_dir,
        bool verify_checksums = true
    );
    
    // Validate index integrity
    static bool validate_index(const std::string& index_dir);
    
    // Get index statistics
    struct IndexStats {
        uint64_t vector_count;
        uint32_t dimension;
        size_t total_size_bytes;
        std::string metric;
        std::chrono::system_clock::time_point created_at;
    };
    
    static IndexStats get_stats(const std::string& index_dir);
};

} // namespace storage
} // namespace ssd_vector
```

---

## Implementation Details

### Memory Mapping (mmap)

#### Unix/Linux/macOS Implementation

```cpp
void* VectorFile::mmap_file(const std::string& filepath, size_t& size) {
    // Open file
    fd_ = open(filepath.c_str(), O_RDONLY);
    if (fd_ == -1) {
        throw StorageException("Failed to open file: " + filepath);
    }
    
    // Get file size
    struct stat sb;
    if (fstat(fd_, &sb) == -1) {
        close(fd_);
        throw StorageException("Failed to stat file: " + filepath);
    }
    size = sb.st_size;
    
    // Memory map (read-only, shared)
    void* ptr = mmap(
        nullptr,           // Let kernel choose address
        size,              // Map entire file
        PROT_READ,         // Read-only
        MAP_SHARED,        // Share across processes
        fd_,               // File descriptor
        0                  // Offset from start
    );
    
    if (ptr == MAP_FAILED) {
        close(fd_);
        throw StorageException("mmap failed: " + filepath);
    }
    
    // Optional: Advise kernel about access pattern
    madvise(ptr, size, MADV_RANDOM);  // Random access expected
    
    return ptr;
}

void VectorFile::unmap_file(void* ptr, size_t size) {
    if (ptr != nullptr) {
        munmap(ptr, size);
    }
    if (fd_ != -1) {
        close(fd_);
        fd_ = -1;
    }
}
```

#### Windows Implementation

```cpp
void* VectorFile::mmap_file_windows(const std::string& filepath, size_t& size) {
    // Open file
    HANDLE file_handle = CreateFileA(
        filepath.c_str(),
        GENERIC_READ,
        FILE_SHARE_READ,
        nullptr,
        OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL,
        nullptr
    );
    
    if (file_handle == INVALID_HANDLE_VALUE) {
        throw StorageException("Failed to open file: " + filepath);
    }
    
    // Get file size
    LARGE_INTEGER file_size;
    if (!GetFileSizeEx(file_handle, &file_size)) {
        CloseHandle(file_handle);
        throw StorageException("Failed to get file size: " + filepath);
    }
    size = file_size.QuadPart;
    
    // Create file mapping
    HANDLE mapping_handle = CreateFileMappingA(
        file_handle,
        nullptr,
        PAGE_READONLY,
        0, 0,  // Map entire file
        nullptr
    );
    
    if (mapping_handle == nullptr) {
        CloseHandle(file_handle);
        throw StorageException("CreateFileMapping failed: " + filepath);
    }
    
    // Map view of file
    void* ptr = MapViewOfFile(
        mapping_handle,
        FILE_MAP_READ,
        0, 0,  // Offset
        0      // Map entire file
    );
    
    if (ptr == nullptr) {
        CloseHandle(mapping_handle);
        CloseHandle(file_handle);
        throw StorageException("MapViewOfFile failed: " + filepath);
    }
    
    // Store handles for cleanup
    file_handle_ = file_handle;
    mapping_handle_ = mapping_handle;
    
    return ptr;
}
```

---

### Vector File Format Implementation

#### Writing Vectors

```cpp
void VectorFile::write(
    const std::string& filepath,
    const float* vectors,
    uint64_t count,
    uint32_t dimension
) {
    std::ofstream file(filepath, std::ios::binary);
    if (!file) {
        throw StorageException("Failed to create file: " + filepath);
    }
    
    // Prepare header
    Header header = {};
    header.magic = 0x5644415441;  // "VDATA"
    header.format_version = 1;
    header.vector_count = count;
    header.dimension = dimension;
    header.data_type = 0;  // f32
    header.alignment = 64;
    
    // Write header (256 bytes)
    file.write(reinterpret_cast<const char*>(&header), sizeof(Header));
    
    // Align to 256 bytes
    file.seekp(256);
    
    // Write vectors with alignment
    const size_t vector_size = dimension * sizeof(float);
    const size_t aligned_size = ((vector_size + 63) / 64) * 64;  // Round up to 64
    
    std::vector<uint8_t> padding(aligned_size - vector_size, 0);
    
    for (uint64_t i = 0; i < count; i++) {
        // Write vector data
        file.write(
            reinterpret_cast<const char*>(vectors + i * dimension),
            vector_size
        );
        
        // Write padding for alignment
        if (!padding.empty()) {
            file.write(reinterpret_cast<const char*>(padding.data()), 
                      padding.size());
        }
    }
    
    file.close();
    
    if (!file) {
        throw StorageException("Failed to write vectors: " + filepath);
    }
}
```

#### Reading Vectors

```cpp
std::unique_ptr<VectorFile> VectorFile::open(const std::string& filepath) {
    auto file = std::make_unique<VectorFile>();
    
    // Memory map the file
    file->mmap_ptr_ = file->mmap_file(filepath, file->mmap_size_);
    
    // Read header
    const uint8_t* ptr = static_cast<const uint8_t*>(file->mmap_ptr_);
    std::memcpy(&file->header_, ptr, sizeof(Header));
    
    // Validate magic number
    if (file->header_.magic != 0x5644415441) {
        throw StorageException("Invalid magic number in: " + filepath);
    }
    
    // Validate format version
    if (file->header_.format_version != 1) {
        throw StorageException("Unsupported format version: " + 
                             std::to_string(file->header_.format_version));
    }
    
    // Validate file size
    size_t expected_size = 256 +  // Header
        file->header_.vector_count * 
        ((file->header_.dimension * 4 + 63) / 64) * 64;  // Aligned vectors
    
    if (file->mmap_size_ < expected_size) {
        throw StorageException("File too small, possibly corrupted: " + filepath);
    }
    
    // Set data pointer (skip header)
    file->data_ = reinterpret_cast<const float*>(ptr + 256);
    
    return file;
}
```

---

### Graph File Format Implementation

#### Writing Graph

```cpp
void GraphFile::write(
    const std::string& filepath,
    const std::vector<std::vector<uint32_t>>& adjacency_lists
) {
    std::ofstream file(filepath, std::ios::binary);
    if (!file) {
        throw StorageException("Failed to create file: " + filepath);
    }
    
    uint64_t node_count = adjacency_lists.size();
    
    // Compute statistics
    uint32_t max_degree = 0;
    uint64_t total_degree = 0;
    for (const auto& neighbors : adjacency_lists) {
        max_degree = std::max(max_degree, static_cast<uint32_t>(neighbors.size()));
        total_degree += neighbors.size();
    }
    uint32_t avg_degree = static_cast<uint32_t>(total_degree / node_count);
    
    // Prepare header
    Header header = {};
    header.magic = 0x4752415048;  // "GRAPH"
    header.format_version = 1;
    header.node_count = node_count;
    header.max_degree = max_degree;
    header.avg_degree = avg_degree;
    
    // Write header (256 bytes)
    file.write(reinterpret_cast<const char*>(&header), sizeof(Header));
    file.seekp(256);
    
    // Build offset table and adjacency lists
    std::vector<uint64_t> offsets(node_count);
    std::vector<uint8_t> adjacency_data;
    
    for (uint64_t i = 0; i < node_count; i++) {
        offsets[i] = adjacency_data.size();
        
        const auto& neighbors = adjacency_lists[i];
        uint32_t degree = static_cast<uint32_t>(neighbors.size());
        
        // Write degree
        adjacency_data.insert(
            adjacency_data.end(),
            reinterpret_cast<const uint8_t*>(&degree),
            reinterpret_cast<const uint8_t*>(&degree) + 4
        );
        
        // Write neighbor IDs
        adjacency_data.insert(
            adjacency_data.end(),
            reinterpret_cast<const uint8_t*>(neighbors.data()),
            reinterpret_cast<const uint8_t*>(neighbors.data()) + 
                (neighbors.size() * 4)
        );
        
        // Align to 8 bytes
        size_t padding = (8 - (adjacency_data.size() % 8)) % 8;
        adjacency_data.insert(adjacency_data.end(), padding, 0);
    }
    
    // Write offset table
    file.write(
        reinterpret_cast<const char*>(offsets.data()),
        offsets.size() * sizeof(uint64_t)
    );
    
    // Write adjacency data
    file.write(
        reinterpret_cast<const char*>(adjacency_data.data()),
        adjacency_data.size()
    );
    
    file.close();
}
```

#### Reading Graph

```cpp
std::unique_ptr<GraphFile> GraphFile::open(const std::string& filepath) {
    auto file = std::make_unique<GraphFile>();
    
    // Memory map the file
    file->mmap_ptr_ = file->mmap_file(filepath, file->mmap_size_);
    
    // Read header
    const uint8_t* ptr = static_cast<const uint8_t*>(file->mmap_ptr_);
    std::memcpy(&file->header_, ptr, sizeof(Header));
    
    // Validate magic number
    if (file->header_.magic != 0x4752415048) {
        throw StorageException("Invalid magic number in graph file");
    }
    
    // Validate format version
    if (file->header_.format_version != 1) {
        throw StorageException("Unsupported graph format version");
    }
    
    // Set offset table pointer (after header)
    file->offset_table_ = reinterpret_cast<const uint64_t*>(ptr + 256);
    
    // Set adjacency data pointer (after offset table)
    size_t offset_table_size = file->header_.node_count * sizeof(uint64_t);
    file->data_ = ptr + 256 + offset_table_size;
    
    return file;
}
```

---

### Checksum Implementation

```cpp
std::string ChecksumFile::compute_sha256(const std::string& filepath) {
    // Use OpenSSL for SHA-256
    std::ifstream file(filepath, std::ios::binary);
    if (!file) {
        throw StorageException("Failed to open file for checksum: " + filepath);
    }
    
    EVP_MD_CTX* ctx = EVP_MD_CTX_new();
    EVP_DigestInit_ex(ctx, EVP_sha256(), nullptr);
    
    const size_t buffer_size = 65536;  // 64KB buffer
    std::vector<uint8_t> buffer(buffer_size);
    
    while (file.read(reinterpret_cast<char*>(buffer.data()), buffer_size)) {
        EVP_DigestUpdate(ctx, buffer.data(), file.gcount());
    }
    
    // Handle remaining bytes
    if (file.gcount() > 0) {
        EVP_DigestUpdate(ctx, buffer.data(), file.gcount());
    }
    
    uint8_t hash[EVP_MAX_MD_SIZE];
    uint32_t hash_len;
    EVP_DigestFinal_ex(ctx, hash, &hash_len);
    EVP_MD_CTX_free(ctx);
    
    // Convert to hex string
    std::ostringstream oss;
    for (uint32_t i = 0; i < hash_len; i++) {
        oss << std::hex << std::setw(2) << std::setfill('0') 
            << static_cast<int>(hash[i]);
    }
    
    return oss.str();
}

bool ChecksumFile::verify(
    const std::string& filepath,
    const std::string& expected_checksum
) {
    std::string actual_checksum = compute_sha256(filepath);
    return actual_checksum == expected_checksum;
}
```

---

### Manifest Implementation (JSON)

```cpp
void ManifestFile::write(
    const std::string& filepath,
    const Manifest& manifest
) {
    nlohmann::json j;
    
    j["version"] = manifest.version;
    j["format_version"] = manifest.format_version;
    j["created_at"] = manifest.created_at;
    j["vector_count"] = manifest.vector_count;
    j["dimension"] = manifest.dimension;
    j["metric"] = manifest.metric;
    
    j["build_parameters"] = {
        {"R", manifest.build_parameters.R},
        {"L", manifest.build_parameters.L},
        {"alpha", manifest.build_parameters.alpha}
    };
    
    j["files"] = {
        {"vectors", manifest.files.vectors},
        {"graph", manifest.files.graph},
        {"metadata", manifest.files.metadata}
    };
    
    j["checksums"] = manifest.checksums;
    
    std::ofstream file(filepath);
    file << j.dump(2);  // Pretty print with 2-space indent
}

ManifestFile::Manifest ManifestFile::read(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file) {
        throw StorageException("Failed to open manifest: " + filepath);
    }
    
    nlohmann::json j;
    file >> j;
    
    Manifest manifest;
    manifest.version = j["version"];
    manifest.format_version = j["format_version"];
    manifest.created_at = j["created_at"];
    manifest.vector_count = j["vector_count"];
    manifest.dimension = j["dimension"];
    manifest.metric = j["metric"];
    
    manifest.build_parameters.R = j["build_parameters"]["R"];
    manifest.build_parameters.L = j["build_parameters"]["L"];
    manifest.build_parameters.alpha = j["build_parameters"]["alpha"];
    
    manifest.files.vectors = j["files"]["vectors"];
    manifest.files.graph = j["files"]["graph"];
    manifest.files.metadata = j["files"]["metadata"];
    
    manifest.checksums = j["checksums"].get<std::map<std::string, std::string>>();
    
    return manifest;
}
```

---

### High-Level Index Loading

```cpp
std::unique_ptr<IndexStorage::Index> IndexStorage::load(
    const std::string& index_dir,
    bool verify_checksums
) {
    auto index = std::make_unique<Index>();
    
    // Read manifest
    index->manifest = ManifestFile::read(index_dir + "/manifest.json");
    
    // Verify checksums if requested
    if (verify_checksums) {
        auto checksums = ChecksumFile::read(index_dir + "/checksums.sha256");
        
        for (const auto& [filename, expected_hash] : checksums) {
            std::string filepath = index_dir + "/" + filename;
            if (!ChecksumFile::verify(filepath, expected_hash)) {
                throw StorageException("Checksum verification failed: " + filename);
            }
        }
    }
    
    // Load vector file
    index->vectors = VectorFile::open(
        index_dir + "/" + index->manifest.files.vectors
    );
    
    // Load graph file
    index->graph = GraphFile::open(
        index_dir + "/" + index->manifest.files.graph
    );
    
    // Load metadata file
    index->metadata = MetadataFile::open(
        index_dir + "/" + index->manifest.files.metadata
    );
    
    // Validate consistency
    if (index->vectors->count() != index->graph->node_count()) {
        throw StorageException("Vector count mismatch between files");
    }
    
    if (index->vectors->count() != index->metadata->entry_count()) {
        throw StorageException("Metadata count mismatch");
    }
    
    return index;
}
```

---

## Performance Requirements

### Load Time

| Operation | Target | Notes |
|-----------|--------|-------|
| **Open index** | < 10 seconds | For 10M vectors |
| **mmap setup** | < 100ms | Per file |
| **Manifest parse** | < 10ms | JSON parsing |
| **Checksum verify** | < 2 seconds | Optional, all files |

### Memory Overhead

| Component | Memory Usage | Notes |
|-----------|--------------|-------|
| **File descriptors** | ~3 FDs | One per binary file |
| **Headers** | ~1KB | In-memory headers |
| **Offset table** | 80MB | For 10M nodes (8 bytes each) |
| **mmap address space** | 60-100GB | Virtual only, not physical RAM |

**Physical RAM**: Only accessed pages loaded (~2-8GB depending on cache)

### Disk I/O

| Operation | Disk Reads | Notes |
|-----------|-----------|-------|
| **Index open** | 3-4KB | Headers only |
| **Checksum verify** | Sequential read | Full file scan |
| **Vector access** | 6KB | Single vector (dimension=1536) |
| **Graph node access** | 128-256 bytes | Node + neighbors |

---

## Error Handling

### Storage Exceptions

```cpp
class StorageException : public std::exception {
public:
    enum class ErrorCode {
        FILE_NOT_FOUND,
        FILE_CORRUPT,
        FORMAT_MISMATCH,
        CHECKSUM_FAILED,
        MMAP_FAILED,
        DISK_FULL,
        PERMISSION_DENIED
    };
    
    StorageException(ErrorCode code, const std::string& message)
        : code_(code), message_(message) {}
    
    ErrorCode code() const { return code_; }
    const char* what() const noexcept override { return message_.c_str(); }
    
private:
    ErrorCode code_;
    std::string message_;
};
```

### Error Detection and Recovery

| Error | Detection | Recovery |
|-------|-----------|----------|
| **Missing file** | File open fails | Report error, cannot continue |
| **Corrupt header** | Magic number mismatch | Report error, refuse to load |
| **Checksum fail** | Hash mismatch | Warn or fail (configurable) |
| **Truncated file** | Size validation | Report corruption |
| **mmap failure** | mmap returns error | Fallback to read() or fail |
| **Out of disk** | Write fails | Clean up partial files |

---

## Validation and Integrity Checks

### File Format Validation

```cpp
bool VectorFile::validate() const {
    // Check magic number
    if (header_.magic != 0x5644415441) {
        LOG_ERROR("Invalid magic number");
        return false;
    }
    
    // Check format version
    if (header_.format_version != 1) {
        LOG_ERROR("Unsupported format version: " 
                  << header_.format_version);
        return false;
    }
    
    // Check dimension range
    if (header_.dimension < 64 || header_.dimension > 4096) {
        LOG_ERROR("Dimension out of range: " << header_.dimension);
        return false;
    }
    
    // Check file size
    size_t expected_size = compute_expected_size();
    if (mmap_size_ < expected_size) {
        LOG_ERROR("File too small: expected " << expected_size 
                  << ", got " << mmap_size_);
        return false;
    }
    
    // Check for NaN/Inf in sample vectors (first 10)
    for (uint64_t i = 0; i < std::min(10UL, header_.vector_count); i++) {
        const float* vec = get_vector(i);
        for (uint32_t j = 0; j < header_.dimension; j++) {
            if (!std::isfinite(vec[j])) {
                LOG_ERROR("Non-finite value in vector " << i << " at dim " << j);
                return false;
            }
        }
    }
    
    return true;
}
```

### Cross-File Validation

```cpp
bool IndexStorage::validate_index(const std::string& index_dir) {
    try {
        auto index = load(index_dir, true);  // Verify checksums
        
        // Check vector count consistency
        if (index->vectors->count() != index->graph->node_count()) {
            LOG_ERROR("Vector/graph count mismatch");
            return false;
        }
        
        // Check all graph edges reference valid node IDs
        uint64_t node_count = index->graph->node_count();
        for (uint64_t i = 0; i < node_count; i++) {
            const auto* node = index->graph->get_node(i);
            for (uint32_t j = 0; j < node->degree; j++) {
                uint32_t neighbor = node->neighbors[j];
                if (neighbor >= node_count) {
                    LOG_ERROR("Invalid neighbor ID " << neighbor 
                              << " in node " << i);
                    return false;
                }
            }
        }
        
        // Check metadata ID range
        for (uint64_t i = 0; i < index->metadata->entry_count(); i++) {
            int64_t external_id = index->metadata->get_external_id(i);
            // External IDs should be non-negative (optional check)
            if (external_id < 0) {
                LOG_WARN("Negative external ID: " << external_id);
            }
        }
        
        return true;
        
    } catch (const StorageException& e) {
        LOG_ERROR("Validation failed: " << e.what());
        return false;
    }
}
```

---

## Platform Abstraction

### Platform-Specific Implementations

```cpp
#ifdef _WIN32
    #define PLATFORM_WINDOWS
#elif defined(__APPLE__)
    #define PLATFORM_MACOS
#elif defined(__linux__)
    #define PLATFORM_LINUX
#endif

class PlatformIO {
public:
    virtual ~PlatformIO() = default;
    
    virtual void* mmap_file(const std::string& path, size_t& size) = 0;
    virtual void unmap_file(void* ptr, size_t size) = 0;
    virtual bool advise_random_access(void* ptr, size_t size) = 0;
    virtual bool prefetch_page(void* ptr) = 0;
};

// Factory
std::unique_ptr<PlatformIO> create_platform_io() {
#ifdef PLATFORM_WINDOWS
    return std::make_unique<WindowsIO>();
#else
    return std::make_unique<UnixIO>();
#endif
}
```

---

## Testing Strategy

### Unit Tests

```cpp
TEST(VectorFile, WriteAndRead) {
    // Create test vectors
    const uint32_t count = 1000;
    const uint32_t dim = 128;
    std::vector<float> vectors(count * dim);
    generate_random_vectors(vectors.data(), count, dim);
    
    // Write to file
    VectorFile::write("test_vectors.bin", vectors.data(), count, dim);
    
    // Read back
    auto file = VectorFile::open("test_vectors.bin");
    
    EXPECT_EQ(file->count(), count);
    EXPECT_EQ(file->dimension(), dim);
    
    // Verify data
    for (uint32_t i = 0; i < count; i++) {
        const float* vec = file->get_vector(i);
        for (uint32_t j = 0; j < dim; j++) {
            EXPECT_FLOAT_EQ(vec[j], vectors[i * dim + j]);
        }
    }
}

TEST(GraphFile, WriteAndRead) {
    // Create test graph
    std::vector<std::vector<uint32_t>> graph = {
        {1, 2, 3},
        {0, 2},
        {0, 1, 3},
        {0, 2}
    };
    
    // Write to file
    GraphFile::write("test_graph.bin", graph);
    
    // Read back
    auto file = GraphFile::open("test_graph.bin");
    
    EXPECT_EQ(file->node_count(), 4);
    
    // Verify edges
    for (uint32_t i = 0; i < graph.size(); i++) {
        const auto* node = file->get_node(i);
        EXPECT_EQ(node->degree, graph[i].size());
        
        for (uint32_t j = 0; j < node->degree; j++) {
            EXPECT_EQ(node->neighbors[j], graph[i][j]);
        }
    }
}

TEST(ChecksumFile, ComputeAndVerify) {
    // Create test file
    create_test_file("test.bin", 1024);
    
    // Compute checksum
    std::string checksum = ChecksumFile::compute_sha256("test.bin");
    
    EXPECT_EQ(checksum.length(), 64);  // SHA-256 is 64 hex chars
    
    // Verify
    EXPECT_TRUE(ChecksumFile::verify("test.bin", checksum));
    
    // Modify file
    modify_file("test.bin");
    
    // Verify should fail
    EXPECT_FALSE(ChecksumFile::verify("test.bin", checksum));
}
```

### Integration Tests

```cpp
TEST(IndexStorage, LoadAndValidate) {
    // Build test index
    std::string index_dir = build_test_index();
    
    // Load index
    auto index = IndexStorage::load(index_dir, true);
    
    EXPECT_NE(index, nullptr);
    EXPECT_EQ(index->vectors->count(), 10000);
    
    // Validate
    EXPECT_TRUE(IndexStorage::validate_index(index_dir));
}

TEST(IndexStorage, CorruptionDetection) {
    std::string index_dir = build_test_index();
    
    // Corrupt graph file
    corrupt_file(index_dir + "/graph.bin");
    
    // Load should fail on checksum verification
    EXPECT_THROW(
        IndexStorage::load(index_dir, true),
        StorageException
    );
}
```

### Performance Tests

```cpp
BENCHMARK(VectorFile, LoadLarge) {
    // Load 10M vector index
    auto file = VectorFile::open("large_vectors.bin");
    
    // Measure load time
    // Target: < 10 seconds
}

BENCHMARK(IndexStorage, RandomAccess) {
    auto index = IndexStorage::load("test_index");
    
    // Random vector access
    for (int i = 0; i < 10000; i++) {
        uint32_t id = random_id();
        const float* vec = index->vectors->get_vector(id);
        benchmark::DoNotOptimize(vec);
    }
}
```

---

## Dependencies

### Internal Dependencies
- **ADR-001**: Storage format specification (implemented here)
- **COMP-001**: Index Builder (produces files this component reads)

### External Dependencies

| Library | Purpose | Required? |
|---------|---------|-----------|
| **OpenSSL** | SHA-256 checksums | Yes |
| **nlohmann/json** | Manifest parsing | Yes |
| **C++ Standard Library** | File I/O, mmap | Yes |
| **POSIX / Win32 API** | Memory mapping | Yes (platform) |

---

## Configuration

### Storage Options

```cpp
struct StorageConfig {
    bool verify_checksums = true;     // Verify on load
    bool use_mmap = true;              // Use mmap vs read()
    bool prewarm_cache = false;        // Touch all pages on load
    
    enum AccessPattern {
        RANDOM,      // madvise MADV_RANDOM
        SEQUENTIAL,  // madvise MADV_SEQUENTIAL
        WILLNEED     // madvise MADV_WILLNEED
    } access_pattern = RANDOM;
};
```

---

## Future Enhancements

1. **Compression Support** (v1.1)
   - Compressed vector storage (zstd, lz4)
   - Transparent decompression on access

2. **Encryption** (v2)
   - Encrypt index files at rest
   - Key management integration

3. **Remote Storage** (v2)
   - S3-backed indexes
   - Streaming from remote storage

4. **Async I/O** (v2)
   - io_uring (Linux)
   - Overlapped I/O (Windows)

---

## References

- `mmap` man pages: https://man7.org/linux/man-pages/man2/mmap.2.html
- OpenSSL SHA-256: https://www.openssl.org/docs/man3.0/man3/EVP_DigestInit.html
- Windows Memory Mapping: https://docs.microsoft.com/en-us/windows/win32/memory/file-mapping

---

**Status**: Draft - Ready for review  
**Next Steps**: Team review, implement basic I/O operations  
**Blocks**: COMP-004 (Search Engine needs to load indexes)  
**Blocked By**: None (can implement independently)
