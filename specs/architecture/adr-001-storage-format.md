# Architecture Decision: Storage Format and Serialization Strategy

## Status
- [x] Proposed
- [ ] Accepted
- [ ] Rejected
- [ ] Deprecated

**Date**: 2025-12-20  
**Deciders**: Engineering Team  
**Related Specs**: COMP-001, COMP-002, COMP-003

---

## Context

The SSD Vector Engine needs a persistent storage format for:
1. **Vector data** - High-dimensional embeddings (typically 128-2048 dimensions)
2. **Graph structures** - DiskANN's proximity graph for navigation
3. **Index metadata** - Version, parameters, statistics
4. **ID mappings** - External IDs to internal offsets

### Key Requirements

- **SSD-optimized**: Minimize random seeks, prefer sequential reads
- **Memory-efficient**: Don't require full index in RAM
- **Version-tolerant**: Support format evolution
- **Corruption-detectable**: Checksums for data integrity
- **Build-deterministic**: Same input → same output
- **Platform-portable**: Works across x86-64, ARM64

### Scale Assumptions

- **v1 Target**: Up to 10M vectors per collection
- **Vector dimensions**: 128-2048 (most common: 384, 768, 1536)
- **Disk overhead**: 2-3x raw vector data acceptable
- **Load time**: Index load should be < 10 seconds for 10M vectors

---

## Decision

We will use a **multi-file, memory-mappable binary format** with the following structure:

### File Structure

```
index_dir/
├── manifest.json           # Human-readable index metadata
├── vectors.bin             # Raw vector data (memory-mappable)
├── graph.bin              # Proximity graph structure
├── metadata.bin           # ID mappings and auxiliary data
└── checksums.sha256       # Integrity verification
```

### Format Specifications

#### 1. Manifest File (`manifest.json`)

**Purpose**: Human-readable metadata and version information

**Format**: JSON

```json
{
  "version": "1.0.0",
  "format_version": 1,
  "created_at": "2025-12-20T10:30:00Z",
  "vector_count": 1000000,
  "dimension": 1536,
  "metric": "cosine",
  "build_parameters": {
    "R": 32,
    "L": 100,
    "alpha": 1.2
  },
  "files": {
    "vectors": "vectors.bin",
    "graph": "graph.bin",
    "metadata": "metadata.bin"
  },
  "checksums": {
    "vectors": "sha256:abc123...",
    "graph": "sha256:def456...",
    "metadata": "sha256:ghi789..."
  }
}
```

**Rationale**: JSON for human inspection, debugging, and tooling

---

#### 2. Vector Data File (`vectors.bin`)

**Purpose**: Store raw vector embeddings efficiently

**Format**: Binary, memory-mappable

**Structure**:
```
┌─────────────────────────────────────────┐
│ Header (256 bytes)                      │
├─────────────────────────────────────────┤
│ Magic Number (8 bytes): 0x5644415441    │  "VDATA"
│ Format Version (4 bytes): 0x00000001    │
│ Vector Count (8 bytes)                  │
│ Dimension (4 bytes)                     │
│ Data Type (4 bytes): 0=f32, 1=f16       │
│ Alignment (4 bytes): byte alignment     │
│ Reserved (224 bytes): for future use    │
├─────────────────────────────────────────┤
│ Vector 0 (dimension × sizeof(type))     │
│ Vector 1 (dimension × sizeof(type))     │
│ ...                                     │
│ Vector N-1 (dimension × sizeof(type))   │
└─────────────────────────────────────────┘
```

**Data Types**:
- `f32`: IEEE 754 single precision (4 bytes per component) - **v1 default**
- `f16`: IEEE 754 half precision (2 bytes per component) - future optimization

**Alignment**: Vectors aligned to 64-byte boundaries for SIMD and cache efficiency

**Memory Mapping**: Designed for `mmap()` on Unix, `MapViewOfFile()` on Windows

**Rationale**:
- Direct memory access without deserialization
- SIMD-friendly alignment
- Minimal CPU overhead on load
- Supports future compression (f16, quantization)

---

#### 3. Graph Structure File (`graph.bin`)

**Purpose**: Store DiskANN proximity graph for navigation

**Format**: Binary, memory-mappable

**Structure**:
```
┌─────────────────────────────────────────┐
│ Header (256 bytes)                      │
├─────────────────────────────────────────┤
│ Magic Number (8 bytes): 0x4752415048    │  "GRAPH"
│ Format Version (4 bytes): 0x00000001    │
│ Node Count (8 bytes)                    │
│ Max Degree (4 bytes): max edges per node│
│ Avg Degree (4 bytes)                    │
│ Reserved (228 bytes)                    │
├─────────────────────────────────────────┤
│ Offset Table (node_count × 8 bytes)    │  Offsets to adjacency lists
├─────────────────────────────────────────┤
│ Node 0 Adjacency List                   │
│   Degree (4 bytes)                      │
│   Neighbor IDs (degree × 4 bytes)       │
│   [Padding to 8-byte alignment]         │
├─────────────────────────────────────────┤
│ Node 1 Adjacency List                   │
│ ...                                     │
└─────────────────────────────────────────┘
```

**Node IDs**: 32-bit unsigned integers (supports up to 4B vectors)

**Rationale**:
- Offset table enables O(1) navigation start
- Sequential layout for adjacency lists (cache-friendly)
- 32-bit IDs sufficient for v1 scale
- Padding ensures aligned access

---

#### 4. Metadata File (`metadata.bin`)

**Purpose**: Store ID mappings and auxiliary index metadata

**Format**: Binary

**Structure**:
```
┌─────────────────────────────────────────┐
│ Header (256 bytes)                      │
├─────────────────────────────────────────┤
│ Magic Number (8 bytes): 0x4d45544144    │  "METAD"
│ Format Version (4 bytes): 0x00000001    │
│ Entry Count (8 bytes)                   │
│ ID Type (4 bytes): 0=int64, 1=string    │
│ Reserved (232 bytes)                    │
├─────────────────────────────────────────┤
│ ID Mapping Section                      │
│   Internal ID → External ID mapping     │
│   (format depends on ID Type)           │
├─────────────────────────────────────────┤
│ Statistics Section (optional)           │
│   Build time, recall estimates, etc.    │
└─────────────────────────────────────────┘
```

**ID Mapping for int64 type**:
- Simple array: `external_id[i] = internal_id_i`
- Size: `entry_count × 8 bytes`

**ID Mapping for string type** (future):
- Offset table + string pool
- More complex, deferred to v2

**Rationale**:
- Separates ID concerns from vector/graph data
- Extensible for future metadata types
- Simple for v1 (int64 IDs only)

---

#### 5. Checksum File (`checksums.sha256`)

**Purpose**: Verify data integrity after transfer or on load

**Format**: Text file with SHA-256 hashes

```
abc123...  vectors.bin
def456...  graph.bin
ghi789...  metadata.bin
```

**Verification Policy**:
- **On build**: Always compute and write checksums
- **On load**: Optional verification (configurable)
- **On corruption**: Fail loudly, log error, refuse to load

**Rationale**:
- Standard format, easy to verify with `sha256sum`
- Catches disk corruption, transfer errors
- Optional verification for performance-critical loads

---

## Versioning Strategy

### Format Version Numbers

**Format**: Semantic versioning within each file type

- `1.0.0`: Initial v1 format
- `1.1.0`: Backward-compatible additions (e.g., new metadata fields)
- `2.0.0`: Breaking changes (e.g., new graph structure)

### Compatibility Rules

1. **Forward Compatibility**: Not guaranteed (old code can't read new formats)
2. **Backward Compatibility**: Within major version (new code reads old formats)
3. **Migration**: Provide tools to upgrade old indexes to new formats

### Version Detection

Load-time checks:
```cpp
// Pseudo-code
if (file_version.major != CURRENT_VERSION.major) {
    throw VersionMismatchError("Incompatible format");
}
if (file_version.minor > CURRENT_VERSION.minor) {
    LOG_WARN("Newer format, some features may be ignored");
}
```

---

## Platform Portability

### Endianness

**Decision**: Little-endian only for v1

**Rationale**:
- All target platforms (x86-64, ARM64 server) are little-endian
- Simplifies implementation
- Big-endian conversion can be added if needed (unlikely)

### Alignment

- All multi-byte values aligned to their natural boundaries
- Vectors aligned to 64-byte boundaries (SIMD + cache line)
- Headers aligned to 256 bytes (page-aligned for mmap)

### File Paths

- Use forward slashes internally, convert on Windows if needed
- UTF-8 encoding for all paths and string data

---

## Build Determinism

**Goal**: Same input vectors → identical output index files

**Requirements**:
1. **Stable sorting**: Use deterministic tie-breaking
2. **Fixed seeds**: Deterministic random number generation
3. **No timestamps**: Exclude build timestamps from binary data (only in manifest)
4. **Ordered iteration**: Process vectors in consistent order

**Validation**:
- CI tests build same dataset twice, compare byte-for-byte
- Checksums must match exactly

---

## Index Loading Strategy

### Memory Mapping

**Primary approach**: Use `mmap()` for large files

```cpp
// Pseudo-code
void* vectors_mmap = mmap(
    nullptr, 
    file_size, 
    PROT_READ,
    MAP_SHARED,  // Allow OS page cache
    fd, 
    0
);
```

**Benefits**:
- Zero-copy access
- OS manages page cache automatically
- Lazy loading (only accessed pages loaded)
- Shared across processes

**Tradeoffs**:
- Address space consumption (not RAM)
- Page fault latency on first access
- Requires 64-bit address space

### Warm-up Strategy

**Optional pre-warming**:
```cpp
// Touch every page to pre-fault
for (size_t i = 0; i < file_size; i += PAGE_SIZE) {
    volatile char c = data[i];
}
```

**Policy**:
- Configurable per deployment
- Default: lazy (no warm-up)
- Production: consider pre-warming for predictable latency

---

## Disk Space Overhead

### Expected Multipliers

For a collection with `N` vectors of dimension `D`:

- **Raw vector data**: `N × D × 4 bytes` (f32)
- **Graph structure**: `N × avg_degree × 4 bytes` ≈ `N × 32 × 4` = `128N bytes`
- **Metadata**: `N × 8 bytes` (int64 IDs)
- **Overhead**: Headers, alignment, checksums ≈ 1%

**Total**: ≈ `(4D + 136) × N bytes`

**Examples**:
- **1M vectors, D=768**: 3.2GB raw + 0.13GB graph + 0.008GB metadata ≈ **3.34GB**
- **10M vectors, D=1536**: 61.4GB raw + 1.3GB graph + 0.08GB metadata ≈ **62.8GB**

**Multiplier**: 2-2.5x raw vector data (acceptable for v1)

---

## Corruption Detection

### Checksum Verification

**On Build**:
```cpp
compute_file_checksum("vectors.bin") → checksums.sha256
compute_file_checksum("graph.bin") → checksums.sha256
compute_file_checksum("metadata.bin") → checksums.sha256
```

**On Load** (optional):
```cpp
if (config.verify_checksums) {
    verify_checksum("vectors.bin") or fail();
    verify_checksum("graph.bin") or fail();
    verify_checksum("metadata.bin") or fail();
}
```

### Header Magic Numbers

All binary files start with unique magic numbers:
- Detects wrong file type
- Catches partial writes
- Enables file type detection

### Format Validation

On load, validate:
- Header fields are within reasonable ranges
- File size matches expected size from header
- All offsets are within file bounds

---

## Migration and Evolution

### Adding New Fields (Minor Version Bump)

Example: Add compression metadata to vectors.bin header

1. Use reserved bytes in header (224 bytes available)
2. Increment minor version: `1.0 → 1.1`
3. Old readers ignore new fields (backward compatible)
4. New readers detect version and use new fields

### Changing Structure (Major Version Bump)

Example: Switch to compressed graph representation

1. Create new file format: `graph_v2.bin`
2. Increment major version: `1.x → 2.0`
3. Provide migration tool: `upgrade_index_v1_to_v2`
4. Support loading v1 indexes (with deprecation warning)

---

## Consequences

### Positive

✅ **Fast Loading**: Memory-mapped files load in milliseconds  
✅ **Low Memory**: OS manages paging, not application  
✅ **Debuggable**: Standard binary format with clear structure  
✅ **Portable**: Works across platforms (little-endian)  
✅ **Verifiable**: Checksums catch corruption  
✅ **Evolvable**: Reserved space and versioning support changes  
✅ **SIMD-Friendly**: Aligned data for vectorized operations  
✅ **Deterministic**: Reproducible builds enable testing  

### Negative

❌ **No Compression**: v1 format is uncompressed (future work)  
❌ **Little-Endian Only**: Not portable to big-endian systems (rare)  
❌ **No Encryption**: Data stored in plaintext (application-level encryption required)  
❌ **Large Address Space**: Requires 64-bit process for large indexes  
❌ **Page Fault Latency**: First access to pages has overhead  

### Mitigation Strategies

- **Compression**: Add in v1.1 or v2.0 (quantization, encoding)
- **Encryption**: Use filesystem-level encryption (dm-crypt, BitLocker)
- **Page Faults**: Pre-warm critical pages if latency-sensitive
- **Monitoring**: Track mmap failures, corruption events

---

## Alternatives Considered

### Alternative 1: Single-File Format

**Approach**: Store everything in one large file

**Rejected because**:
- Harder to partially load/reload components
- More complex offset management
- Less debuggable (can't inspect individual components)
- Atomic updates require full file rewrite

### Alternative 2: Database (SQLite, RocksDB)

**Approach**: Use embedded database for storage

**Rejected because**:
- Overhead of DB abstractions
- Not optimized for vector/graph access patterns
- Harder to memory-map efficiently
- Adds complexity and dependencies

### Alternative 3: Columnar Format (Apache Arrow, Parquet)

**Approach**: Use existing columnar storage format

**Rejected because**:
- Designed for tabular data, not graphs
- Over-engineered for v1 needs
- Adds large dependency
- Not optimized for mmap access patterns

### Alternative 4: HDF5

**Approach**: Use HDF5 for multi-dimensional arrays

**Rejected because**:
- Large, complex dependency
- Not designed for graph structures
- Overkill for simple vector storage
- Harder to debug and maintain

---

## Open Questions

1. **Compression Strategy** (deferred to ADR-026)
   - When to add compression? (v1.1 or v2.0?)
   - Which methods? (quantization, encoding, zstd?)

2. **Update Strategy** (deferred to ADR-008)
   - How to update vectors without full rebuild?
   - Delta format compatibility with main format?

3. **Distribution Format** (deferred)
   - Should indexes be distributed as tar.gz?
   - Or individual files for incremental sync?

---

## Implementation Checklist

- [ ] Implement binary format writers for all file types
- [ ] Implement binary format readers with mmap support
- [ ] Add checksum computation and verification
- [ ] Create format validation functions
- [ ] Write deterministic build tests (build twice, compare)
- [ ] Add corruption injection tests
- [ ] Document format in detail (binary specification)
- [ ] Create debugging tools (index inspector CLI)

---

## References

- DiskANN paper: https://arxiv.org/abs/1907.10310
- Memory mapping: `man mmap`, Windows `MapViewOfFile` docs
- IEEE 754 floating point: https://en.wikipedia.org/wiki/IEEE_754

---

**Status**: Proposed - Needs team review and approval  
**Next Steps**: Present to team, gather feedback, revise, approve  
**Implementation**: Blocks COMP-001, COMP-002, COMP-003
