# ADR-005: Error Handling and Recovery Philosophy

## Status
Proposed

## Context

The SSD Vector Engine is a stateful system that manages persistent indexes, accepts writes via a write-ahead log, and serves real-time queries. A robust error handling strategy is critical for:

1. **Reliability**: Prevent data loss and corruption
2. **Debuggability**: Provide clear error messages for troubleshooting
3. **Resilience**: Gracefully handle transient failures
4. **Operations**: Enable automated recovery and monitoring

### Error Categories

We must handle errors across multiple layers:

**Infrastructure Errors:**
- Disk I/O failures (read/write errors, disk full)
- Memory exhaustion (OOM)
- Network failures (if distributed)
- Process crashes

**Application Errors:**
- Invalid user input (malformed vectors, bad queries)
- Resource limits exceeded (too large batch, query timeout)
- Concurrency conflicts (write conflicts, stale reads)
- Index corruption (checksum mismatches)

**Transient vs Permanent:**
- **Transient**: Network blips, temporary disk I/O errors, rate limits
- **Permanent**: Corrupted data, invalid input, unsupported operations

### Requirements

From operational perspective:
- **Zero data loss** for committed writes (fsync'd to WAL)
- **Fast recovery** from crashes (< 30 seconds)
- **Clear error messages** for debugging
- **Automatic retry** for transient errors
- **Fail-fast** for permanent errors

---

## Decision

We adopt a **layered error handling strategy** with different policies at each layer:

### 1. **Error Taxonomy and Classification**

All errors inherit from a base exception hierarchy:

```cpp
namespace ssd_vector {

// Base exception with error code
class Exception : public std::exception {
public:
    enum class Category {
        INVALID_ARGUMENT,      // User input error
        IO_ERROR,              // Disk/network I/O
        CORRUPTION,            // Data integrity failure
        RESOURCE_EXHAUSTED,    // OOM, disk full, timeout
        CONCURRENCY,           // Lock contention, write conflict
        INTERNAL,              // Bug in our code
        UNAVAILABLE            // Service temporarily down
    };
    
    enum class Severity {
        FATAL,        // Process must terminate
        ERROR,        // Operation failed, retry unlikely to help
        WARNING,      // Operation failed, retry may help
        INFO          // Informational, not really an error
    };
    
    Exception(
        Category category,
        Severity severity,
        const std::string& message,
        const std::string& detail = ""
    );
    
    Category category() const { return category_; }
    Severity severity() const { return severity_; }
    const char* what() const noexcept override;
    const std::string& detail() const { return detail_; }
    
    // For structured logging
    std::string to_json() const;
    
private:
    Category category_;
    Severity severity_;
    std::string message_;
    std::string detail_;
};

// Specific exception types
class InvalidArgumentException : public Exception { /*...*/ };
class IOErrorException : public Exception { /*...*/ };
class CorruptionException : public Exception { /*...*/ };
class ResourceExhaustedException : public Exception { /*...*/ };
class ConcurrencyException : public Exception { /*...*/ };
class InternalException : public Exception { /*...*/ };
class UnavailableException : public Exception { /*...*/ };

} // namespace ssd_vector
```

**Classification Rules:**

| Error | Category | Severity | Retryable? | Example |
|-------|----------|----------|------------|---------|
| Bad vector dimension | INVALID_ARGUMENT | ERROR | ❌ No | Dimension 1024 != 1536 |
| Disk read failure | IO_ERROR | WARNING | ✅ Yes (once) | Read error on sector 12345 |
| Checksum mismatch | CORRUPTION | FATAL | ❌ No | Expected ABC123, got DEF456 |
| Out of memory | RESOURCE_EXHAUSTED | ERROR | ❌ No (scale up) | Cannot allocate 8GB |
| Disk full | RESOURCE_EXHAUSTED | ERROR | ❌ No (add space) | 0 bytes available on /data |
| Query timeout | RESOURCE_EXHAUSTED | WARNING | ✅ Yes (with backoff) | Query exceeded 100ms |
| Write conflict | CONCURRENCY | WARNING | ✅ Yes | Vector ID 123 modified concurrently |
| Null pointer | INTERNAL | FATAL | ❌ No | Null pointer at line 456 |
| Service starting | UNAVAILABLE | WARNING | ✅ Yes | Index still loading |

---

### 2. **Layer-Specific Error Handling**

#### **Layer 1: Storage Layer (COMP-002)**

**Philosophy**: Fail fast on corruption, retry once on transient I/O errors.

```cpp
class VectorFile {
    const float* get_vector(uint32_t id) {
        try {
            // Attempt read
            auto data = read_from_disk(id);
            
            // Validate checksum
            if (!verify_checksum(data)) {
                throw CorruptionException(
                    "Checksum mismatch",
                    "vector_id=" + std::to_string(id)
                );
            }
            
            return data;
            
        } catch (const IOErrorException& e) {
            // Retry once for transient errors
            LOG_WARN("I/O error reading vector, retrying: " << e.what());
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            
            return read_from_disk(id);  // Second attempt, let it throw
        }
    }
};
```

**Rules:**
- ✅ **Retry once** for I/O errors (disk blip)
- ❌ **No retry** for corruption (data is bad)
- ❌ **No retry** for OOM (won't fix itself)
- 📝 **Log all errors** with context

---

#### **Layer 2: Index Builder (COMP-001)**

**Philosophy**: Crash on errors during build, don't create invalid indexes.

```cpp
void IndexBuilder::build() {
    try {
        LOG_INFO("Starting index build");
        
        load_vectors();       // May throw IOError
        build_graph();        // May throw ResourceExhausted (OOM)
        serialize_index();    // May throw IOError (disk full)
        
        LOG_INFO("Index build complete");
        
    } catch (const IOErrorException& e) {
        LOG_ERROR("I/O error during build: " << e.what());
        cleanup_partial_index();
        throw;  // Re-throw, build failed
        
    } catch (const ResourceExhaustedException& e) {
        LOG_ERROR("Out of resources during build: " << e.what());
        cleanup_partial_index();
        throw;
        
    } catch (const Exception& e) {
        LOG_ERROR("Unexpected error during build: " << e.what());
        cleanup_partial_index();
        throw InternalException("Build failed", e.what());
    }
}
```

**Rules:**
- ❌ **Never produce partial index** (all-or-nothing)
- 🧹 **Clean up on failure** (delete partial files)
- 📝 **Log detailed context** (which phase failed)
- 🚨 **Crash build process** (restart from scratch)

---

#### **Layer 3: Search Engine (COMP-004)**

**Philosophy**: Degrade gracefully, continue serving other queries.

```cpp
SearchResponse SearchEngine::search(const SearchQuery& query) {
    try {
        // Validate input
        validate_query(query);  // Throws InvalidArgument
        
        // Execute search
        return search_internal(query);
        
    } catch (const InvalidArgumentException& e) {
        // User error, return error response
        LOG_WARN("Invalid query: " << e.what());
        return SearchResponse::error(e.message());
        
    } catch (const IOErrorException& e) {
        // Disk error, try to continue
        LOG_ERROR("I/O error during search: " << e.what());
        metrics_.io_error_count++;
        
        // Return partial results if available
        if (partial_results_available()) {
            return SearchResponse::partial(get_partial_results());
        } else {
            return SearchResponse::error("I/O error, please retry");
        }
        
    } catch (const CorruptionException& e) {
        // Corruption, mark index as bad
        LOG_FATAL("Index corruption detected: " << e.what());
        mark_index_unhealthy();
        return SearchResponse::error("Index corruption, rebuilding");
        
    } catch (const std::exception& e) {
        // Unknown error, log and continue
        LOG_ERROR("Unexpected error in search: " << e.what());
        metrics_.unknown_error_count++;
        return SearchResponse::error("Internal error");
    }
}
```

**Rules:**
- ✅ **Return errors, don't crash** (isolate query failures)
- ✅ **Partial results OK** for transient errors
- ⚠️ **Mark index unhealthy** on corruption
- 📈 **Track error metrics** for monitoring
- 📝 **Include request ID** in all logs

---

#### **Layer 4: Write Path (COMP-013 WAL)**

**Philosophy**: Zero data loss, fsync aggressively, validate before commit.

```cpp
void WAL::append(const WriteEntry& entry) {
    try {
        // Validate entry before writing
        validate_entry(entry);  // Throws InvalidArgument
        
        // Serialize
        auto data = serialize(entry);
        
        // Write to buffer
        buffer_.append(data);
        
        // Flush if buffer full or timeout
        if (buffer_.size() >= flush_threshold_bytes_ ||
            time_since_last_flush() > flush_interval_) {
            flush();
        }
        
    } catch (const InvalidArgumentException& e) {
        LOG_ERROR("Invalid write entry: " << e.what());
        throw;  // Caller must handle
        
    } catch (const IOErrorException& e) {
        // Disk write failed - CRITICAL
        LOG_FATAL("WAL write failed: " << e.what());
        
        // Attempt emergency flush to alternate location
        emergency_flush();
        
        throw;  // Write failed, caller must abort transaction
    }
}

void WAL::flush() {
    // Write buffer to disk
    ssize_t written = write(fd_, buffer_.data(), buffer_.size());
    if (written < 0) {
        throw IOErrorException("WAL write failed: " + std::string(strerror(errno)));
    }
    
    // CRITICAL: Must fsync for durability
    if (fsync(fd_) < 0) {
        throw IOErrorException("WAL fsync failed: " + std::string(strerror(errno)));
    }
    
    buffer_.clear();
}
```

**Rules:**
- 🔒 **Fsync before success** (durability guarantee)
- ❌ **No retry on write failure** (data may be corrupted)
- 🚨 **Emergency backup** on critical failure
- 📝 **Log every write** (for recovery)
- ✅ **Validate before write** (detect bad data early)

---

#### **Layer 5: API Layer (COMP-017)**

**Philosophy**: Return structured errors, hide internal details.

```cpp
// API Error Response Format
struct ErrorResponse {
    std::string error_code;       // "INVALID_DIMENSION"
    std::string message;          // "Vector dimension mismatch"
    std::string detail;           // "Expected 1536, got 768"
    std::string request_id;       // "req_abc123"
    bool retryable;               // true/false
};

// Example handler
Response APIGateway::handle_insert(const InsertRequest& req) {
    try {
        validate_request(req);
        
        auto result = engine_->insert(req.vector, req.metadata);
        
        return Response::success(result);
        
    } catch (const InvalidArgumentException& e) {
        return Response::error(
            "INVALID_ARGUMENT",
            e.message(),
            e.detail(),
            req.request_id,
            false  // Not retryable
        );
        
    } catch (const ResourceExhaustedException& e) {
        return Response::error(
            "RESOURCE_EXHAUSTED",
            "Request rate limit exceeded",
            "Try again in 1 second",
            req.request_id,
            true  // Retryable
        );
        
    } catch (const Exception& e) {
        // Log internal error, return generic message
        LOG_ERROR("Internal error: " << e.what() << ", request=" << req.request_id);
        
        return Response::error(
            "INTERNAL_ERROR",
            "An internal error occurred",
            "",  // Don't expose internal details
            req.request_id,
            false
        );
    }
}
```

**Error Codes:**

| Code | HTTP Status | Retryable | Description |
|------|-------------|-----------|-------------|
| `INVALID_ARGUMENT` | 400 | ❌ No | Bad input (dimension, format) |
| `NOT_FOUND` | 404 | ❌ No | Collection/vector not found |
| `RESOURCE_EXHAUSTED` | 429 | ✅ Yes | Rate limit, timeout |
| `INTERNAL_ERROR` | 500 | ❌ No | Bug or unexpected error |
| `UNAVAILABLE` | 503 | ✅ Yes | Service temporarily down |
| `CORRUPTION` | 500 | ❌ No | Data corruption detected |

**Rules:**
- 📋 **Structured errors** (machine-readable)
- 🔒 **Hide internal details** (security)
- 🆔 **Include request ID** (tracing)
- ✅ **Indicate retryable** (client guidance)
- 📊 **Map to HTTP status** (REST convention)

---

### 3. **Crash Recovery Strategy**

**Principle**: Use WAL for durability, recover to last committed state.

#### **Recovery Phases**

```cpp
class RecoveryManager {
public:
    void recover() {
        LOG_INFO("Starting crash recovery");
        
        // Phase 1: Validate index integrity
        if (!validate_index_checksums()) {
            LOG_FATAL("Index corruption detected, cannot recover");
            throw CorruptionException("Index checksums invalid");
        }
        
        // Phase 2: Replay WAL
        auto last_committed = get_last_committed_offset();
        auto wal_entries = read_wal_from(last_committed);
        
        LOG_INFO("Replaying " << wal_entries.size() << " WAL entries");
        
        for (const auto& entry : wal_entries) {
            try {
                replay_entry(entry);
            } catch (const Exception& e) {
                LOG_ERROR("Failed to replay entry: " << e.what());
                // Continue replaying, log error
            }
        }
        
        // Phase 3: Rebuild delta index from WAL
        rebuild_delta_index(wal_entries);
        
        // Phase 4: Mark as healthy
        mark_healthy();
        
        LOG_INFO("Recovery complete, service ready");
    }
};
```

**Recovery Guarantees:**
- ✅ **All fsync'd writes are recovered** (durability)
- ✅ **Recovery is idempotent** (can replay multiple times)
- ✅ **Partial writes are discarded** (atomicity)
- ⏱️ **Recovery completes in < 30 seconds** for 1M writes

**Failure Scenarios:**

| Scenario | Detection | Recovery | Data Loss |
|----------|-----------|----------|-----------|
| Process crash | Restart | Replay WAL | 0 (all fsync'd) |
| Disk read error | Checksum fail | Retry read, mark bad | 0 if readable |
| Disk corruption | Checksum fail | Restore from backup | Depends on backup age |
| OOM kill | Restart | Replay WAL | 0 (all fsync'd) |
| Power loss | Restart | Replay WAL | 0 (all fsync'd) |

---

### 4. **Logging Strategy**

**Structured Logging Format:**

```json
{
  "timestamp": "2025-12-20T10:15:30.123Z",
  "level": "ERROR",
  "component": "SearchEngine",
  "message": "I/O error during search",
  "error_code": "IO_ERROR",
  "error_category": "IO_ERROR",
  "severity": "WARNING",
  "request_id": "req_abc123",
  "vector_id": 456789,
  "latency_ms": 15.3,
  "retry_count": 1,
  "stack_trace": "..."
}
```

**Log Levels:**

| Level | Purpose | Persistence | Example |
|-------|---------|-------------|---------|
| **FATAL** | Process must exit | Permanent | Index corruption, OOM |
| **ERROR** | Operation failed | Permanent | I/O error, invalid input |
| **WARN** | Unexpected but handled | Permanent | Retry triggered, slow query |
| **INFO** | Significant event | Permanent | Index built, service started |
| **DEBUG** | Development detail | Debug builds only | Cache hit rate, timing |
| **TRACE** | Function-level trace | Debug builds only | Entering function X |

**Logging Rules:**
- 📝 **Log all exceptions** with full context
- 🆔 **Include request ID** in all query logs
- ⏱️ **Log timing** for slow operations (> 100ms)
- 🔒 **Sanitize PII** in logs (no user data)
- 📊 **Rate limit** high-frequency logs

---

### 5. **Monitoring and Alerting**

**Key Metrics to Track:**

```cpp
struct ErrorMetrics {
    // By category
    Counter invalid_argument_count;
    Counter io_error_count;
    Counter corruption_count;
    Counter resource_exhausted_count;
    Counter internal_error_count;
    
    // By retryability
    Counter retryable_error_count;
    Counter fatal_error_count;
    
    // Recovery
    Counter crash_recovery_count;
    Histogram recovery_time_seconds;
    
    // Health
    Gauge index_health;  // 0 = unhealthy, 1 = healthy
    Gauge wal_lag_bytes;
};
```

**Alert Rules:**

| Metric | Threshold | Severity | Action |
|--------|-----------|----------|--------|
| `corruption_count` | > 0 | 🚨 CRITICAL | Page on-call, restore from backup |
| `io_error_count` | > 100/min | ⚠️ WARNING | Check disk health |
| `fatal_error_count` | > 10/min | 🚨 CRITICAL | Service degraded |
| `recovery_time` | > 60s | ⚠️ WARNING | Investigate slow recovery |
| `index_health` | < 1 | 🚨 CRITICAL | Service unavailable |

---

## Consequences

### Positive

1. **Predictable Behavior**
   - Clear error taxonomy
   - Documented retry policies
   - Known recovery guarantees

2. **Debuggability**
   - Structured logs with context
   - Request ID tracing
   - Detailed error messages

3. **Resilience**
   - Graceful degradation
   - Automatic retry for transient errors
   - Fast crash recovery

4. **Operations**
   - Clear monitoring metrics
   - Actionable alerts
   - Self-healing where possible

### Negative

1. **Code Complexity**
   - Exception handling adds LOC
   - Multiple retry policies to maintain
   - Testing all error paths is challenging

2. **Performance Overhead**
   - Logging on every error
   - Checksum validation
   - fsync latency on write path

3. **Partial Failure Modes**
   - Partial results may confuse users
   - Error propagation across layers is complex
   - Testing corner cases is hard

---

## Implementation Checklist

### Milestone 1: Core Error Handling
- ✅ Define exception hierarchy
- ✅ Implement structured logging
- ✅ Add checksums to storage layer
- ✅ Basic crash recovery (WAL replay)
- ✅ Metrics for error tracking

### Milestone 2: Resilience
- ⬜ Automatic retry for transient errors
- ⬜ Graceful degradation (partial results)
- ⬜ Health checks and circuit breakers
- ⬜ Advanced recovery (index rebuild)

### Milestone 3: Production Hardening
- ⬜ API error code standardization
- ⬜ Rate limiting and backpressure
- ⬜ Distributed tracing (request IDs across services)
- ⬜ Chaos engineering tests

---

## Testing Strategy

### Unit Tests

```cpp
TEST(ErrorHandling, IOErrorRetry) {
    MockVectorFile file;
    file.fail_next_read(IOError);  // First read fails
    
    auto data = file.get_vector(123);  // Should retry
    
    EXPECT_EQ(file.read_count(), 2);  // 1 failed + 1 retry
    EXPECT_NE(data, nullptr);
}

TEST(ErrorHandling, CorruptionNoRetry) {
    MockVectorFile file;
    file.corrupt_vector(123);
    
    EXPECT_THROW(file.get_vector(123), CorruptionException);
    EXPECT_EQ(file.read_count(), 1);  // No retry
}
```

### Integration Tests

```cpp
TEST(Recovery, WALReplay) {
    // Write 1000 entries
    for (int i = 0; i < 1000; i++) {
        wal.append(create_write_entry(i));
    }
    wal.flush();
    
    // Simulate crash
    kill_process();
    
    // Restart and recover
    RecoveryManager recovery;
    recovery.recover();
    
    // Verify all writes recovered
    EXPECT_EQ(delta_index.size(), 1000);
}
```

### Chaos Tests

```bash
# Random disk failures during queries
chaos-test --inject-disk-errors --rate 0.01 --duration 60s

# OOM during index build
chaos-test --memory-limit 8GB --build-index

# Network partition during distributed query
chaos-test --partition-network --duration 10s
```

---

## Related Decisions

- **ADR-001**: Storage format (checksums for corruption detection)
- **ADR-012**: WAL format (enables crash recovery)
- **COMP-013**: WAL implementation (durability guarantees)
- **FEAT-007**: Error response format (API layer errors)

---

## References

1. **Google SRE Book**: Error Budgets and Monitoring
   - https://sre.google/sre-book/table-of-contents/

2. **Crash Recovery Patterns**: Write-Ahead Logging
   - ARIES recovery algorithm

3. **Error Handling Best Practices**:
   - "Exceptional C++" by Herb Sutter
   - "Effective Error Handling in C++" by Herb Sutter

4. **Chaos Engineering**:
   - Principles of Chaos Engineering (Netflix)
   - https://principlesofchaos.org/

---

**Decision Date**: 2025-12-20  
**Status**: Proposed  
**Deciders**: Engineering Team  
**Revisit Date**: After Milestone 1 implementation
