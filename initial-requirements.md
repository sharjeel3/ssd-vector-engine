# SSD-First Vector Search Engine

An **SSD-based vector search system** built on DiskANN-style indexing, designed to offer **simple Python SDKs** while running on **raw VM + NVMe infrastructure** using a **C++ core**.

The system prioritizes **cost efficiency**, **predictable performance**, and **explicit operational boundaries**, while keeping the **developer experience intentionally simple**.

---

## Technology Choices (Explicit)

* **Core ANN engine & indexing**: C++
* **Index builder & merger**: C++
* **Query execution**: C++
* **SDK**: Python
* **Deployment model**: VM-based
* **Storage**: NVMe SSD (disk-first)
* **Memory usage**: cache + routing only

---

## What This Is

* A production-oriented **vector retrieval engine**
* Optimized for **large-scale datasets**
* SSD-first (DiskANN-style), not memory-heavy
* Exposes a **Python SDK** similar to managed vector databases

---

## What This Is Not

* ❌ Not a transactional database
* ❌ Not strongly consistent
* ❌ Not in-memory ANN
* ❌ Not a full search engine

---

## Developer Experience (DX)

```python
client = VectorClient(api_key="...")

client.create_collection(
    name="documents",
    dimension=1536,
    metric="cosine"
)

client.upsert(
    collection="documents",
    vectors=[
        {"id": "doc1", "vector": embedding, "metadata": {"type": "pdf"}}
    ]
)

results = client.query(
    collection="documents",
    vector=query_embedding,
    top_k=10,
    filter={"type": "pdf"}
)
```

### DX Assumptions

* Indexing is asynchronous
* Results may improve over time
* SDK users do not manage indexes
* SDK users do not tune ANN internals

---

## High-Level Architecture

```
Python SDK
 │
API Layer
 │
Query Service (C++)
 │
ANN Engine (C++)
 │
SSD Index Files
```

---

## Core Components (With Assumptions)

### 1. Python SDK

**Purpose**

* Simple API
* Batching, retries, error handling

**Assumptions**

* Network-bound
* Async-safe
* No synchronous consistency guarantees

---

### 2. API Layer

**Purpose**

* Auth, validation, routing

**Assumptions**

* Stateless
* Horizontally scalable
* No data storage

---

### 3. Ingestion Pipeline

**Purpose**

* Accept upserts/deletes
* Persist writes durably (WAL / append log)
* Enqueue vectors for indexing

**Assumptions**

* Write ACK ≠ index convergence
* Backpressure is acceptable
* Ordering between writes is not guaranteed

---

### 4. Delta Index (Mutable Layer)

**Purpose**

* Near-real-time visibility for recent writes

**Assumptions**

* Small and bounded
* Periodically merged
* Queried alongside main index

---

### 5. Main ANN Index (DiskANN-Style, C++)

**Purpose**

* Store and search most vectors
* Optimize SSD IO

**Assumptions**

* Immutable once published
* Rebuilt asynchronously
* RAM is cache, not ground truth

---

### 6. Query Service (C++)

**Purpose**

* Execute ANN search
* Apply filters
* Merge results

**Assumptions**

* Stateless
* Read-only access to indexes
* Latency budgets enforced

---

### 7. Metadata Store

**Purpose**

* IDs, metadata, filters, tombstones

**Assumptions**

* Small relative to vectors
* Slight staleness acceptable

---

### 8. Index Builder & Merger (C++)

**Purpose**

* Build new indexes
* Merge delta layers
* Produce immutable snapshots

**Assumptions**

* Resource intensive
* Runs outside query path

---

## Write Path (Step-by-Step)

1. SDK sends `upsert`
2. API validates and authenticates
3. Write is appended to WAL / log
4. Metadata store updated
5. Vector added to delta index queue
6. ACK returned to client
7. Background index builder eventually merges data

**Guarantee**: durability, not immediate recall.

---

## Read Path (Step-by-Step)

1. SDK sends `query`
2. API validates request
3. Query service:

   * Searches delta index
   * Searches main ANN index
4. Metadata filters applied
5. Results merged and ranked
6. Response returned

---

## Index Lifecycle States

* **Building**: index is being constructed
* **Published**: index is queryable
* **Retired**: index kept briefly for rollback, then deleted

Indexes are swapped atomically.

---

## Concurrency Model (High-Level)

* Query path: **read-only**, lock-free where possible
* Index build path: isolated processes or threads
* Delta index: bounded, controlled mutation
* No shared mutable state between query and build paths

---

## Failure Model & Recovery Assumptions

* If a node crashes during build → rebuild from WAL
* If query service crashes → restart (stateless)
* If index snapshot corrupt → fallback to last published index
* Metadata and index may temporarily diverge → eventual convergence via rebuild

---

## Configuration Boundaries

**Configurable**

* Index build frequency
* Delta size limits
* Query latency budget
* SSD paths

**Not Configurable (v1)**

* ANN algorithm choice
* Consistency model
* Memory vs disk strategy

---

## Minimal v1 Implementation Scope

**Included**

* Single-node
* One collection
* DiskANN-style index
* Delta layer
* Python SDK
* Async rebuild
* Basic metrics

**Excluded**

* Replication
* Auto-scaling
* Hybrid search
* Re-ranking
* Multi-region

---

## Suggested Repository Layout

```
/core
  /ann        # DiskANN wrappers (C++)
  /index      # Build, merge, publish logic (C++)
  /query      # Query execution (C++)
  /storage    # SSD IO, WAL, snapshots

/sdk
  /python     # Python SDK

/api
  /gateway    # HTTP/gRPC API layer

/docs
```

---

## Design Philosophy

> Simple outside.
> Explicit inside.
> Predictable beats perfect.

This system treats vector search as **infrastructure**, not magic.

---

# Implementation Plan — v1 (4 Milestones)

The v1 goal is **a working, SSD-first vector search system** with a clean Python SDK and a C++ core — **not** a managed-service clone.

Each milestone ends with something **runnable and testable**.

---

## Milestone 1 — Core ANN Engine (C++)

**Goal:**
Be able to **build and query a DiskANN-style index on SSD** from C++ only.

### Scope

This milestone is *pure backend*, no API, no SDK.

### Deliverables

* C++ library that can:

  * Build an ANN index from vectors
  * Persist index files to SSD
  * Load index files
  * Execute `search(vector, top_k)` with predictable latency
* Deterministic builds (same input → same index)
* Basic benchmark tool (latency, recall on small dataset)

### Key Components

* DiskANN integration or equivalent ANN core
* SSD IO abstraction
* Vector format + ID mapping
* Read-only search path

### Explicit Assumptions

* Index is **immutable**
* No updates or deletes
* Single-threaded build is acceptable
* Multi-threaded read path is optional

### Out of Scope

* Delta index
* Metadata
* Filtering
* Python bindings
* Networking

### Exit Criteria

* You can:

  ```bash
  build_index vectors.bin index_dir/
  query_index index_dir/ query.vec
  ```
* Index survives process restart
* Queries do not require full index in RAM

---

## Milestone 2 — Delta Layer + Index Lifecycle

**Goal:**
Support **near-real-time writes** without mutating the main index.

### Scope

Still local / single-node, but introduces **index lifecycle discipline**.

### Deliverables

* Delta index implementation (simple but correct)
* Query fan-out across:

  * Delta index
  * Main ANN index
* Background index builder
* Atomic index publish (swap old → new)

### Key Components

* Mutable delta store (in-memory or append-only SSD)
* Index builder process
* Snapshot publishing mechanism
* Index state tracking (building / published / retired)

### Explicit Assumptions

* Delta index is small and bounded
* Delta recall may be worse than main index
* Rebuilds are asynchronous
* Queries always hit *both* layers

### Out of Scope

* Metadata filtering
* API layer
* Python SDK
* Crash recovery beyond rebuild

### Exit Criteria

* New vectors become searchable quickly
* Main index can be rebuilt without blocking queries
* Atomic swap does not crash queries

---

## Milestone 3 — Query Service + Metadata

**Goal:**
Turn the engine into a **real service**, with filtering and clean separation.

### Scope

This is where it becomes *production-shaped*.

### Deliverables

* Query service (C++)
* Metadata store integration
* Filtering (pre or post ANN)
* WAL for durability
* Clear read/write paths

### Key Components

* Query service process
* Metadata store (simple KV or embedded DB)
* WAL / append log
* Result merge + ranking

### Explicit Assumptions

* Metadata is small
* Metadata may lag slightly
* Filtering is not perfectly optimal
* Service is stateless

### Out of Scope

* Authentication
* Multi-tenant isolation
* SDK polish
* Observability beyond logs + counters

### Exit Criteria

* You can:

  * Insert vectors + metadata
  * Restart the service
  * Query with filters
* Index rebuild recovers from WAL

---

## Milestone 4 — Python SDK + DX Hardening

**Goal:**
Make it **pleasant to use**, without lying about guarantees.

### Scope

No core algorithm changes — this is *developer experience*.

### Deliverables

* Python SDK
* Simple API layer (HTTP or gRPC)
* Clear error messages
* Timeouts, retries, batching
* Minimal metrics

### Key Components

* Python client
* API gateway
* Request validation
* Mapping SDK calls → service calls

### Explicit Assumptions

* SDK hides indexing complexity
* SDK users accept async behavior
* Defaults matter more than flexibility

### Out of Scope

* Auto-scaling
* Multi-region
* Tiered storage
* Hybrid search

### Exit Criteria

* A user can:

  ```python
  client.upsert(...)
  client.query(...)
  ```
* Without knowing:

  * What DiskANN is
  * How indexes are rebuilt
* System behaves predictably under load

---

## Final v1 Definition of “Done”

v1 is complete when:

* ✅ Index is SSD-resident
* ✅ Writes are durable
* ✅ Queries are predictable
* ✅ SDK is boring and simple
* ✅ Trade-offs are explicit, not hidden

Not when:

* ❌ It matches Pinecone feature-for-feature
* ❌ It promises perfect real-time recall
* ❌ It hides complexity behind marketing

