# Fixture Memory Service

This synthetic repository exercises Markdown and Go source ingestion without network access.

## Usage

1. Construct the in-memory store.
2. Add a record with a fake identifier.
3. Print a stable snapshot.

```bash
go run ./main.go
```

## Recovery

- Re-run the program.
- Confirm `fake-memory-001` is present.

## Design

The fixture contains an interface, structs, functions, and methods so symbol-aware chunking has stable targets.

## Recovery

This repeated heading verifies heading occurrence participates in chunk identity.
