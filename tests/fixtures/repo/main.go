package main

import (
	"fmt"
	"sort"
)

// Memory is one deterministic record in the fixture store.
type Memory struct {
	ID   string
	Fact string
}

// Store describes the operations used by the sample program.
type Store interface {
	Put(Memory) error
	Snapshot() []Memory
}

// InMemoryStore is a fake implementation with no external dependencies.
type InMemoryStore struct {
	items map[string]Memory
}

// NewStore constructs an empty fixture store.
func NewStore() *InMemoryStore {
	return &InMemoryStore{items: make(map[string]Memory)}
}

// Put records a memory by deterministic identifier.
func (s *InMemoryStore) Put(memory Memory) error {
	if memory.ID == "" {
		return fmt.Errorf("memory ID is required")
	}
	s.items[memory.ID] = memory
	return nil
}

// Snapshot returns records sorted by identifier.
func (s *InMemoryStore) Snapshot() []Memory {
	ids := make([]string, 0, len(s.items))
	for id := range s.items {
		ids = append(ids, id)
	}
	sort.Strings(ids)

	out := make([]Memory, 0, len(ids))
	for _, id := range ids {
		out = append(out, s.items[id])
	}
	return out
}

func main() {
	store := NewStore()
	_ = store.Put(Memory{ID: "fake-memory-001", Fact: "synthetic fixture"})
	fmt.Println(store.Snapshot())
}
