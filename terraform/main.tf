terraform {
  required_version = ">= 1.6.0, < 2.0.0"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 6.0"
    }
  }
}

provider "google" {
  project = var.project_id
}

resource "google_bigquery_dataset" "hermes_memory" {
  dataset_id  = var.dataset_id
  location    = var.location
  description = "Hermes Agent memory layer — Memory Bank mirror + analytics"
  labels      = { app = "hermes", layer = "memory" }
}

resource "google_bigquery_table" "memories" {
  dataset_id          = google_bigquery_dataset.hermes_memory.dataset_id
  table_id            = "memories"
  description         = "Mirror of Memory Bank facts"
  schema              = jsonencode(jsondecode(file("${path.module}/schemas/memories.json")).fields)
  deletion_protection = false
}

resource "google_bigquery_table" "sessions" {
  dataset_id          = google_bigquery_dataset.hermes_memory.dataset_id
  table_id            = "sessions"
  schema              = jsonencode(jsondecode(file("${path.module}/schemas/sessions.json")).fields)
  deletion_protection = false
}

resource "google_bigquery_table" "memory_revisions" {
  dataset_id          = google_bigquery_dataset.hermes_memory.dataset_id
  table_id            = "memory_revisions"
  schema              = jsonencode(jsondecode(file("${path.module}/schemas/memory_revisions.json")).fields)
  deletion_protection = false
}

resource "google_bigquery_table" "document_sources" {
  dataset_id          = google_bigquery_dataset.hermes_memory.dataset_id
  table_id            = "document_sources"
  description         = "Canonical document source identity and revision lifecycle"
  schema              = jsonencode(jsondecode(file("${path.module}/schemas/document_sources.json")).fields)
  clustering          = ["user_id", "agent_name", "corpus_id", "source_kind"]
  deletion_protection = true

  lifecycle {
    prevent_destroy = true
  }
}

resource "google_bigquery_table" "document_chunks" {
  dataset_id          = google_bigquery_dataset.hermes_memory.dataset_id
  table_id            = "document_chunks"
  description         = "Citation-bearing document chunks and retrieval embeddings"
  schema              = jsonencode(jsondecode(file("${path.module}/schemas/document_chunks.json")).fields)
  clustering          = ["user_id", "agent_name", "corpus_id", "source_id"]
  deletion_protection = true

  lifecycle {
    prevent_destroy = true
  }
}
