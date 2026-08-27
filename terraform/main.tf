terraform {
  required_providers {
    google = { source = "hashicorp/google", version = ">= 5.0" }
  }
}

variable "project_id" { type = string }
variable "location"   { type = string  default = "US" }
variable "dataset_id" { type = string  default = "hermes_memory" }

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
  dataset_id = google_bigquery_dataset.hermes_memory.dataset_id
  table_id   = "memories"
  description = "Mirror of Memory Bank facts"
  schema = file("${path.module}/schemas/memories.json")
  deletion_protection = false
}

resource "google_bigquery_table" "sessions" {
  dataset_id = google_bigquery_dataset.hermes_memory.dataset_id
  table_id   = "sessions"
  schema = file("${path.module}/schemas/sessions.json")
  deletion_protection = false
}

resource "google_bigquery_table" "memory_revisions" {
  dataset_id = google_bigquery_dataset.hermes_memory.dataset_id
  table_id   = "memory_revisions"
  schema = file("${path.module}/schemas/memory_revisions.json")
  deletion_protection = false
}
