#!/usr/bin/env python3
"""
Optimized Embedding Pipeline for Multi-domain RAG.
Single unified collection for: Legal docs, Textbooks, etc.
Best HNSW config for high recall across diverse content.
"""
import os
import json
import time
import sys
import hashlib
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    PayloadSchemaType, TextIndexParams, TokenizerType,
    OptimizersConfigDiff, HnswConfigDiff
)
from src.client import client as vnpt_client, RateLimitException

# ==============================================
# CONFIG - OPTIMIZED FOR MULTI-DOMAIN RAG
# ==============================================
PROCESSED_DIR = "data/processed"
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "vnpt_rag"  # Unified collection for all domains
EMBEDDING_DIM = 1024
CHECKPOINT_FILE = "data/processed/embed_checkpoint.json"

# HNSW Optimal Parameters for High Recall Multi-domain
HNSW_M = 32              # Connections per node (higher = better recall, more RAM)
HNSW_EF_CONSTRUCT = 200  # Build accuracy (higher = better index quality)
HNSW_EF_SEARCH = 256     # Search accuracy (set at query time)

# Rate limit handling
INITIAL_BACKOFF = 30
MAX_BACKOFF = 300
BATCH_SIZE = 50
# Optimized for 500 requests/minute (0.12s total delay per request)
REQUEST_DELAY = 0.08

# ==============================================
# CHECKPOINT
# ==============================================
def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
            data["embedded_ids"] = set(data.get("embedded_ids", []))
            return data
    return {"embedded_ids": set(), "total_embedded": 0}

def save_checkpoint(checkpoint):
    cp = {
        "embedded_ids": list(checkpoint["embedded_ids"]),
        "total_embedded": checkpoint["total_embedded"]
    }
    with open(CHECKPOINT_FILE, 'w', encoding='utf-8') as f:
        json.dump(cp, f)

# ==============================================
# QDRANT - OPTIMAL CONFIG FOR MULTI-DOMAIN
# ==============================================
def init_qdrant():
    """Create unified collection if not exists (Incremental Support)."""
    qdrant = QdrantClient(url=QDRANT_URL)
    
    try:
        collections = [c.name for c in qdrant.get_collections().collections]
        
        if COLLECTION_NAME in collections:
            print(f"✓ Using existing collection: {COLLECTION_NAME}")
            info = qdrant.get_collection(COLLECTION_NAME)
            print(f"   Points: {info.points_count}")
            return qdrant
            
    except Exception as e:
        print(f"⚠ Connection check failed, attempting creation: {e}")
        
    print(f"\n{'='*60}")
    print(f"🔧 Creating Unified Collection: {COLLECTION_NAME}")
    print(f"{'='*60}")
    
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=EMBEDDING_DIM,
            distance=Distance.COSINE,
            on_disk=True
        ),
        # OPTIMAL HNSW for multi-domain high recall
        hnsw_config=HnswConfigDiff(
            m=HNSW_M,
            ef_construct=HNSW_EF_CONSTRUCT,
            full_scan_threshold=20000,
            max_indexing_threads=0,  # Auto
            on_disk=False  # Keep index in RAM for speed
        ),
        optimizers_config=OptimizersConfigDiff(
            indexing_threshold=20000,
            memmap_threshold=50000
        ),
        on_disk_payload=True
    )
    
    print(f"\n📊 HNSW Configuration:")
    print(f"   m = {HNSW_M} (connections/node)")
    print(f"   ef_construct = {HNSW_EF_CONSTRUCT} (build accuracy)")
    print(f"   ef_search = {HNSW_EF_SEARCH} (at query time)")
    
    # Payload indexes for filtering
    print(f"\n📝 Creating Payload Indexes...")
    
    # Domain filter (legal, textbook, etc.)
    qdrant.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="domain",
        field_schema=PayloadSchemaType.KEYWORD
    )
    
    qdrant.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="doc_id",
        field_schema=PayloadSchemaType.KEYWORD
    )
    
    qdrant.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="doc_title",
        field_schema=PayloadSchemaType.KEYWORD
    )
    
    # Full-text for hybrid search
    print(f"🔍 Creating Full-text Index...")
    qdrant.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="content",
        field_schema=TextIndexParams(
            type="text",
            tokenizer=TokenizerType.MULTILINGUAL,
            min_token_len=2,
            max_token_len=20,
            lowercase=True
        )
    )
    
    print(f"\n✅ Collection created successfully!")
    print(f"   📦 Dense: {EMBEDDING_DIM}D COSINE")
    print(f"   🔗 HNSW: m={HNSW_M}, ef={HNSW_EF_CONSTRUCT}")
    print(f"   🏷️  Indexes: domain, doc_id, doc_title")
    print(f"   📖 Full-text: multilingual")
    print(f"{'='*60}\n")

    return qdrant

# ==============================================
# EMBEDDING
# ==============================================
def generate_point_id(chunk_id: str) -> int:
    hash_bytes = hashlib.md5(chunk_id.encode()).digest()
    return int.from_bytes(hash_bytes[:8], byteorder='big') % (2**63)

def embed_with_retry(text: str, max_retries: int = 10) -> list:
    backoff = INITIAL_BACKOFF
    
    for attempt in range(max_retries):
        try:
            embedding = vnpt_client.get_embedding(text)
            if embedding and len(embedding) == EMBEDDING_DIM:
                return embedding
            return None
        except RateLimitException:
            print(f"\n⚠ Rate Limit - waiting {backoff}s (attempt {attempt + 1})...")
            time.sleep(backoff)
            backoff = min(backoff * 2, MAX_BACKOFF)
        except Exception as e:
            print(f"  ✗ Error: {e}")
            return None
    return None

def embed_chunks(chunks: list, qdrant: QdrantClient, checkpoint: dict, domain: str = "legal"):
    """Embed chunks to unified collection with domain tag."""
    embedded_ids = checkpoint["embedded_ids"]
    total_embedded = checkpoint.get("total_embedded", 0)
    
    remaining = [c for c in chunks if c["chunk_id"] not in embedded_ids]
    
    print(f"\n{'='*60}")
    print(f"📊 Embedding: {domain.upper()}")
    print(f"{'='*60}")
    print(f"  Total: {len(chunks)} | Done: {len(embedded_ids)} | Remaining: {len(remaining)}")
    print(f"{'='*60}\n")
    
    if not remaining:
        print("✓ All chunks already embedded!")
        return
    
    points_batch = []
    start_time = time.time()
    
    for idx, chunk in enumerate(remaining):
        chunk_id = chunk["chunk_id"]
        content = chunk.get("content", "")
        
        if not content or len(content) < 20:
            continue
        
        if len(content) > 8000:
            content = content[:8000]
        
        progress = (idx + 1) / len(remaining) * 100
        elapsed = time.time() - start_time
        rate = (idx + 1) / elapsed if elapsed > 0 else 0
        eta = (len(remaining) - idx - 1) / rate / 60 if rate > 0 else 0
        
        print(f"[{idx + 1}/{len(remaining)}] {progress:.1f}% | ETA: {eta:.1f}min | {chunk_id[:30]}...")
        
        embedding = embed_with_retry(content)
        
        if embedding is None:
            continue
        
        # Point with domain tag for multi-domain collection
        point = PointStruct(
            id=generate_point_id(chunk_id),
            vector=embedding,
            payload={
                "chunk_id": chunk_id,
                "domain": domain,  # legal, textbook, etc.
                "doc_id": chunk.get("doc_id", ""),
                "doc_title": chunk.get("doc_title", ""),
                "article_num": chunk.get("article_num", ""),
                "chapter": chunk.get("chapter", ""),
                "content": content
            }
        )
        points_batch.append(point)
        embedded_ids.add(chunk_id)
        total_embedded += 1
        
        if len(points_batch) >= BATCH_SIZE:
            qdrant.upsert(collection_name=COLLECTION_NAME, points=points_batch)
            checkpoint["embedded_ids"] = embedded_ids
            checkpoint["total_embedded"] = total_embedded
            save_checkpoint(checkpoint)
            print(f"  ✓ Batch ({BATCH_SIZE}) | Total: {total_embedded}")
            points_batch = []
        
        time.sleep(REQUEST_DELAY)
    
    if points_batch:
        qdrant.upsert(collection_name=COLLECTION_NAME, points=points_batch)
        checkpoint["embedded_ids"] = embedded_ids
        checkpoint["total_embedded"] = total_embedded
        save_checkpoint(checkpoint)
    
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"✅ Embedding Complete: {domain.upper()}")
    print(f"   Total: {total_embedded} | Time: {elapsed/60:.1f}min")
    print(f"   UI: http://localhost:6333/dashboard")
    print(f"{'='*60}")

# ==============================================
# MAIN
# ==============================================
def main():
    print(f"\n{'='*60}")
    print(f"🚀 VNPT Multi-domain RAG - Embedding Pipeline")
    print(f"   {datetime.now()}")
    print(f"{'='*60}\n")
    
    chunks_file = f"{PROCESSED_DIR}/chunks.json"
    if not os.path.exists(chunks_file):
        print(f"✗ Not found: {chunks_file}")
        return
    
    with open(chunks_file, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    print(f"✓ Loaded {len(chunks)} chunks")
    
    qdrant = init_qdrant()
    checkpoint = load_checkpoint()
    
    # Embed with domain tag
    embed_chunks(chunks, qdrant, checkpoint, domain="legal")
    
    print(f"\n🏁 Finished: {datetime.now()}")

if __name__ == "__main__":
    main()
