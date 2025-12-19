#!/usr/bin/env python3
"""
Optimized Embedding Pipeline for Multi-domain RAG.
Single unified collection for: Legal docs, Textbooks, etc.
Best HNSW config for high recall across diverse content.

Usage: 
    uv run data_pipeline/embedder.py
"""
import os
import json
import time
import sys
import hashlib
from datetime import datetime
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Generator

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    PayloadSchemaType, TextIndexParams, TokenizerType,
    OptimizersConfigDiff, HnswConfigDiff
)
from src.client import client as vnpt_client, RateLimitException

# Config
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "vnpt_rag_v2" 
EMBEDDING_DIM = 1024
CHECKPOINT_FILE = "data/qdrant_storage/embed_checkpoint_v2.json" # tracking progress

# HNSW Optimal Parameters for High Recall Multi-domain
HNSW_M = 16              # Connections per node (higher = better recall, more RAM)
HNSW_EF_CONSTRUCT = 200  # Build accuracy (higher = better index quality)
HNSW_EF_SEARCH = 100     # Search accuracy (set at query time)

# Rate limit handling
INITIAL_BACKOFF = 30     # Reduce backoff to 30s to retry faster if only temporarily rate limited
MAX_BACKOFF = 300
BATCH_SIZE = 50          # Group 50 vectors to upload at once
REQUEST_DELAY = 0

# Junk Filtering
JUNK_PATTERNS = [
    r"đăng nhập", r"đăng ký", r"quên mật khẩu", r"chia sẻ qua email", 
    r"bản quyền thuộc", r"liên hệ quảng cáo", r"về đầu trang", 
    r"xem thêm", r"bình luận", r"báo xấu", r"trang chủ", 
    r"facebook", r"twitter", r"linkedin", r"zalo", 
    r"kết nối với chúng tôi", r"thông tin tòa soạn",
    r"wikipedia", r"bách khoa toàn thư", r"sửa đổi", r"biểu quyết",
]

import re as regex_module

def _is_junk_text(text: str) -> bool:
    """Filter out junk text (nav, footer, ads) from chunks."""
    if len(text.split()) < 5:  # Too short
        return True
    text_lower = text.lower()
    for pattern in JUNK_PATTERNS:
        if regex_module.search(pattern, text_lower):
            return True
    return False




# Rate Limiter
class RateLimiter:
    """Thread-safe rate limiter to enforce max requests per minute."""
    def __init__(self, max_calls_per_minute: int):
        self.interval = 60.0 / max_calls_per_minute
        self.lock = threading.Lock()
        self.last_call = 0.0

    def acquire(self):
        """Block until a request can be made."""
        with self.lock:
            now = time.time()
            elapsed = now - self.last_call
            wait = self.interval - elapsed
            
            if wait > 0:
                time.sleep(wait)
            
            self.last_call = time.time()

# Set 480 RPM instead of 500 RPM, vnpt_embedding has quota limit 500 RPM
# Avoid server block due to clock skew or network delay
rate_limiter = RateLimiter(480)


def _prepend_title_to_chunk(chunk_text: str, title: str) -> str:
    """Prepend title to chunk content for better context in embeddings."""
    if title and title.strip():
        return f"Title: {title.strip()}\nContent: {chunk_text}"
    return chunk_text

# Checkpoint
def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
            data["embedded_ids"] = set(data.get("embedded_ids", []))
            return data
    return {"embedded_ids": set(), "total_embedded": 0}

def sync_checkpoint_from_qdrant(qdrant, checkpoint):
    """
    Sync local checkpoint with actual Qdrant state.
    Useful if checkpoint file was deleted but collection exists.
    """
    try:
        # Check if collection exists
        collections = [c.name for c in qdrant.get_collections().collections]
        if COLLECTION_NAME not in collections:
            return checkpoint

        print("Syncing checkpoint from Qdrant...")
        
        # Scroll all points to get chunk_ids
        offset = None
        count = 0
        
        while True:
            points, next_offset = qdrant.scroll(
                collection_name=COLLECTION_NAME,
                scroll_filter=None,
                limit=1000,
                with_payload=["chunk_id"],
                with_vectors=False,
                offset=offset
            )
            
            for point in points:
                if point.payload and "chunk_id" in point.payload:
                    checkpoint["embedded_ids"].add(point.payload["chunk_id"])
                    count += 1
            
            offset = next_offset
            if offset is None:
                break
                
        checkpoint["total_embedded"] = len(checkpoint["embedded_ids"])
        print(f"Synced {count} points from Qdrant")
        save_checkpoint(checkpoint)
        return checkpoint
        
    except Exception as e:
        print(f"Sync failed: {e}")
        return checkpoint

def save_checkpoint(checkpoint):
    cp = {
        "embedded_ids": list(checkpoint["embedded_ids"]),
        "total_embedded": checkpoint["total_embedded"]
    }
    # Ensure dir exists
    os.makedirs(os.path.dirname(CHECKPOINT_FILE), exist_ok=True)
    
    with open(CHECKPOINT_FILE, 'w', encoding='utf-8') as f:
        json.dump(cp, f)

# Qdrant - Optimal Config for Multi-Domain
def init_qdrant():
    """Create unified collection if not exists (Incremental Support)."""
    qdrant = QdrantClient(url=QDRANT_URL)
    
    try:
        collections = [c.name for c in qdrant.get_collections().collections]
        
        if COLLECTION_NAME in collections:
            print(f"Using existing collection: {COLLECTION_NAME}")
            info = qdrant.get_collection(COLLECTION_NAME)
            return qdrant
            
    except Exception as e:
        print(f"Connection check failed, attempting creation: {e}")
        
    print(f"\n{'='*60}")
    print(f"Creating Unified Collection: {COLLECTION_NAME}")
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
    
    print(f"\nHNSW Configuration:")
    print(f"m = {HNSW_M} (connections/node)")
    print(f"ef_construct = {HNSW_EF_CONSTRUCT} (build accuracy)")
    print(f"ef_search = {HNSW_EF_SEARCH} (at query time)")
    
    # Payload indexes for filtering
    print(f"\nCreating Payload Indexes...")
    
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
    print(f"Creating Full-text Index...")
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
    
    print(f"\nCollection created successfully!")
    print(f"Dense: {EMBEDDING_DIM}D COSINE")
    print(f"HNSW: m={HNSW_M}, ef={HNSW_EF_CONSTRUCT}")
    print(f"Indexes: domain, doc_id, doc_title")
    print(f"Full-text: multilingual")
    print(f"{'='*60}\n")

    return qdrant

# Embedding
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
            # Overwrite line with warning
            msg = f"Rate Limit - waiting {backoff}s (attempt {attempt + 1})..."
            sys.stdout.write(f"\r\033[2K{msg}")
            sys.stdout.flush()
            time.sleep(backoff)
            backoff = min(backoff * 2, MAX_BACKOFF)
        except Exception as e:
            # Overwrite line with error
            msg = f"Error: {e}"
            sys.stdout.write(f"\r\033[2K{msg}")
            sys.stdout.flush()
            return None
    return None

def process_single_chunk(chunk: dict, domain: str) -> dict:
    """Process a single chunk: prepare content, embed, and return point struct."""
    chunk_id = chunk["chunk_id"]
    content = chunk.get("content", "")
    
    if not content or len(content) < 20:
        return None
    
    if len(content) > 8000:
        content = content[:8000]
    
    # Apply title-aware embedding
    if "doc_title" in chunk:
        content_to_embed = _prepend_title_to_chunk(content, chunk.get("doc_title", ""))
    else:
        content_to_embed = content
    
    # Block for rate limit
    rate_limiter.acquire()
    
    # Embed
    embedding = embed_with_retry(content_to_embed)
    
    if embedding is None:
        return None
    
    return PointStruct(
        id=generate_point_id(chunk_id),
        vector=embedding,
        payload={
            "chunk_id": chunk_id,
            "domain": chunk.get("domain", domain),
            "doc_id": chunk.get("doc_id", ""),
            "doc_title": chunk.get("doc_title", ""),
            "article_num": chunk.get("article_num", ""),
            "chapter": chunk.get("chapter", ""),
            "content": content
        }
    )


def embed_chunks(chunks: list, qdrant: QdrantClient, checkpoint: dict, domain: str = "legal"):
    """Embed chunks concurrently."""
    embedded_ids = checkpoint["embedded_ids"]
    total_embedded = checkpoint.get("total_embedded", 0)
    
    remaining = [c for c in chunks if c["chunk_id"] not in embedded_ids]
    
    done_in_current_set = len(chunks) - len(remaining)

    print(f"\n{'='*60}")
    print(f"Embedding: {domain.upper()} (Multithreaded: 8 workers, 500 RPM)") # Change print if change thread pool
    print(f"{'='*60}")
    print(f"Total points: {len(chunks)}")
    print(f"Embedded points: {done_in_current_set}")
    print(f"Remaining points: {len(remaining)}")
    print(f"{'='*60}\n")
    
    if not remaining:
        print("All chunks already embedded!")
        return
    
    points_batch = []
    start_time = time.time()
    batch_lock = threading.Lock()
    
    # Thread pool
    MAX_WORKERS = 8
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all tasks
        future_to_chunk = {executor.submit(process_single_chunk, chunk, domain): chunk for chunk in remaining}
        completed_count = 0
        
        for future in as_completed(future_to_chunk):
            chunk = future_to_chunk[future]
            completed_count += 1
            
            try:
                point = future.result()
                if point:
                    with batch_lock:
                        points_batch.append(point)
                        embedded_ids.add(chunk["chunk_id"])
                        total_embedded += 1
                        
                        if len(points_batch) >= BATCH_SIZE:
                            qdrant.upsert(collection_name=COLLECTION_NAME, points=points_batch)
                            checkpoint["embedded_ids"] = embedded_ids
                            checkpoint["total_embedded"] = total_embedded
                            save_checkpoint(checkpoint)
                            points_batch = []
            except Exception as e:
                # Log error thread-safely
                msg = f"Error processing chunk: {e}"
                sys.stdout.write(f"\r\033[2K{msg}")
                sys.stdout.flush()

            # Logging progress (thread-safe due to GIL on I/O, but good to be careful)
            elapsed = time.time() - start_time
            if elapsed > 0:
                rpm = completed_count / (elapsed / 60)
            else:
                rpm = 0
                
            elapsed_str = time.strftime("%M:%S", time.gmtime(elapsed))
            
            # Throttle log updates to avoid slowing down (e.g., every 10 chunks or 0.5s)
            if completed_count % 10 == 0 or completed_count == len(remaining):
                msg = f"Progress: {completed_count/len(remaining)*100:5.1f}% | {completed_count}/{len(remaining)} | RPM: {rpm:.0f} | Time: {elapsed_str}"
                sys.stdout.write(f"\r\033[2K{msg}")
                sys.stdout.flush()
    
    # Upsert remaining
    if points_batch:
        qdrant.upsert(collection_name=COLLECTION_NAME, points=points_batch)
        checkpoint["embedded_ids"] = embedded_ids
        checkpoint["total_embedded"] = total_embedded
        save_checkpoint(checkpoint)
    
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Embedding Complete: {domain.upper()}")
    print(f"Total: {total_embedded} | Time: {elapsed/60:.1f}min")
    print(f"UI: http://localhost:6333/dashboard")
    print(f"{'='*60}")

def main():
    print(f"\n{'='*60}")
    print(f"VNPT Multi-domain RAG - Embedding Pipeline")
    print(f"{datetime.now()}")
    print(f"{'='*60}\n")
    
    # Load chunks from all sources
    all_chunks = []
    
    # 1. Wikipedia
    wiki_chunks = []
    wiki_path = "data/wiki/processed/chunks.jsonl"
    if os.path.exists(wiki_path):
        print(f"Loading Wikipedia: {wiki_path}")
        with open(wiki_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        chunk = json.loads(line)
                        # Ensure domain and ID
                        chunk["domain"] = "wiki"
                        if "chunk_id" not in chunk:
                            chunk["chunk_id"] = chunk.get("id", f"wiki_{hashlib.md5(chunk.get('chunk_text','').encode()).hexdigest()[:12]}")
                        
                        # Map chunk_text to content if needed
                        if "content" not in chunk:
                            chunk["content"] = chunk.get("chunk_text", "")
                            
                        wiki_chunks.append(chunk)
                    except: pass
        print(f"Loaded {len(wiki_chunks)} Wikipedia chunks")

    # 2. TVPL & VBPL (Legal)
    legal_chunks = []
    for source in ["tvpl", "vbpl"]:
        base_path = f"data/{source}/processed/chunks"
        path = None
        is_jsonl = True
        
        if os.path.exists(f"{base_path}.jsonl"):
            path = f"{base_path}.jsonl"
        elif os.path.exists(f"{base_path}.json"):
            path = f"{base_path}.json"
            is_jsonl = False
            
        if path:
            print(f"Loading {source.upper()}: {path}")
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    if is_jsonl:
                        for line in f:
                            if line.strip():
                                try:
                                    chunk = json.loads(line)
                                    chunk["domain"] = "legal"
                                    legal_chunks.append(chunk)
                                except: pass
                    else:
                        # Standard JSON list
                        chunks = json.load(f)
                        for chunk in chunks:
                            chunk["domain"] = "legal"
                            legal_chunks.append(chunk)
            except Exception as e:
                print(f"Error loading {path}: {e}")
                
            print(f"Loaded {len(legal_chunks)} legal chunks (Accumulated)")

    # 3. Dethitracnghiem (MCQ)
    mcq_chunks = []
    mcq_path = "data/dethitracnghiem/processed/chunks.jsonl"
    if os.path.exists(mcq_path):
        print(f"Loading MCQ: {mcq_path}")
        with open(mcq_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        chunk = json.loads(line)
                        mcq_chunks.append({
                            "chunk_id": chunk.get("metadata", {}).get("uid", f"mcq_{len(mcq_chunks)}"),
                            "content": chunk.get("content", ""),
                            "domain": "mcq",
                            **chunk.get("metadata", {})
                        })
                    except: pass
        print(f"Loaded {len(mcq_chunks)} MCQ chunks")
    
    # 4. ViWiki2025 (Wikipedia)
    viwiki_chunks = []
    viwiki_path = "data/ViWiki2025/processed/chunks.jsonl"
    if os.path.exists(viwiki_path):
        print(f"Loading ViWiki2025: {viwiki_path}")
        with open(viwiki_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        chunk = json.loads(line)
                        # Ensure domain is set
                        if "domain" not in chunk:
                            chunk["domain"] = "viwiki2025"
                        viwiki_chunks.append(chunk)
                    except: pass
        print(f"Loaded {len(viwiki_chunks)} ViWiki2025 chunks")
    
    all_chunks = wiki_chunks + legal_chunks + mcq_chunks + viwiki_chunks
    
    if not all_chunks:
        print("No chunks found in data/vbpl/processed/, data/tvpl/processed/, or data/dethitracnghiem/processed/")
        return
        
    chunks = all_chunks
    
    print(f"Loaded {len(chunks)} chunks")
    
    qdrant = init_qdrant()
    checkpoint = load_checkpoint()
    
    # Sync if empty checkpoint (e.g. first run or deleted file)
    if not checkpoint["embedded_ids"]:
        checkpoint = sync_checkpoint_from_qdrant(qdrant, checkpoint)
    
    # Embed with domain tag
    embed_chunks(chunks, qdrant, checkpoint, domain="legal")
    print(f"\nFinished: {datetime.now()}")

if __name__ == "__main__":
    main()