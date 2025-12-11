#!/usr/bin/env python3
"""
Optimized data processing pipeline for Vietnamese legal documents.
- Deep text cleaning
- Article-based chunking with rich metadata
- Context enrichment for better retrieval
"""
import os
import json
import re
from glob import glob
from typing import List, Dict
import unicodedata

DATA_SOURCE_DIR = "data/data_source"
PROCESSED_DIR = "data/processed"

# ============================================
# TEXT CLEANING
# ============================================

def clean_text(text: str) -> str:
    """Deep clean Vietnamese legal text."""
    if not text:
        return ""
    
    # 1. Normalize Unicode (NFC for Vietnamese)
    text = unicodedata.normalize('NFC', text)
    
    # 2. Remove HTML entities
    text = re.sub(r'&[a-zA-Z]+;', ' ', text)
    text = re.sub(r'&#\d+;', ' ', text)
    
    # 3. Remove special characters but keep Vietnamese
    text = re.sub(r'[^\w\s\.\,\;\:\-\(\)\[\]\/\"\'\%\°\§\đĐ\u00C0-\u1EF9]', ' ', text)
    
    # 4. Normalize whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'^\s+', '', text, flags=re.MULTILINE)
    
    # 5. Remove common noise patterns
    noise_patterns = [
        r'Văn bản quy phạm pháp luật',
        r'Văn bản hợp nhất',
        r'Hệ thống hóa VBQPPL',
        r'Đang cập nhật',
        r'Loading\.\.\.',
    ]
    for pattern in noise_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    return text.strip()

def extract_doc_metadata(doc: dict) -> dict:
    """Extract useful metadata from document."""
    info = doc.get("info", {})
    
    # Try to extract key fields
    so_hieu = info.get("Số hiệu", info.get("Số/Ký hiệu", ""))
    ngay_ban_hanh = info.get("Ngày ban hành", info.get("Ngày", ""))
    co_quan = info.get("Cơ quan ban hành", info.get("Nguồn", ""))
    hieu_luc = info.get("Tình trạng hiệu lực", info.get("Hiệu lực", ""))
    
    return {
        "so_hieu": so_hieu,
        "ngay_ban_hanh": ngay_ban_hanh,
        "co_quan": co_quan,
        "hieu_luc": hieu_luc
    }

# ============================================
# CHUNKING STRATEGIES
# ============================================

def chunk_by_article(content: str, doc_title: str, doc_meta: dict) -> List[Dict]:
    """
    Chunk by Điều (Article) with rich context.
    Each chunk includes: doc title, chapter, article number, content
    """
    chunks = []
    
    # Pattern: "Điều X" or "Điều X."
    article_pattern = r'(Điều\s+\d+[a-z]?\.?\s*)'
    
    # Split by Điều
    parts = re.split(article_pattern, content, flags=re.IGNORECASE)
    
    current_chapter = ""
    current_section = ""
    
    i = 0
    while i < len(parts):
        part = parts[i].strip()
        
        # Track Chapter (Chương)
        chapter_match = re.search(r'(Chương\s+[IVXLCDM\d]+[^\n]*)', part, re.IGNORECASE)
        if chapter_match:
            current_chapter = chapter_match.group(1).strip()
        
        # Track Section (Mục)
        section_match = re.search(r'(Mục\s+\d+[^\n]*)', part, re.IGNORECASE)
        if section_match:
            current_section = section_match.group(1).strip()
        
        # If this is an article header
        if re.match(r'Điều\s+\d+', part, re.IGNORECASE):
            article_num = part.strip()
            
            if i + 1 < len(parts):
                article_content = parts[i + 1].strip()
                
                # Get first line as article title
                lines = article_content.split('\n')
                article_title = lines[0].strip()[:100] if lines else ""
                
                # Build rich context chunk
                # Format: [Doc Title] - [Chapter] - Điều X: [Title]
                context_header = f"[{doc_title}]"
                if current_chapter:
                    context_header += f" - {current_chapter}"
                context_header += f" - {article_num}"
                if article_title:
                    context_header += f": {article_title}"
                
                full_text = f"{context_header}\n\n{article_content}"
                
                if len(article_content) > 30:
                    chunks.append({
                        "type": "article",
                        "article_num": article_num,
                        "article_title": article_title,
                        "chapter": current_chapter,
                        "section": current_section,
                        "context_header": context_header,
                        "content": clean_text(full_text),
                        "char_count": len(full_text)
                    })
                
                i += 2
                continue
        
        i += 1
    
    return chunks

def chunk_by_semantic_blocks(content: str, doc_title: str, 
                             target_size: int = 1500, 
                             min_size: int = 500,
                             overlap: int = 200) -> List[Dict]:
    """
    Fallback: Chunk by semantic blocks (paragraphs) with overlap.
    Target ~1500 chars per chunk for optimal retrieval.
    """
    chunks = []
    content = clean_text(content)
    
    if len(content) <= target_size:
        return [{
            "type": "full_doc",
            "content": f"[{doc_title}]\n\n{content}",
            "char_count": len(content)
        }]
    
    # Split by double newline (paragraphs)
    paragraphs = re.split(r'\n\n+', content)
    
    current_chunk = f"[{doc_title}]\n\n"
    chunk_idx = 0
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        
        # Check if adding this paragraph exceeds target
        if len(current_chunk) + len(para) > target_size and len(current_chunk) > min_size:
            chunks.append({
                "type": "block",
                "chunk_idx": chunk_idx,
                "content": current_chunk.strip(),
                "char_count": len(current_chunk)
            })
            chunk_idx += 1
            
            # Overlap: keep last ~200 chars
            overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else ""
            current_chunk = f"[{doc_title}]\n\n{overlap_text}\n\n{para}\n\n"
        else:
            current_chunk += para + "\n\n"
    
    # Final chunk
    if len(current_chunk.strip()) > min_size:
        chunks.append({
            "type": "block",
            "chunk_idx": chunk_idx,
            "content": current_chunk.strip(),
            "char_count": len(current_chunk)
        })
    
    return chunks

# ============================================
# MAIN PROCESSING
# ============================================

def process_document(doc_path: str) -> List[Dict]:
    """Process a single document into optimized chunks."""
    with open(doc_path, 'r', encoding='utf-8') as f:
        doc = json.load(f)
    
    doc_id = doc.get("id", "")
    title = doc.get("title", f"Document {doc_id}")
    content = doc.get("content", "")
    
    # Skip if too short
    if not content or len(content) < 200:
        return []
    
    # Clean content
    content = clean_text(content)
    
    # Extract metadata
    doc_meta = extract_doc_metadata(doc)
    
    # Build enriched title
    if doc_meta["so_hieu"]:
        enriched_title = f"{title} ({doc_meta['so_hieu']})"
    else:
        enriched_title = title
    
    # Primary: Article-based chunking
    chunks = chunk_by_article(content, enriched_title, doc_meta)
    
    # Fallback: If too few articles, use semantic blocks
    if len(chunks) < 3:
        chunks = chunk_by_semantic_blocks(content, enriched_title)
    
    # Add document metadata to all chunks
    processed = []
    for i, chunk in enumerate(chunks):
        processed.append({
            "doc_id": doc_id,
            "doc_title": title,
            "doc_meta": doc_meta,
            "chunk_id": f"{doc_id}_{i}",
            **chunk
        })
    
    return processed

def process_all_documents():
    """Process all crawled documents with optimized chunking."""
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    all_chunks = []
    doc_files = glob(f"{DATA_SOURCE_DIR}/**/*.json", recursive=True)
    
    # Filter out checkpoint
    doc_files = [f for f in doc_files if "checkpoint" not in f]
    
    print(f"Processing {len(doc_files)} documents...")
    print(f"Strategy: Article-based chunking with context enrichment")
    
    for doc_path in doc_files:
        try:
            chunks = process_document(doc_path)
            all_chunks.extend(chunks)
            if chunks:
                print(f"  {os.path.basename(doc_path)}: {len(chunks)} chunks")
        except Exception as e:
            print(f"  Error: {doc_path}: {e}")
    
    # Save
    output_path = f"{PROCESSED_DIR}/chunks.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
    
    # Stats
    print(f"\n{'='*50}")
    print(f"Total chunks: {len(all_chunks)}")
    print(f"Saved to: {output_path}")
    
    if all_chunks:
        sizes = [c.get("char_count", 0) for c in all_chunks]
        avg_size = sum(sizes) / len(sizes)
        min_size = min(sizes)
        max_size = max(sizes)
        
        article_chunks = sum(1 for c in all_chunks if c.get("type") == "article")
        block_chunks = sum(1 for c in all_chunks if c.get("type") in ["block", "full_doc"])
        
        print(f"Average chunk size: {avg_size:.0f} chars")
        print(f"Min/Max: {min_size}/{max_size} chars")
        print(f"Article chunks: {article_chunks} ({100*article_chunks/len(all_chunks):.1f}%)")
        print(f"Block chunks: {block_chunks} ({100*block_chunks/len(all_chunks):.1f}%)")
    
    print(f"{'='*50}")
    
    return all_chunks

if __name__ == "__main__":
    process_all_documents()
