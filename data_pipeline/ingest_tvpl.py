#!/usr/bin/env python3
import os
import re
import json
import hashlib
from typing import List, Dict, Any
from glob import glob

# --- Configuration ---
INPUT_DIR = "project/data/tvpl/raw"
OUTPUT_DIR = "project/data/tvpl/processed"
OUTPUT_FILE = "chunks.jsonl"

# Chunking parameters for vnpt embedding (8k context)
MAX_CHUNK_SIZE = 5000  
MIN_CHUNK_SIZE = 1000 
CHUNK_OVERLAP = 300    

class LegalHybridChunker:
    """
    Structure-aware chunker for Vietnamese legal documents.
    Hybrid approach:
    1. Splits by Legal Hierarchy (Chương -> Điều).
    2. Fallback to Recursive Semantic Splitting for long articles or unstructured text.
    """
    
    def __init__(self, min_chunk_size=MIN_CHUNK_SIZE, max_chunk_size=MAX_CHUNK_SIZE, overlap=CHUNK_OVERLAP):
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.overlap = overlap
        
    def chunk(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        if not text:
            return []
            
        chunks = []
        base_meta = metadata or {}
        
        # 1. Split by "Điều X" (Article)
        # Regex to find "Điều \d+." or "Điều \d+:" or "Điều \d+ "
        # We capture the delimiter to keep it
        article_pattern = r"(?:^|\n)(Điều\s+\d+[\.:\s])"
        
        parts = re.split(article_pattern, text)
        
        # parts[0] is Preamble/Context
        preamble = parts[0].strip()
        if preamble:
            # Preamble might be long, check size
            if len(preamble) > self.max_chunk_size:
                chunks.extend(self._recursive_split_wrapper(preamble, {**base_meta, "section": "Preamble"}))
            else:
                chunks.append({
                    "content": preamble,
                    "metadata": {**base_meta, "section": "Preamble"}
                })
        
        # Iterate pairs: [delimiter, content]
        for i in range(1, len(parts), 2):
            header = parts[i].strip() # "Điều 1."
            content = parts[i+1].strip() # Content
            
            # Reconstruct article for context
            full_article = f"{header} {content}"
            
            # Extract Article Number for metadata
            article_num = re.sub(r"[^0-9]", "", header)
            meta = {
                **base_meta,
                "article_id": article_num,
                "section": header
            }
            
            # 2. Check size. If too big, use recursive splitter
            if len(full_article) > self.max_chunk_size:
                sub_chunks = self._recursive_split_wrapper(full_article, meta)
                chunks.extend(sub_chunks)
            else:
                chunks.append({
                    "content": full_article,
                    "metadata": meta
                })
                
        # Fallback: If no "Điều" was found (len parts == 1), treat whole text as one block
        if len(parts) == 1 and preamble:
            # We already handled preamble above, but if it was the ONLY thing, 
            # and it wasn't split by "Điều", it checks if it needs recursive splitting there.
            pass 
            
        return chunks

    def _recursive_split_wrapper(self, text: str, meta: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Wraps the recursive splitter to return dicts with metadata."""
        text_chunks = self.recursive_split(text, self.max_chunk_size, self.overlap)
        results = []
        for i, chunk_text in enumerate(text_chunks):
            # If sub-chunking, we might want to append index to metadata
            # but usually just sharing the metadata is fine.
            results.append({
                "content": chunk_text,
                "metadata": meta
            })
        return results

    def recursive_split(self, text: str, max_size: int, overlap: int) -> List[str]:
        """Recursively splits long text into smaller chunks. (Ported from ingest_wiki.py)"""
        if len(text) <= max_size:
            return [text] if len(text) >= self.min_chunk_size else []
        
        chunks = []
        
        # Try splitting by paragraphs
        paragraphs = text.split('\n\n')
        
        current_chunk = ""
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
                
            if len(current_chunk) + len(para) + 2 <= max_size:
                current_chunk += ("\n\n" + para if current_chunk else para)
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                
                # Split huge paragraph by newlines (single) or sentences
                if len(para) > max_size:
                    # Try breaking by single newline first which is common in bullet points
                    sub_lines = para.split('\n')
                    if len(sub_lines) > 1 and all(len(l) < max_size for l in sub_lines):
                         # Re-assemble sub-lines
                         sub_chunk = ""
                         for line in sub_lines:
                             if len(sub_chunk) + len(line) + 1 <= max_size:
                                 sub_chunk += ("\n" + line if sub_chunk else line)
                             else:
                                 if sub_chunk: chunks.append(sub_chunk)
                                 sub_chunk = line
                         if sub_chunk: chunks.append(sub_chunk)
                    else:
                        # Split by sentences
                        sentences = re.split(r'(?<=[.!?])\s+', para)
                        sentence_chunk = ""
                        for sent in sentences:
                            if len(sentence_chunk) + len(sent) + 1 <= max_size:
                                sentence_chunk += (" " + sent if sentence_chunk else sent)
                            else:
                                if sentence_chunk:
                                    chunks.append(sentence_chunk)
                                if len(sent) > max_size:
                                    # Hard split
                                    for k in range(0, len(sent), max_size - overlap):
                                        chunks.append(sent[k:k + max_size])
                                else:
                                    sentence_chunk = sent
                        if sentence_chunk:
                            chunks.append(sentence_chunk)
                    current_chunk = ""
                else:
                    current_chunk = para
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return [c for c in chunks if len(c) >= self.min_chunk_size]

def clean_content(text):
    """
    Clean raw text from TVPL:
    1. Remove header noise (navigation, ads)
    2. Remove footer noise (contact info, related links)
    3. Normalize whitespace
    """
    if not text:
        return ""
    
    # 1. Normalize line endings and whitespace
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join([l for l in lines if l])
    
    # 2. Heuristic to find start of content
    start_patterns = [
        r"CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM",
        r"Độc lập - Tự do - Hạnh phúc",
        r"(QUYẾT ĐỊNH|NGHỊ ĐỊNH|LUẬT|THÔNG TƯ|CHỈ THỊ|NGHỊ QUYẾT|KẾ HOẠCH)\s*$"
    ]
    
    start_idx = 0
    for pattern in start_patterns:
        match = re.search(pattern, text, re.MULTILINE)
        if match:
            if match.start() > start_idx: # Find the LAST occurrence? No, usually the first valid header.
                 # Actually, sometimes ads come before header.
                 # Let's trust the first match if it is reasonable
                 start_idx = match.start()
                 break
            
    if start_idx > 0:
        # Check if we have MUC LUC
        pass
        
    if "MỤC LỤC VĂN BẢN" in text:
        parts = text.split("MỤC LỤC VĂN BẢN", 1)
        # Usually content follows TOC
        # But sometimes TOC is at the end? rare.
        # Let's take the part after TOC if it exists and is substantial
        if len(parts) > 1:
            text = parts[1]
    
    # 3. Remove footer noise
    footer_markers = [
        r"Nơi nhận:",
        r"Văn bản liên quan",
        r"Ghi chú\s*Ý kiến\s*Facebook",
        r"Địa chỉ:.*TP\.HCM",
        r"Mã số thuế:",
        r"Trang chủ\s*Các Gói Dịch Vụ",
        r"Mọi vướng mắc, chưa rõ"
    ]
    
    for marker in footer_markers:
        match = re.search(marker, text, re.MULTILINE)
        if match:
            text = text[:match.start()]
            
    return text.strip()

def ingest_tvpl(input_dir, output_dir):
    chunker = LegalHybridChunker()
    all_chunks = []
    
    # Enable recursive search for JSON files
    search_path = os.path.join(input_dir, "**", "*.json")
    files = glob(search_path, recursive=True)
    
    print(f"Searching in: {input_dir}")
    print(f"Found {len(files)} files.")
    
    for i, file_path in enumerate(files):
        if i % 100 == 0:
            print(f"[{i}/{len(files)}] Processing...")
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                doc = json.load(f)
            
            # Basic validation
            if 'content' not in doc:
                continue
                
            raw_content = doc.get('content', '')
            cleaned_content = clean_content(raw_content)
            
            if not cleaned_content or len(cleaned_content) < 50:
                continue
            
            # Base metadata
            base_meta = {
                "doc_id": doc.get('id'),
                "doc_title": doc.get('title'),
                "doc_type": "legal",
                "domain": doc.get('domain', 'legal'),
                "url": doc.get('url'),
                "source": "tvpl"
            }
            
            doc_chunks = chunker.chunk(cleaned_content, metadata=base_meta)
            
            # Post-process to flatten structure for embedder
            for chunk in doc_chunks:
                # generate deterministic ID
                content_hash = hashlib.md5(chunk["content"].encode()).hexdigest()[:12]
                chunk_id = f"tvpl_{doc.get('id')}_{content_hash}"
                
                flat_chunk = {
                    "chunk_id": chunk_id,
                    "content": chunk["content"],
                    **chunk["metadata"] # flatten metadata
                }
                all_chunks.append(flat_chunk)
            
        except Exception as e:
            pass
            
    # Save as JSONL
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, OUTPUT_FILE)
    
    print(f"Saving {len(all_chunks)} chunks to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + '\n')
            
    print("Done!")


def main():
    # Resolve paths relative to project root if needed
    if os.path.exists(INPUT_DIR):
        input_path = INPUT_DIR
        output_path = OUTPUT_DIR
    elif os.path.exists(os.path.join("..", INPUT_DIR)):
        input_path = os.path.join("..", INPUT_DIR)
        output_path = os.path.join("..", OUTPUT_DIR)
    else:
        # Fallback to absolute paths if running from weird context
        input_path = "/Users/AI/vnptAI/project/data/tvpl/raw"
        output_path = "/Users/AI/vnptAI/project/data/tvpl/processed"

    ingest_tvpl(input_path, output_path)

if __name__ == "__main__":
    main()