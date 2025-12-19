#!/usr/bin/env python3
"""
ViWiki2025 Data Ingestion Script
Processes Wikipedia dump format from ViWiki2025, chunks by sections.
"""

import os
import re
import json
import hashlib
from typing import List, Dict, Generator

# --- Configuration ---
INPUT_DIR = "data/ViWiki2025"
OUTPUT_DIR = "data/ViWiki2025/processed"
OUTPUT_FILE = "chunks.jsonl"

# Chunking parameters for vnpt embedding (8k context)
MAX_CHUNK_SIZE = 5000  
MIN_CHUNK_SIZE = 1000   
CHUNK_OVERLAP = 300    


def extract_articles(filepath: str) -> Generator[Dict[str, str], None, None]:
    """Extract articles from ViWiki2025 files (separated by = END OF PAGE =)."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split by END OF PAGE markers
    articles = content.split('= END OF PAGE =')
    
    for i, article_text in enumerate(articles):
        article_text = article_text.strip()
        if not article_text or len(article_text) < 50:
            continue
        
        # Extract title (first non-empty line)
        lines = article_text.split('\n')
        title = ""
        content_start = 0
        
        for idx, line in enumerate(lines):
            line = line.strip()
            if line and not line.startswith('<') and not line.startswith('__'):
                title = line
                content_start = idx + 1
                break
        
        if not title:
            title = f"Article_{i}"
        
        # Rest is content
        text = '\n'.join(lines[content_start:]).strip()
        
        if text:
            yield {
                'title': title,
                'text': text
            }

def clean_wiki_text(text: str) -> str:
    """Clean Wikipedia text artifacts."""
    # Remove template tags
    text = re.sub(r'<templatestyles[^>]*>', '', text)
    text = re.sub(r'__[A-Z]+__', '', text)
    
    # Remove URLs
    text = re.sub(r'https?://\S+', '', text)
    
    # Remove citation patterns
    text = re.sub(r'\[cần dẫn nguồn.*?\]', '', text, flags=re.IGNORECASE)
    
    # Normalize whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    
    return text.strip()


def split_by_sections(text: str) -> List[Dict[str, str]]:
    """Split text by MediaWiki section headers (== Section ==)."""
    sections = []
    
    # Split by == headers
    parts = re.split(r'(^==+\s+.+?\s+==+$)', text, flags=re.MULTILINE)
    
    current_header = "Giới thiệu"
    current_content = []
    
    for part in parts:
        part = part.strip()
        if not part:
            continue
        
        if part.startswith('==') and part.endswith('=='):
            # Save previous section
            if current_content:
                sections.append({
                    'header': current_header,
                    'content': '\n'.join(current_content).strip()
                })
            # Extract header text
            current_header = re.sub(r'^==+\s+|\s+==+$', '', part).strip()
            current_content = []
        else:
            current_content.append(part)
    
    # Last section
    if current_content:
        sections.append({
            'header': current_header,
            'content': '\n'.join(current_content).strip()
        })
    
    return sections

def recursive_split(text: str, max_size: int = MAX_CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Recursively split long text into chunks."""
    if len(text) <= max_size:
        return [text] if len(text) >= MIN_CHUNK_SIZE else []
    
    chunks = []
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
            
            # If single paragraph too long, split by sentences
            if len(para) > max_size:
                sentences = re.split(r'(?<=[.!?])\s+', para)
                sentence_chunk = ""
                for sent in sentences:
                    if len(sentence_chunk) + len(sent) + 1 <= max_size:
                        sentence_chunk += (" " + sent if sentence_chunk else sent)
                    else:
                        if sentence_chunk:
                            chunks.append(sentence_chunk)
                        # Hard split if single sentence too long
                        if len(sent) > max_size:
                            for j in range(0, len(sent), max_size - overlap):
                                chunks.append(sent[j:j + max_size])
                        else:
                            sentence_chunk = sent
                if sentence_chunk:
                    chunks.append(sentence_chunk)
                current_chunk = ""
            else:
                current_chunk = para
    
    if current_chunk:
        chunks.append(current_chunk)
    
    return [c for c in chunks if len(c) >= MIN_CHUNK_SIZE]


def generate_chunk_id(title: str, section: str, chunk_index: int) -> str:
    """Generate unique chunk ID."""
    content = f"viwiki2025:{title}:{section}:{chunk_index}"
    return hashlib.md5(content.encode('utf-8')).hexdigest()[:16]


def process_article(article: Dict[str, str]) -> Generator[Dict, None, None]:
    """Process a single Wikipedia article and yield chunks."""
    title = article['title']
    text = clean_wiki_text(article['text'])
    
    if not text:
        return
    
    # Split by sections
    sections = split_by_sections(text)
    
    chunk_index = 0
    for section in sections:
        header = section['header']
        content = section['content']
        
        if not content or len(content) < MIN_CHUNK_SIZE:
            continue
        
        # If section is short enough, use directly
        if len(content) <= MAX_CHUNK_SIZE:
            yield {
                'chunk_id': generate_chunk_id(title, header, chunk_index),
                'doc_id': hashlib.md5(title.encode('utf-8')).hexdigest()[:12],
                'doc_title': title,
                'section': header,
                'content': content,
                'chunk_index': chunk_index,
                'domain': 'viwiki2025'
            }
            chunk_index += 1
        else:
            # Recursively split long sections
            sub_chunks = recursive_split(content)
            for sub_chunk in sub_chunks:
                yield {
                    'chunk_id': generate_chunk_id(title, header, chunk_index),
                    'doc_id': hashlib.md5(title.encode('utf-8')).hexdigest()[:12],
                    'doc_title': title,
                    'section': header,
                    'content': sub_chunk,
                    'chunk_index': chunk_index,
                    'domain': 'viwiki2025'
                }
                chunk_index += 1


def main():
    print(f"\n{'='*60}")
    print(f"ViWiki2025 Ingestion Pipeline")
    print(f"{'='*60}\n")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_file = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    
    print(f"Input: {INPUT_DIR}")
    print(f"Output: {output_file}\n")
    
    # Get all subdirectories (AA, AB, ...)
    subdirs = [d for d in os.listdir(INPUT_DIR) 
               if os.path.isdir(os.path.join(INPUT_DIR, d)) and not d.startswith('.')]
    subdirs.sort()
    
    print(f"Found {len(subdirs)} subdirectories\n")
    
    total_chunks = 0
    total_files = 0
    
    with open(output_file, 'w', encoding='utf-8') as out_f:
        for subdir in subdirs:
            subdir_path = os.path.join(INPUT_DIR, subdir)
            wiki_files = [f for f in os.listdir(subdir_path) 
                         if f.startswith('wiki_') and not f.startswith('.')]
            
            print(f"Processing {subdir}/ ({len(wiki_files)} files)...")
            
            for filename in wiki_files:
                filepath = os.path.join(subdir_path, filename)
                file_chunks = 0
                
                try:
                    # Extract all articles from this file
                    for article in extract_articles(filepath):
                        # Process each article into chunks
                        for chunk in process_article(article):
                            out_f.write(json.dumps(chunk, ensure_ascii=False) + '\n')
                            file_chunks += 1
                            total_chunks += 1
                    
                    total_files += 1
                    if file_chunks > 0:
                        print(f"  ✓ {filename}: {file_chunks} chunks")
                        
                except Exception as e:
                    print(f"  ✗ Error processing {filename}: {e}")
    
    print(f"\n{'='*60}")
    print(f"Done!")
    print(f"Files: {total_files}")
    print(f"Chunks: {total_chunks}")
    print(f"Output: {output_file}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()