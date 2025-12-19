#!/usr/bin/env python3
"""
Wiki Data Ingestion Script
Processes raw Wikipedia text files, cleans them, and chunks them for embedding.
"""

import os
import re
import json
import hashlib
from typing import List, Dict, Generator

# config
INPUT_DIR = "project/data/wiki/raw"
OUTPUT_DIR = "project/data/wiki/processed"
OUTPUT_FILE = "chunks.jsonl"

# Chunking parameters for vnpt embedding (8k context)
MAX_CHUNK_SIZE = 5000  
MIN_CHUNK_SIZE = 1000  
CHUNK_OVERLAP = 300    

def clean_text(text: str) -> str:
    """Cleans residual Wikipedia scraping artifacts from text."""
    
    # Remove leading pipe characters from lines (table remnants)
    text = re.sub(r'^\s*\|\s*', '', text, flags=re.MULTILINE)
    
    # Remove standalone dashes/pipes at start of lines
    text = re.sub(r'^[-|]\s*$', '', text, flags=re.MULTILINE)
    
    # Remove Wikipedia infobox table patterns (Key | Value |)
    text = re.sub(r'^[^|]*\|[^|]*\|\s*$', '', text, flags=re.MULTILINE)
    
    # Remove citation-like patterns [cần dẫn nguồn], [sửa], etc.
    text = re.sub(r'\\\[.*?\]', '', text)
    text = re.sub(r'\[cần dẫn nguồn.*?\]', '', text, flags=re.IGNORECASE)
    
    # Remove URLs
    text = re.sub(r'https?://\S+', '', text)
    
    # Remove excess whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    
    # Remove lines that are only punctuation or special chars
    text = re.sub(r'^[\-\|\*\#\=\:\;\.\,]+$', '', text, flags=re.MULTILINE)
    
    return text.strip()

def extract_title_from_filename(filename: str) -> str:
    """Extracts the article title from a Wikipedia filename."""
    # Pattern: "Title – Wikipedia tiếng Việt.txt"
    title = filename.replace('.txt', '')
    title = re.sub(r'\s*–\s*Wikipedia tiếng Việt$', '', title)
    return title.strip()


def split_by_headers(text: str) -> List[Dict[str, str]]:
    """Splits text by markdown H2 headers (##)."""
    sections = []
    
    # Split by ## headers, keeping the header with its content
    parts = re.split(r'(^##\s+.+$)', text, flags=re.MULTILINE)
    
    current_header = "Giới thiệu"  # Default for content before first header
    current_content = []
    
    for part in parts:
        part = part.strip()
        if not part:
            continue
            
        if part.startswith('## '):
            # Save previous section if exists
            if current_content:
                sections.append({
                    'header': current_header,
                    'content': '\n'.join(current_content).strip()
                })
            current_header = part.replace('## ', '').strip()
            current_content = []
        else:
            current_content.append(part)
    
    # Don't forget the last section
    if current_content:
        sections.append({
            'header': current_header,
            'content': '\n'.join(current_content).strip()
        })
    
    return sections

def recursive_split(text: str, max_size: int = MAX_CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Recursively splits long text into smaller chunks."""
    if len(text) <= max_size:
        return [text] if len(text) >= MIN_CHUNK_SIZE else []
    
    chunks = []
    
    # Try splitting by paragraphs first
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
            
            # If a single paragraph is too long, split by sentences
            if len(para) > max_size:
                sentences = re.split(r'(?<=[.!?])\s+', para)
                sentence_chunk = ""
                for sent in sentences:
                    if len(sentence_chunk) + len(sent) + 1 <= max_size:
                        sentence_chunk += (" " + sent if sentence_chunk else sent)
                    else:
                        if sentence_chunk:
                            chunks.append(sentence_chunk)
                        # If a single sentence is still too long, truncate with overlap
                        if len(sent) > max_size:
                            # Hard split
                            for i in range(0, len(sent), max_size - overlap):
                                chunks.append(sent[i:i + max_size])
                        else:
                            sentence_chunk = sent
                if sentence_chunk:
                    chunks.append(sentence_chunk)
                current_chunk = ""
            else:
                current_chunk = para
    
    if current_chunk:
        chunks.append(current_chunk)
    
    # Filter out chunks that are too small
    return [c for c in chunks if len(c) >= MIN_CHUNK_SIZE]

def generate_chunk_id(source_file: str, section: str, chunk_index: int) -> str:
    """Generates a unique ID for a chunk."""
    content = f"{source_file}:{section}:{chunk_index}"
    return hashlib.md5(content.encode('utf-8')).hexdigest()[:16]


def process_file(filepath: str) -> Generator[Dict, None, None]:
    """Processes a single file and yields chunks."""
    filename = os.path.basename(filepath)
    title = extract_title_from_filename(filename)
    
    with open(filepath, 'r', encoding='utf-8') as f:
        raw_text = f.read()
    
    # Clean the text
    cleaned_text = clean_text(raw_text)
    
    if not cleaned_text:
        return
    
    # Split by headers
    sections = split_by_headers(cleaned_text)
    
    chunk_index = 0
    for section in sections:
        header = section['header']
        content = section['content']
        
        if not content or len(content) < MIN_CHUNK_SIZE:
            continue
        
        # If section content is short enough, use it directly
        if len(content) <= MAX_CHUNK_SIZE:
            yield {
                'id': generate_chunk_id(filename, header, chunk_index),
                'source_file': filename,
                'title': title,
                'section': header,
                'chunk_text': content,
                'chunk_index': chunk_index
            }
            chunk_index += 1
        else:
            # Recursively split long sections
            sub_chunks = recursive_split(content)
            for sub_chunk in sub_chunks:
                # Prepend section header for context
                contextualized_chunk = f"## {header}\n\n{sub_chunk}"
                yield {
                    'id': generate_chunk_id(filename, header, chunk_index),
                    'source_file': filename,
                    'title': title,
                    'section': header,
                    'chunk_text': contextualized_chunk,
                    'chunk_index': chunk_index
                }
                chunk_index += 1

def main():
    # Resolve paths
    if os.path.exists(INPUT_DIR):
        input_path = INPUT_DIR
        output_path = OUTPUT_DIR
    elif os.path.exists(os.path.join("..", INPUT_DIR)):
        input_path = os.path.join("..", INPUT_DIR)
        output_path = os.path.join("..", OUTPUT_DIR)
    else:
        # Fallback, direct path from my local machine
        input_path = "/Users/AI/vnptAI/project/data/wiki/raw"
        output_path = "/Users/AI/vnptAI/project/data/wiki/processed"
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, OUTPUT_FILE)
    
    print(f"Reading files from: {input_path}")
    print(f"Writing output to: {output_file}")
    
    # Get all txt files
    txt_files = [f for f in os.listdir(input_path) if f.endswith('.txt')]
    print(f"Found {len(txt_files)} files to process.")
    
    total_chunks = 0
    
    with open(output_file, 'w', encoding='utf-8') as out_f:
        for i, filename in enumerate(txt_files):
            filepath = os.path.join(input_path, filename)
            file_chunks = 0
            
            try:
                for chunk in process_file(filepath):
                    out_f.write(json.dumps(chunk, ensure_ascii=False) + '\n')
                    file_chunks += 1
                    total_chunks += 1
                
                print(f"[{i+1}/{len(txt_files)}] {filename}: {file_chunks} chunks")
                
            except Exception as e:
                print(f"[{i+1}/{len(txt_files)}] Error processing {filename}: {e}")
    
    print(f"\nDone! Total chunks: {total_chunks}")
    print(f"Output saved to: {output_file}")

if __name__ == "__main__":
    main()