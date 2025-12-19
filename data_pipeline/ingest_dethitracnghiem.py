#!/usr/bin/env python3
"""
Ingest script for Dethitracnghiem.vn questions data.
Converts raw JSONL questions into chunks suitable for vector database (RAG).

Strategy:
- Each question is a single chunk containing:
    - The question text
    - All options (A, B, C, D, ...)
    - The correct answer
- Metadata: uid, category, url

This keeps each Q&A self-contained for accurate retrieval.
"""
import json
import os
from pathlib import Path
from typing import List, Dict, Any
import hashlib
import argparse

# Configuration
DATA_DIR = Path("data/dethitracnghiem")
INPUT_FILE = DATA_DIR / "questions.jsonl"
OUTPUT_DIR = DATA_DIR / "processed"
OUTPUT_FILE = OUTPUT_DIR / "chunks.jsonl"


class QAChunker:
    """
    Simple chunker for Q&A data.
    Each question becomes one chunk with question + options + answer.
    """
    
    def format_chunk(self, question_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format a question record into a chunk for vector embedding.
        
        Format:
            <question text>
            
            A. <option A>
            B. <option B>
            C. <option C>
            D. <option D>
            
            Đáp án đúng:
            <letter>. <full content>
        """
        question = question_data.get("question", "")
        options = question_data.get("options", {})
        correct_answer = question_data.get("correct_answer", "")
        category = question_data.get("category", "")
        
        # Format options on separate lines
        options_lines = []
        for letter in sorted(options.keys()):
            options_lines.append(f"{letter}. {options[letter]}")
        options_text = "\n".join(options_lines)
        
        # Build content in user's requested format
        content = f"""{question}

{options_text}

Đáp án đúng:
{correct_answer}"""

        # Metadata for filtering and traceability
        metadata = {
            "uid": question_data.get("uid"),
            "category": category,
            "url": question_data.get("url"),
            "doc_type": "mcq",
            "source": "dethitracnghiem"
        }
        
        return {
            "content": content,
            "metadata": metadata
        }


def ingest_dethitracnghiem(input_file: Path, output_file: Path):
    """
    Main ingestion function.
    Reads questions.jsonl and outputs chunks.jsonl.
    """
    if not input_file.exists():
        print(f"❌ Input file not found: {input_file}")
        return
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    chunker = QAChunker()
    total_questions = 0
    chunks_written = 0
    errors = 0
    
    print(f"Reading from: {input_file}")
    print(f"Writing to: {output_file}")
    
    with open(input_file, 'r', encoding='utf-8') as fin, \
         open(output_file, 'w', encoding='utf-8') as fout:
        
        for line_num, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue
            
            total_questions += 1
            
            try:
                question_data = json.loads(line)
                chunk = chunker.format_chunk(question_data)
                
                # Validate content exists
                if not chunk.get("content"):
                    errors += 1
                    continue
                
                fout.write(json.dumps(chunk, ensure_ascii=False) + '\n')
                chunks_written += 1
                
                # Progress update every 10k
                if chunks_written % 10000 == 0:
                    print(f"Processed {chunks_written:,} chunks...")
                    
            except json.JSONDecodeError as e:
                print(f"Line {line_num}: JSON parse error - {e}")
                errors += 1
            except Exception as e:
                print(f"Line {line_num}: Error - {e}")
                errors += 1
    
    print(f"\n{'='*60}")
    print(f"Ingestion Complete")
    print(f"{'='*60}")
    print(f"Total Questions Read: {total_questions:,}")
    print(f"Chunks Written:       {chunks_written:,}")
    print(f"Errors:               {errors:,}")
    print(f"Output:               {output_file}")
    print(f"{'='*60}\n")
    
ingest_dethitracnghiem(INPUT_FILE, OUTPUT_FILE)