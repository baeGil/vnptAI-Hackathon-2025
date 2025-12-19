#!/usr/bin/env python3
"""
Crawler for dethitracnghiem.vn using Crawl4AI
Extracts MCQ questions from grades 1-12 (excluding Math subjects)

Output format matches existing questions.jsonl:
{uid, url, category, question, options: {A,B,C,D}, correct_answer}
"""

import asyncio
import json
import hashlib
import re
import os
from datetime import datetime
from typing import List, Dict, Optional, Set
from bs4 import BeautifulSoup
from crawl4ai import AsyncWebCrawler

# Configuration
BASE_URL = "https://dethitracnghiem.vn"
GRADES = [f"lop-{i}" for i in range(1, 13)]  # lop-1 to lop-12
EXCLUDED_SUBJECTS = ["toán", "toan", "toán học", "toan hoc", "math"]  # Exclude math
OUTPUT_FILE = "../data/dethitracnghiem/questions.jsonl"
CHECKPOINT_FILE = "../data/dethitracnghiem/crawl_checkpoint.json"
DELAY_BETWEEN_REQUESTS = 1.0  # seconds

# Get absolute paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(SCRIPT_DIR, OUTPUT_FILE)
CHECKPOINT_PATH = os.path.join(SCRIPT_DIR, CHECKPOINT_FILE)


def generate_uid(url: str, question: str) -> str:
    """Generate unique ID for a question."""
    content = f"{url}:{question}"
    return hashlib.md5(content.encode()).hexdigest()


def is_excluded_subject(subject_name: str) -> bool:
    """Check if subject should be excluded (Math)."""
    subject_lower = subject_name.lower()
    return any(excluded in subject_lower for excluded in EXCLUDED_SUBJECTS)


def load_checkpoint() -> Dict:
    """Load crawl checkpoint."""
    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {"crawled_urls": [], "last_updated": None}


def save_checkpoint(checkpoint: Dict):
    """Save crawl checkpoint."""
    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
    checkpoint["last_updated"] = datetime.now().isoformat()
    with open(CHECKPOINT_PATH, 'w', encoding='utf-8') as f:
        json.dump(checkpoint, f, ensure_ascii=False, indent=2)


def append_questions(questions: List[Dict]):
    """Append questions to output file."""
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, 'a', encoding='utf-8') as f:
        for q in questions:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")


def extract_subject_urls(html: str, base_url: str) -> List[Dict[str, str]]:
    """
    Extract subject URLs from a grade page.
    Returns list of {name, url} for non-math subjects.
    """
    soup = BeautifulSoup(html, 'html.parser')
    subjects = []
    
    # Look for subject links - typically in buttons or list items
    # Common patterns: <a> tags with subject names
    for link in soup.find_all('a', href=True):
        href = link.get('href', '')
        text = link.get_text(strip=True)
        
        # Filter: must be a valid subject link on the same domain
        if not href.startswith(base_url) and not href.startswith('/'):
            continue
            
        # Skip if it's a math subject
        if is_excluded_subject(text):
            continue
            
        # Skip navigation/footer links
        if any(skip in href for skip in ['gioi-thieu', 'lien-he', 'chinh-sach', 'dieu-khoan']):
            continue
            
        # Normalize URL
        if href.startswith('/'):
            href = BASE_URL + href
            
        # Only include if it looks like a subject/exam page
        if 'trac-nghiem' in href or any(subj in text.lower() for subj in 
            ['văn', 'anh', 'sử', 'địa', 'lý', 'hóa', 'sinh', 'công nghệ', 
             'tin học', 'gdcd', 'ktpl', 'gdqp', 'đạo đức', 'tự nhiên', 'xã hội']):
            subjects.append({"name": text, "url": href})
    
    # Remove duplicates
    seen = set()
    unique_subjects = []
    for s in subjects:
        if s['url'] not in seen:
            seen.add(s['url'])
            unique_subjects.append(s)
    
    return unique_subjects


def extract_exam_urls(html: str) -> List[str]:
    """
    Extract exam/test page URLs from a subject page.
    """
    soup = BeautifulSoup(html, 'html.parser')
    urls = []
    
    for link in soup.find_all('a', href=True):
        href = link.get('href', '')
        text = link.get_text(strip=True)
        
        # Look for exam links containing "trac-nghiem", "bai", "de"
        if 'trac-nghiem' in href and href not in urls:
            if href.startswith('/'):
                href = BASE_URL + href
            urls.append(href)
    
    return list(set(urls))


def extract_questions_from_html(html: str, url: str, category: str) -> List[Dict]:
    """
    Extract questions from an exam page using HTML parsing.
    
    Strategy:
    1. Find question containers (<p> or <div>)
    2. For each question, find options (A., B., C., D.)
    3. Detect correct answer by finding bold/strong text
    """
    soup = BeautifulSoup(html, 'html.parser')
    questions = []
    
    # Get the main content area
    content = soup.find('div', class_='entry-content') or soup.find('article') or soup.body
    if not content:
        return questions
    
    # Strategy 1: Look for structured question elements
    # Many sites use <p> tags with "Câu X:" pattern
    
    text_content = content.get_text()
    
    # Find all question patterns
    question_pattern = r'Câu\s*(\d+)[:.]\s*(.+?)(?=Câu\s*\d+[:.]\s*|$)'
    
    # First, let's try to find questions by looking at the HTML structure
    # Look for elements that contain "Câu" and options
    
    all_text = str(content)
    
    # Parse questions using regex on raw text first to get boundaries
    # Then use HTML to find which option is bold/correct
    
    # Get all paragraph-like elements
    paragraphs = content.find_all(['p', 'div', 'span'])
    
    current_question = None
    current_options = {}
    current_correct = None
    
    for elem in paragraphs:
        text = elem.get_text(strip=True)
        
        # Check if this is a new question
        q_match = re.match(r'^Câu\s*(\d+)[:.]\s*(.+)', text)
        if q_match:
            # Save previous question if exists
            if current_question and current_options:
                uid = generate_uid(url, current_question)
                questions.append({
                    "uid": uid,
                    "url": url,
                    "category": category,
                    "question": current_question,
                    "options": current_options,
                    "correct_answer": current_correct or ""
                })
            
            # Start new question
            current_question = q_match.group(2).strip()
            current_options = {}
            current_correct = None
            
            # Check if options are in the same element
            for opt in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']:
                opt_match = re.search(rf'\b{opt}[.)\s]+(.+?)(?=[A-H][.)\s]+|$)', text)
                if opt_match:
                    opt_text = opt_match.group(1).strip()
                    current_options[opt] = opt_text
                    
                    # Check if this option is bold (correct answer)
                    # Look for <strong>, <b>, or style with bold
                    bold_elems = elem.find_all(['strong', 'b'])
                    for bold in bold_elems:
                        if opt_text in bold.get_text() or f"{opt}." in bold.get_text():
                            current_correct = f"{opt}. {opt_text}"
            continue
        
        # Check for standalone options
        for opt in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']:
            opt_match = re.match(rf'^{opt}[.)\s]+(.+)', text)
            if opt_match:
                opt_text = opt_match.group(1).strip()
                current_options[opt] = opt_text
                
                # Check if bold
                if elem.find('strong') or elem.find('b') or elem.name in ['strong', 'b']:
                    current_correct = f"{opt}. {opt_text}"
                # Also check parent
                elif elem.parent and elem.parent.name in ['strong', 'b']:
                    current_correct = f"{opt}. {opt_text}"
    
    # Don't forget the last question
    if current_question and current_options:
        uid = generate_uid(url, current_question)
        questions.append({
            "uid": uid,
            "url": url,
            "category": category,
            "question": current_question,
            "options": current_options,
            "correct_answer": current_correct or ""
        })
    
    # Fallback: If no questions found with above method, try alternative parsing
    if not questions:
        questions = extract_questions_fallback(content, url, category)
    
    return questions


def extract_questions_fallback(content, url: str, category: str) -> List[Dict]:
    """
    Fallback extraction method for different page formats.
    Uses more aggressive text parsing.
    """
    questions = []
    
    # Get full text and try to parse
    full_text = content.get_text(separator='\n')
    
    # Split by "Câu X:" pattern
    parts = re.split(r'(Câu\s*\d+[:.]\s*)', full_text)
    
    for i in range(1, len(parts), 2):
        if i + 1 >= len(parts):
            break
            
        q_header = parts[i]  # "Câu X:"
        q_content = parts[i + 1]  # Question text and options
        
        # Extract question number
        q_num_match = re.search(r'\d+', q_header)
        if not q_num_match:
            continue
        
        # Split content into question and options
        lines = q_content.strip().split('\n')
        if not lines:
            continue
        
        # First part before options is the question
        question_text = ""
        options = {}
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check if it's an option
            opt_match = re.match(r'^([A-H])[.)\s]+(.+)', line)
            if opt_match:
                opt_letter = opt_match.group(1)
                opt_text = opt_match.group(2).strip()
                options[opt_letter] = opt_text
            elif not options:  # No options yet, still part of question
                question_text += " " + line
        
        if question_text and options:
            uid = generate_uid(url, question_text.strip())
            questions.append({
                "uid": uid,
                "url": url,
                "category": category,
                "question": question_text.strip(),
                "options": options,
                "correct_answer": ""  # Can't detect from fallback
            })
    
    return questions


async def crawl_page(crawler: AsyncWebCrawler, url: str) -> Optional[str]:
    """Crawl a single page and return HTML."""
    try:
        result = await crawler.arun(url=url)
        return result.html
    except Exception as e:
        print(f"Error crawling {url}: {e}")
        return None


async def discover_all_exam_urls(crawler: AsyncWebCrawler) -> List[Dict]:
    """
    Discover all exam URLs from all grades and subjects.
    Returns list of {url, category}
    """
    all_exams = []
    
    for grade in GRADES:
        grade_url = f"{BASE_URL}/{grade}"
        print(f"\n📚 Processing grade: {grade}")
        
        html = await crawl_page(crawler, grade_url)
        if not html:
            continue
        
        subjects = extract_subject_urls(html, BASE_URL)
        print(f"  Found {len(subjects)} non-math subjects")
        
        for subject in subjects:
            print(f"    📖 Subject: {subject['name']}")
            await asyncio.sleep(DELAY_BETWEEN_REQUESTS)
            
            subject_html = await crawl_page(crawler, subject['url'])
            if not subject_html:
                continue
            
            exam_urls = extract_exam_urls(subject_html)
            for exam_url in exam_urls:
                all_exams.append({
                    "url": exam_url,
                    "category": f"{grade.replace('-', ' ').title()} - {subject['name']}"
                })
            
            print(f"      Found {len(exam_urls)} exam pages")
    
    return all_exams


async def crawl_and_extract(exam_info: Dict, crawler: AsyncWebCrawler, 
                            checkpoint: Dict) -> List[Dict]:
    """Crawl an exam page and extract questions."""
    url = exam_info['url']
    category = exam_info['category']
    
    if url in checkpoint.get('crawled_urls', []):
        return []
    
    html = await crawl_page(crawler, url)
    if not html:
        return []
    
    questions = extract_questions_from_html(html, url, category)
    
    # Mark as crawled
    checkpoint.setdefault('crawled_urls', []).append(url)
    
    return questions


async def main():
    """Main crawler function."""
    print("=" * 60)
    print("Crawl4AI MCQ Crawler for dethitracnghiem.vn")
    print("=" * 60)
    
    # Load checkpoint
    checkpoint = load_checkpoint()
    crawled_count = len(checkpoint.get('crawled_urls', []))
    print(f"Resume checkpoint: {crawled_count} URLs already crawled")
    
    total_questions = 0
    
    async with AsyncWebCrawler() as crawler:
        # Phase 1: Discover all exam URLs
        print("\nPhase 1: Discovering exam URLs...")
        all_exams = await discover_all_exam_urls(crawler)
        print(f"\nFound {len(all_exams)} total exam pages")
        
        # Filter already crawled
        pending_exams = [e for e in all_exams if e['url'] not in checkpoint.get('crawled_urls', [])]
        print(f"{len(pending_exams)} pages remaining to crawl")
        
        # Phase 2: Crawl and extract questions
        print("\nPhase 2: Extracting questions...")
        
        for i, exam in enumerate(pending_exams):
            print(f"\r [{i+1}/{len(pending_exams)}] {exam['url'][:60]}...", end='', flush=True)
            
            await asyncio.sleep(DELAY_BETWEEN_REQUESTS)
            
            questions = await crawl_and_extract(exam, crawler, checkpoint)
            
            if questions:
                append_questions(questions)
                total_questions += len(questions)
                print(f" ✓ {len(questions)} questions")
            else:
                print(f" - No questions found")
            
            # Save checkpoint periodically
            if i % 10 == 0:
                save_checkpoint(checkpoint)
        
        # Final checkpoint save
        save_checkpoint(checkpoint)
    
    print("\n" + "=" * 60)
    print("Crawl complete!")
    print(f"Total new questions extracted: {total_questions}")
    print(f"Output file: {OUTPUT_PATH}")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())