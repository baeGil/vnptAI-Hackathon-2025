#!/usr/bin/env python3
"""
Improved University Crawler for dethitracnghiem.vn - Version 3
Uses precise DOM selectors, JSONL output, and progress tracking.
Usage:
    uv run data_pipeline/crawler_university.py
"""
import requests
from bs4 import BeautifulSoup
import json
import re
import time
import hashlib
from pathlib import Path
from urllib.parse import urljoin
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

# Configuration
BASE_URL = 'https://dethitracnghiem.vn'
UNIVERSITY_URL = 'https://dethitracnghiem.vn/dai-hoc/'
DATA_DIR = Path('data/dethitracnghiem')
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_JSON = DATA_DIR / 'questions.jsonl'
FAILED_URLS_FILE = DATA_DIR / "failed_crawls.jsonl"
STATUS_FILE = Path('data/crawler_status.json')
MAX_WORKERS = 5
REQUEST_DELAY = 0.5
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}

class ImprovedCrawler:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        self.completed_subjects = set()
        self.seen_exam_urls = set()
        self.seen_question_uids = set()
        self.all_questions = []
        self.failed_urls = []
        
        self.progress_file = DATA_DIR / "crawler_progress.json"
        
        self.load_state()
        
    def _update_status(self, status, error=None):
        """Update status file for external monitoring"""
        data = {
            "status": status,
            "last_updated": datetime.now().isoformat(),
            "subjects_count": len(self.completed_subjects),
            "questions_count": len(self.all_questions)
        }
        if error:
            data["error"] = error
            
        try:
            with open(STATUS_FILE, 'w', encoding='utf-8') as f:
                json.dump(data, f)
        except Exception as e:
            print(f"Could not update status: {e}")

    def load_state(self):
        """Load existing questions from JSONL and progress"""
        # Load questions from JSONL
        if OUTPUT_JSON.exists():
            try:
                count = 0
                with open(OUTPUT_JSON, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                q = json.loads(line)
                                self.all_questions.append(q)
                                self.seen_question_uids.add(q.get('uid'))
                                count += 1
                            except json.JSONDecodeError:
                                continue
                print(f"Loaded {count} existing questions")
            except Exception as e:
                print(f"Could not load existing questions: {e}")
        
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'r', encoding='utf-8') as f:
                    self.completed_subjects = set(json.load(f))
                print(f"Resuming: {len(self.completed_subjects)} subjects completed")
            except Exception as e:
                print(f"Could not load progress: {e}")

    def save_progress(self, subject_url):
        """Save progress"""
        self.completed_subjects.add(subject_url)
        try:
            with open(self.progress_file, 'w', encoding='utf-8') as f:
                json.dump(list(self.completed_subjects), f, indent=2)
        except Exception as e:
            print(f"Error saving progress: {e}")
    
    def log_failed_url(self, url, url_type, reason, subject_name=None):
        """Log failed URL to JSONL file"""
        failed_record = {
            'timestamp': datetime.now().isoformat(),
            'url': url,
            'type': url_type,  # 'subject' or 'exam'
            'reason': reason,
            'subject': subject_name
        }
        self.failed_urls.append(failed_record)
        
        # Append to JSONL file
        try:
            with open(FAILED_URLS_FILE, 'a', encoding='utf-8') as f:
                f.write(json.dumps(failed_record, ensure_ascii=False) + '\n')
        except Exception as e:
            print(f"⚠ Error logging failed URL: {e}")
        
    def fetch_page(self, url, max_retries=3):
        """Fetch page with retry mechanism"""
        for attempt in range(max_retries):
            try:
                time.sleep(REQUEST_DELAY)
                response = self.session.get(url, timeout=30)
                response.raise_for_status()
                response.encoding = 'utf-8'
                return response.text
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2
                    time.sleep(wait_time)
                else:
                    return None

    def clean_text(self, text):
        if not text: return ""
        # Remove zero-width spaces and normalize whitespace
        text = text.replace('\u200b', '').replace('\xa0', ' ').strip()
        return re.sub(r'\s+', ' ', text)

    def generate_uid(self, url, question_text):
        """Generate deterministic UID"""
        content = f"{url}|{question_text}"
        return hashlib.md5(content.encode()).hexdigest()

    # STAGE 1: Get Subjects
    
    def get_subjects(self):
        """Get all university subjects"""
        print(f"fetching subjects from {UNIVERSITY_URL}...")
        html = self.fetch_page(UNIVERSITY_URL)
        if not html: return []
        
        soup = BeautifulSoup(html, 'html.parser')
        seen_urls = set()
        found_subjects = []
        
        # 1. Check sidebar nav menu (common structure)
        nav_menus = soup.select('ul.elementor-nav-menu li a')
        for link in nav_menus:
            href = link.get('href')
            text = link.get_text(strip=True)
            if href and text:
                full_url = urljoin(BASE_URL, href)
                if BASE_URL in full_url and full_url not in seen_urls:
                    seen_urls.add(full_url)
                    found_subjects.append({'name': text, 'url': full_url})
        
        # 2. Check main content grid (elementor-item)
        grid_links = soup.select('div.elementor-widget-container a.elementor-item')
        for link in grid_links:
             href = link.get('href')
             text = link.get_text(strip=True)
             if href and text:
                 full_url = urljoin(BASE_URL, href)
                 if BASE_URL in full_url and full_url not in seen_urls:
                     seen_urls.add(full_url)
                     found_subjects.append({'name': text, 'url': full_url})
        
        # 3. Fallback: Check for post cards
        post_cards = soup.select('article.elementor-post h3.elementor-post__title a')
        for link in post_cards:
             href = link.get('href')
             text = link.get_text(strip=True)
             if href and text:
                 full_url = urljoin(BASE_URL, href)
                 if BASE_URL in full_url and full_url not in seen_urls:
                     seen_urls.add(full_url)
                     found_subjects.append({'name': text, 'url': full_url})
        
        print(f"Found {len(found_subjects)} subject URLs from navigation\n")
        return found_subjects
    
    # STAGE 2: Get Exams
    
    def get_exams_from_subject(self, subject):
        """Get proper exam URLs (pagination support)"""
        subject_url = subject['url']
        subject_name = subject['name']
        
        all_exams = []
        current_url = subject_url
        page_num = 1
        max_pages = 50  # Safety limit
        
        while current_url and page_num <= max_pages:
            html = self.fetch_page(current_url)
            if not html:
                break
                
            soup = BeautifulSoup(html, 'html.parser')
            
            # Select articles that look like exams
            articles = soup.select('article.elementor-post')
            page_exams = []
            
            for article in articles:
                link = article.select_one('h3.elementor-post__title a')
                if link:
                    href = link.get('href')
                    title = link.get_text(strip=True)
                    if href:
                        full_url = urljoin(BASE_URL, href)
                        page_exams.append({'title': title, 'url': full_url})
            
            all_exams.extend(page_exams)
            
            # Check for next page
            next_link = soup.select_one('a.page-numbers.next')
            if next_link and next_link.get('href'):
                current_url = next_link.get('href')
                page_num += 1
            else:
                break
        
        if len(all_exams) == 0:
            self.log_failed_url(
                subject_url, 
                'subject', 
                'No exams found',
                subject_name
            )
            
        return all_exams

    # STAGE 3: Parse Questions

    def parse_exam_questions(self, exam):
        """Extract questions from exam page"""
        exam_url = exam['url']
        html = self.fetch_page(exam_url)
        if not html: return []
        
        soup = BeautifulSoup(html, 'html.parser')
        
        # Locate content container
        content = soup.find('div', class_='elementor-widget-theme-post-content')
        if not content:
            content = soup.find('div', class_='entry-content')
        
        container = content if content else soup.find('body')
        if not container:
            return []

        # Strategy: Iterate all block tags (p, li)
        # Identify a block as a question if it contains option markers (A. ... B. ...)
        
        candidates = container.find_all(['p', 'li'])
        processed_texts = set()
        questions = []
        
        for block in candidates:
            # Use separator=' ' to prevent text glueing
            text = block.get_text(separator=' ', strip=True)
            text = re.sub(r'\s+', ' ', text)
            
            if len(text) < 10 or text in processed_texts:
                continue
            
            # Structural Heuristic: Must contain "A." AND "B." 
            # (Relaxed: allow lowercase or missing dot if structure is clear, but let's strict first)
            has_options = (
                ('A.' in text and 'B.' in text) or
                ('a.' in text and 'b.' in text) or
                (re.search(r'\bA\b[\.\)]', text) and re.search(r'\bB\b[\.\)]', text))
            )
            
            if has_options:
                question_data = self.parse_question_text(block, exam_url, exam['title'])
                if question_data:
                    questions.append(question_data)
                    processed_texts.add(text) 
                    
        return questions

    def parse_question_text(self, block, exam_url, category):
        """Parse text block into question dict"""
        raw_text = block.get_text(separator=' ', strip=True)
        raw_text = re.sub(r'\s+', ' ', raw_text)
        
        # 1. Extract Question Text (before first option)
        # Look for first occurrence of A. or a. or A)
        split_match = re.search(r'(?=\s+[A-Da-d][\.\)])', raw_text)
        if not split_match:
             # Try stricter if space missing
             split_match = re.search(r'(?=[A-Da-d][\.\)])', raw_text)
             
        if not split_match:
            return None
            
        question_text = raw_text[:split_match.start()].strip()
        remaining_text = raw_text[split_match.start():]
        
        # Clean question number prefix "Câu 1:" or "1."
        question_text = re.sub(r'^(Câu\s+\d+[:\.]?|\d+[\.:])\s*', '', question_text, flags=re.I).strip()
        
        # 2. Extract Options
        options = {}
        # Pattern: Letter + dot/paren + content + lookahead for next letter or end
        # Matches: A. Text... B. Text...
        pattern = r'([A-D])[\.\)]\s*(.+?)(?=\s+[A-D][\.\)]|$)'
        
        for match in re.finditer(pattern, remaining_text, re.I | re.DOTALL):
            letter = match.group(1).upper()
            content = match.group(2).strip()
            options[letter] = content
            
        if len(options) < 2:
            return None
            
        # 3. Extract Correct Answer
        # Strategy: HTML inspection for bold/strong/color within the block
        correct_answer_letter = self._find_correct_answer(block, options)
        
        correct_answer_text = options.get(correct_answer_letter, "")
        correct_answer = f"{correct_answer_letter}. {correct_answer_text}"
        
        uid = self.generate_uid(exam_url, question_text)
        
        # Deduplication check
        if uid in self.seen_question_uids:
            return None
        
        self.seen_question_uids.add(uid)
        
        return {
            'uid': uid,
            'url': exam_url,
            'category': category,
            'question': question_text,
            'options': options,
            'correct_answer': correct_answer
        }
    
    def _find_correct_answer(self, elem, options):
        """Find correct answer from HTML element"""
        if not elem:
            return 'A'  # Default
        
        bold_elements = elem.find_all(['strong', 'b', 'span']) # Added span for color
        if not bold_elements:
            return 'A'
        
        bold_texts = [self.clean_text(b.get_text()) for b in bold_elements]
        
        # Strategy 1: Look for exact letter match in bold (e.g. "A.")
        for letter in ['A', 'B', 'C', 'D']:
            for bold in bold_texts:
                # FIX: Check for bold letter prefix
                if re.match(rf'^{letter}[\.\)]?\s*', bold, re.I):
                    return letter
        
        # Strategy 2: Look for option text in bold
        for letter, text in options.items():
            if not text: continue
            
            # If significant part of option text is bold
            for bold in bold_texts:
                if len(bold) > 5 and (bold in text or text in bold):
                    return letter
                    
        # Strategy 3: Check for red color/style if present (simple check)
        for child in elem.find_all(style=True):
            style = child.get('style', '').lower()
            text = child.get_text(strip=True)
            if 'color' in style and ('red' in style or '#f' in style): # simplistic red check
                # Check if this colored text matches an option
                for letter, opt_text in options.items():
                     if text in opt_text:
                         return letter

        return 'A'
    
    # Orchestration
    
    def crawl_subject(self, subject):
        """Crawl one subject"""
        if subject['url'] in self.completed_subjects:
            return 0
        
        # Get exams
        exams = self.get_exams_from_subject(subject)
        
        if not exams:
            self.save_progress(subject['url'])
            return 0
        
        # Parse questions from each exam
        subject_questions = []
        
        for exam in exams:
            if exam['url'] in self.seen_exam_urls:
                continue
            
            questions = self.parse_exam_questions(exam)
            
            if questions:
                subject_questions.extend(questions)
            else:
                # Log exam with 0 questions
                self.log_failed_url(
                    exam['url'],
                    'exam',
                    'No questions extracted (questions=0)',
                    subject['name']
                )
            
            self.seen_exam_urls.add(exam['url'])
        
        # Save results
        if subject_questions:
            self.all_questions.extend(subject_questions)
            self.save_results(silent=True)
        
        self.save_progress(subject['url'])
        return len(subject_questions)
    
    def save_results(self, silent=False):
        """Save all questions to JSONL (append mode)"""
        try:
            # Write entire list as JSONL (overwrite file with current state)
            # Since we keep all_questions in memory, overwriting is consistent
            with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
                for question in self.all_questions:
                    f.write(json.dumps(question, ensure_ascii=False) + '\n')
            
            if not silent:
                print(f"Saved {len(self.all_questions)} questions to {OUTPUT_JSON}")
        except Exception as e:
            print(f"Error saving: {e}")
    
    def run(self):
        """Main entry point"""
        print(f"\n{'='*70}")
        self._update_status("running")
        
        try:
            # Stage 1: Get subjects
            subjects = self.get_subjects()
            
            if not subjects:
                print("No subjects found!")
                self._update_status("error", "No subjects found")
                return
            
            total_subjects = len(subjects)
            
            # Filter completed
            subjects_to_crawl = [s for s in subjects if s['url'] not in self.completed_subjects]
            initial_done = total_subjects - len(subjects_to_crawl)
            
            if not subjects_to_crawl:
                print(f"All subjects completed ({total_subjects}/{total_subjects})")
                self._update_status("completed")
                return
            
            print(f"Starting crawl: {len(subjects_to_crawl)} subjects remaining of {total_subjects} total\n")
            
            # Crawl subjects (multi-threaded)
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = {
                    executor.submit(self.crawl_subject, subj): subj 
                    for subj in subjects_to_crawl
                }
                
                completed_count = initial_done
                
                for future in as_completed(futures):
                    completed_count += 1
                    try:
                        future.result()
                        # Real-time progress update
                        progress_pct = (completed_count / total_subjects) * 100
                        print(f"\rProgress: {completed_count}/{total_subjects} subjects ({progress_pct:.1f}%) | Questions: {len(self.all_questions)}", end='', flush=True)
                        self._update_status("running")
                    except Exception as e:
                        subj = futures[future]
                        print(f"\nError: {subj['name']}: {e}")
            
            print() # Newline after progress
            
            # Final save
            self.save_results()
            
            print(f"\n{'='*70}")
            print(f"CRAWLING COMPLETE")
            print(f"{'='*70}")
            print(f"Total Subjects: {total_subjects}")
            print(f"Total Questions: {len(self.all_questions)}")
            print(f"Output: {OUTPUT_JSON}")
            print(f"{'='*70}\n")
            
            self._update_status("completed")
            
        except KeyboardInterrupt:
            print("\n\nStopped by user")
            self.save_results()
            self._update_status("stopped")
        except Exception as e:
            print(f"\nError: {e}")
            self._update_status("error", str(e))

if __name__ == "__main__":
    crawler = ImprovedCrawler()
    crawler.run()