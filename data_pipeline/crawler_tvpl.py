#!/usr/bin/env python3
"""
Production crawler for thuvienphapluat.vn using Selenium.
Refactored to Class-based structure for unified crawling.

Usage:
    # Run default
    - uv run data_pipeline/crawler_tvpl.py
    # Run specific type
    - uv run data_pipeline/crawler_tvpl.py --keywords "Thuế thu nhập" "Luật đất đai"
    # Limit number (test)
    - uv run data_pipeline/crawler_tvpl.py --limit 10
"""

import time
import json
import os
import re
import traceback
from datetime import datetime
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from urllib.parse import quote

# Configuration
BASE_URL = "https://thuvienphapluat.vn"
OUTPUT_DIR = "data/tvpl"
CHECKPOINT_FILE = f"{OUTPUT_DIR}/checkpoint.json"
REQUEST_DELAY = 2  # seconds between requests

# Document type IDs from TVPL /van-ban-moi page
# These are the actual filter IDs used by the site
DOC_TYPE_IDS = {
    "Hiến pháp": "14",
    "Luật": "17",
    "Bộ luật": "16", 
    "Pháp lệnh": "18",
    "Nghị định": "5",
    "Nghị quyết": "4",
    "Quyết định": "6",
    "Thông tư": "9",
    "Thông tư liên tịch": "10",
    "Chỉ thị": "2",
    "Văn bản hợp nhất": "19",
    "Văn bản mới": ""  # Empty = browse all new docs
}

# Priority document types to crawl (comprehensive coverage)
HIGH_PRIORITY_KEYWORDS = [
    "Văn bản mới",          # Get latest across all types first
    "Hiến pháp",
    "Bộ luật",             
    "Luật",                 
    "Pháp lệnh",
    "Nghị định",            # Very common
    "Nghị quyết",           # Administrative decisions (mergers, etc.)
    "Quyết định",           # Decisions
    "Thông tư",             # Circulars
    "Văn bản hợp nhất"      # Consolidated texts (important!)
]

class TVPLSeleniumCrawler:
    def __init__(self, headless=True):
        self.output_dir = OUTPUT_DIR
        self.checkpoint_file = CHECKPOINT_FILE
        self.driver = self._init_driver(headless)
        self.checkpoint = self._load_checkpoint()
        
        # Ensure base directories exist immediately to avoid monitor errors
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        (Path(self.output_dir) / "raw").mkdir(parents=True, exist_ok=True)
        
    def _init_driver(self, headless=True):
        """Initialize Chrome driver"""
        chrome_options = Options()
        if headless:
            chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36")
        
        # Try native Selenium Manager (cleaner and often more robust in newer versions)
        try:
            return webdriver.Chrome(options=chrome_options)
        except Exception as e:
            print(f"Native Selenium Manager failed: {e}")
            print("Falling back to webdriver_manager...")
            # Fallback to webdriver_manager
            try:
                from webdriver_manager.chrome import ChromeDriverManager
                service = Service(ChromeDriverManager().install())
                return webdriver.Chrome(service=service, options=chrome_options)
            except Exception as e2:
                print(f"Webdriver initialization failed: {e2}")
                raise e2

    def _load_checkpoint(self):
        """Load checkpoint from file"""
        if os.path.exists(self.checkpoint_file):
            with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {
            "crawled_ids": [],
            "last_keyword": None,
            "last_page": 1,
            "started_at": datetime.now().isoformat()
        }

    def _save_checkpoint(self):
        """Save checkpoint to file"""
        os.makedirs(self.output_dir, exist_ok=True)
        with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(self.checkpoint, f, ensure_ascii=False, indent=2)

    def browse_new_documents_by_type(self, doc_type="", page=1):
        """
        Browse documents from /van-ban-moi filtered by document type.
        This uses the actual site structure instead of broken search.
        
        Args:
            doc_type: Document type filter ID (e.g., "17" for Luật, "16" for Bộ luật)
            page: Page number for pagination
        
        Returns: (documents, has_next) tuple
        """
        try:
            # Build URL for /van-ban-moi with type filter
            if doc_type:
                url = f"{BASE_URL}/van-ban-moi?type={doc_type}&page={page}"
            else:
                url = f"{BASE_URL}/van-ban-moi?page={page}"
            
            print(f"  Browsing: {url}")
            self.driver.get(url)
            time.sleep(5)
            
            # Find document links in the main content area
            # These typically have specific patterns
            doc_links = []
            
            # Try to find document rows - they usually have specific CSS classes
            # Pattern: Links in the document list that point to /van-ban/...
            elements = self.driver.find_elements(By.CSS_SELECTOR, "a[href*='/van-ban/']")
            
            seen_urls = set()
            for elem in elements:
                try:
                    href = elem.get_attribute("href")
                    title = elem.text.strip()
                    
                    if not href or not title or href in seen_urls:
                        continue
                    
                    # Basic validation - must be actual document link
                    if '/van-ban/' not in href or '.aspx' not in href:
                        continue
                    
                    # Ignore navigation/download links
                    if 'Tải về' in title or 'Xem' in title or len(title) < 10:
                        continue
                    
                    seen_urls.add(href)
                    
                    # Extract document ID
                    doc_id = None
                    id_patterns = [r'-(\d{6,7})\.aspx', r'/(\d{6,7})\.aspx']
                    for pattern in id_patterns:
                        match = re.search(pattern, href)
                        if match:
                            doc_id = match.group(1)
                            break
                    
                    if doc_id and len(title) > 10:
                        doc_links.append({"id": doc_id, "title": title, "url": href})
                except Exception as e:
                    continue
            
            # Deduplicate by ID
            unique_docs = {d["id"]: d for d in doc_links}.values()
            unique_docs = list(unique_docs)
            
            # Check for next page
            has_next = False
            try:
                next_buttons = self.driver.find_elements(By.XPATH, "//*[contains(text(), 'Trang sau') or contains(text(), '›')]")
                has_next = len(next_buttons) > 0
            except:
                pass
            
            print(f"    Found {len(unique_docs)} documents")
            
            if len(unique_docs) == 0:
                self.driver.save_screenshot("debug_empty_page.png")
                
            return unique_docs, has_next
            
        except Exception as e:
            print(f"    Error browsing: {e}")
            return [], False

    def get_document_content(self, doc_url, doc_id):
        """Fetch full content of a document."""
        try:
            print(f"    Fetching: {doc_url[:80]}...")
            self.driver.get(doc_url)
            time.sleep(2)
            
            # 1. Get title
            title = ""
            try:
                title_elem = self.driver.find_element(By.TAG_NAME, "h1")
                title = title_elem.text.strip()
            except:
                pass
            
            # 2. Get metadata
            info = {}
            try:
                tables = self.driver.find_elements(By.TAG_NAME, "table")
                for table in tables:
                    rows = table.find_elements(By.TAG_NAME, "tr")
                    if 1 < len(rows) < 20:
                        for row in rows:
                            cells = row.find_elements(By.TAG_NAME, "td")
                            if len(cells) >= 2:
                                key = cells[0].text.strip().replace(':', '').strip()
                                value = cells[1].text.strip()
                                if key and value:
                                    info[key] = value
                        if info:
                            break
            except:
                pass
            
            # 3. Get main content
            content = ""
            content_selectors = [
                '#content', '.content', '.fulltext', '.document-content', 
                'article', '.main-content'
            ]
            
            for selector in content_selectors:
                try:
                    elem = self.driver.find_element(By.CSS_SELECTOR, selector)
                    content = elem.text.strip()
                    if len(content) > 200:
                        break
                except:
                    pass
            
            # Fallback
            if len(content) < 200:
                try:
                    body = self.driver.find_element(By.TAG_NAME, "body")
                    content = body.text.strip()
                except:
                    pass
            
            # Clean content
            content = re.sub(r'\n{3,}', '\n\n', content)
            content = re.sub(r'[ \t]+', ' ', content)
            
            if len(content) < 100:
                print(f"Short content ({len(content)} chars)")
            
            return {
                "id": doc_id,
                "title": title or f"Document {doc_id}",
                "url": doc_url,
                "metadata": info,
                "content": content,
                "content_length": len(content),
                "crawled_at": datetime.now().isoformat(),
                "source": "thuvienphapluat.vn"
            }
            
        except Exception as e:
            print(f"Error: {e}")
            return None

    def save_document(self, doc, keyword):
        """Save document to JSON file in data/tvpl/raw/{keyword_slug}"""
        folder_name = keyword.strip().replace(' ', '_').replace('"', '').replace("'", "")
        
        slug_map = {
            "Hiến pháp": "Hien_phap",
            "Bộ luật": "Bo_luat",
            "Luật": "Luat",
            "Nghị định": "Nghi_dinh",
            "Thông tư": "Thong_tu"
        }
        folder_name = slug_map.get(keyword, folder_name)
        
        output_path = Path(self.output_dir) / "raw" / folder_name
        output_path.mkdir(parents=True, exist_ok=True)
        
        filepath = output_path / f"{doc['id']}.json"
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(doc, f, ensure_ascii=False, indent=2)
        
        return str(filepath)

    def crawl_by_keyword(self, keyword, limit=None):
        """Crawl documents filtered by keyword"""
        print(f"\n{'='*60}")
        print(f"Crawling documents (filtering by: {keyword})")
        print(f"Crawling documents (type: {keyword})")
        print(f"{'='*60}\n")
        
        # Get document type ID for filtering
        doc_type_id = DOC_TYPE_IDS.get(keyword, "")
        
        page = 1
        total_crawled = 0
        
        # The checkpoint logic for resuming is removed as the instruction implies a full replacement of this block.
        # if self.checkpoint.get("last_keyword") == keyword:
        #     page = self.checkpoint.get("last_page", 1)
        #     print(f"Resuming from page {page}")
        
        while True:
            print(f"\n[Page {page}]")
            
            # Use browse instead of search
            documents, has_next = self.browse_new_documents_by_type(doc_type_id, page)
            
            if not documents:
                print("No documents found")
                break
            
            for doc_info in documents:
                doc_id = doc_info["id"]
                title = doc_info["title"]
                
                if doc_id in self.checkpoint["crawled_ids"]:
                    print(f"Skip {title[:50]}... (already crawled)")
                    continue
                
                # Relax filter: If we searched interactively, results are likely relevant
                # Only skip if it's clearly English
                if "tiếng anh" in title.lower() or "english" in title.lower():
                    print(f"Skip {title[:50]}... (English)")
                    continue
                
                # Loose check: verify if individual words appear, instead of exact phrase
                # Or just accept it. Let's just log if it doesn't match
                if keyword and keyword.lower() not in title.lower():
                    print(f"Note: Title '{title[:30]}...' might not contain '{keyword}' exactly")
                
                print(f"  📄 {title[:60]}...")
                
                doc = self.get_document_content(doc_info["url"], doc_id)
                
                if doc and doc.get("content") and len(doc["content"]) > 100:
                    self.save_document(doc, keyword)
                    print(f"Saved ({doc['content_length']} chars)")
                    
                    total_crawled += 1
                    self.checkpoint["crawled_ids"].append(doc_id)
                    self.checkpoint["last_keyword"] = keyword
                    self.checkpoint["last_page"] = page
                    self._save_checkpoint()
                    
                    if limit and total_crawled >= limit:
                        print(f"\n✓ Reached limit: {limit}")
                        return total_crawled
                else:
                    print(f"Failed or empty content")
                
                time.sleep(REQUEST_DELAY)
            
            if not has_next:
                print("No more pages")
                break
            
            page += 1
        
        print(f"\nTotal for '{keyword}': {total_crawled}")
        return total_crawled

    def run(self, keywords=None, limit=None):
        """
        Run the crawler for the specified keywords.
        :param keywords: List of keywords to crawl (optional)
        :param limit: Max documents PER KEYWORD (not global)
        """
        keywords_to_crawl = keywords if keywords else HIGH_PRIORITY_KEYWORDS
        
        print(f"Started at: {datetime.now()}")
        print(f"Keywords: {', '.join(keywords_to_crawl)}")
        print(f"Output: {OUTPUT_DIR}")
        print(f"Limit per keyword: {limit if limit else 'Unlimited'}")
        
        total_session = 0
        try:
            for keyword in keywords_to_crawl:
                try:
                    # Check driver limits or health
                    if not self.driver:
                        self.driver = self._init_driver()
                        
                    # Limit is applied per keyword now
                    count = self.crawl_by_keyword(keyword, limit)
                    total_session += count
                except Exception as e:
                    print(f"    ✗ Error crawling '{keyword}': {e}")
                    # Try to restart driver for next keyword
                    try:
                        self.driver.quit()
                    except: pass
                    self.driver = self._init_driver()
                    print("    ↻ Driver restarted.")
                
                # Small pause between keywords
                time.sleep(2)
                
        finally:
            self.close()
            print(f"\n{'='*60}")
            print(f"CRAWLING SESSION COMPLETE")
            print(f"Total documents crawled: {total_session}")
            print(f"{'='*60}\n")

    def close(self):
        """Close browser resources explicitly"""
        if hasattr(self, 'driver') and self.driver:
            try:
                self.driver.quit()
                print("TVPL Browser closed")
            except:
                pass
            self.driver = None

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Selenium crawler for thuvienphapluat.vn")
    parser.add_argument('--keywords', nargs='+', default=HIGH_PRIORITY_KEYWORDS,
                        help='Keywords to search for')
    parser.add_argument('--limit', type=int, default=None,
                        help='Maximum number of documents to crawl')
    parser.add_argument('--test', action='store_true',
                        help='Test mode: crawl only 10 documents')
    
    args = parser.parse_args()
    
    crawler = TVPLSeleniumCrawler(headless=True)
    try:
        if args.test:
            print("TEST MODE: Limiting to 10 documents")
            crawler.run(limit=10)
        else:
            crawler.run(keywords=args.keywords, limit=args.limit)
    finally:
        crawler.close()
