#!/usr/bin/env python3
"""
Crawler for thuvienphapluat.vn
Crawls high-priority Vietnamese legal documents.
"""
import os
import json
import time
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from urllib.parse import urljoin

# Config
BASE_URL = "https://thuvienphapluat.vn"
OUTPUT_DIR = "data/data_source_thuvien"
CHECKPOINT_FILE = "data/data_source_thuvien/checkpoint.json"
REQUEST_DELAY = 3  # Be respectful to server
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
}

# Document types (type IDs from site)
DOC_TYPES = {
    "bo_luat": {"type": "3", "name": "Bộ luật"},
    "luat": {"type": "1", "name": "Luật"},
    "nghi_dinh": {"type": "5", "name": "Nghị định"}
}

def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {"crawled_urls": []}

def save_checkpoint(checkpoint):
    os.makedirs(os.path.dirname(CHECKPOINT_FILE), exist_ok=True)
    with open(CHECKPOINT_FILE, 'w', encoding='utf-8') as f:
        json.dump(checkpoint, f, ensure_ascii=False, indent=2)

def get_document_list(doc_type_id: str, page: int = 1):
    """Get list of documents."""
    url = f"{BASE_URL}/page/tim-van-ban.aspx?keyword=&type={doc_type_id}&match=True&area=0&page={page}"
    
    try:
        response = requests.get(url, headers=HEADERS, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        doc_links = []
        
        # Find document links - adjust selectors based on actual HTML
        for link in soup.select('a[href*="/van-ban/"]'):
            href = link.get('href')
            title = link.get_text(strip=True)
            
            if href and len(title) > 10:
                full_url = urljoin(BASE_URL, href)
                doc_links.append({
                    'url': full_url,
                    'title': title
                })
        
        # Deduplicate
        seen = set()
        unique_links = []
        for item in doc_links:
            if item['url'] not in seen:
                seen.add(item['url'])
                unique_links.append(item)
        
        print(f"  Found {len(unique_links)} documents")
        return unique_links
        
    except Exception as e:
        print(f"  Error: {e}")
        return []

def get_document_content(url: str):
    """Fetch document content."""
    try:
        response = requests.get(url, headers=HEADERS, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        doc_data = {
            'url': url,
            'title': '',
            'info': {},
            'content': ''
        }
        
        # Title
        title_elem = soup.select_one('h1.bold')
        if not title_elem:
            title_elem = soup.select_one('h1')
        if title_elem:
            doc_data['title'] = title_elem.get_text(strip=True)
        
        # Info
        for row in soup.select('.tbl-property tr'):
            cells = row.find_all('td')
            if len(cells) >= 2:
                key = cells[0].get_text(strip=True).rstrip(':')
                value = cells[1].get_text(strip=True)
                if key and value:
                    doc_data['info'][key] = value
        
        # Content
        content_elem = soup.select_one('#tab-1') or soup.select_one('.content1') or soup.select_one('#content1')
        if content_elem:
            # Clean up
            for script in content_elem(["script", "style", "iframe"]):
                script.decompose()
            doc_data['content'] = content_elem.get_text(separator='\n', strip=True)
        
        return doc_data
        
    except Exception as e:
        print(f"  Error: {e}")
        return None

def crawl_document_type(doc_type_key: str, checkpoint: dict):
    """Crawl all documents of a type."""
    doc_config = DOC_TYPES[doc_type_key]
    type_id = doc_config["type"]
    type_name = doc_config["name"]
    
    print(f"\n{'='*50}")
    print(f"Crawling: {type_name}")
    print(f"{'='*50}\n")
    
    output_dir = os.path.join(OUTPUT_DIR, doc_type_key)
    os.makedirs(output_dir, exist_ok=True)
    
    crawled_urls = set(checkpoint.get('crawled_urls', []))
    page = 1
    total_docs = 0
    
    while page <= 50:  # Safety limit
        print(f"Page {page}...")
        doc_list = get_document_list(type_id, page)
        
        if not doc_list:
            print("No more documents")
            break
        
        new_docs = 0
        for doc_info in doc_list:
            url = doc_info['url']
            
            if url in crawled_urls:
                continue
            
            print(f"  {doc_info['title'][:60]}...")
            
            doc_data = get_document_content(url)
            
            if doc_data and doc_data['content']:
                # Save
                filename = os.path.join(output_dir, f"doc_{total_docs}.json")
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(doc_data, f, ensure_ascii=False, indent=2)
                
                print(f"    ✓ Saved ({len(doc_data['content'])} chars)")
                
                crawled_urls.add(url)
                total_docs += 1
                new_docs += 1
                
                checkpoint['crawled_urls'] = list(crawled_urls)
                save_checkpoint(checkpoint)
                
                time.sleep(REQUEST_DELAY)
        
        if new_docs == 0:
            break
        
        page += 1
    
    print(f"\nTotal: {total_docs}")
    return total_docs

def main():
    print(f"\n{'='*50}")
    print(f"Thuvienphapluat.vn Crawler")
    print(f"Started: {datetime.now()}")
    print(f"{'='*50}\n")
    
    checkpoint = load_checkpoint()
    total = 0
    
    for doc_type in DOC_TYPES:
        count = crawl_document_type(doc_type, checkpoint)
        total += count
    
    print(f"\n{'='*50}")
    print(f"Done: {datetime.now()}")
    print(f"Total: {total}")
    print(f"{'='*50}\n")

if __name__ == "__main__":
    main()
