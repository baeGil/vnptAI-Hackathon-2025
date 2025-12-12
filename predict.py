import sys
import os
import json
import pandas as pd
from datetime import datetime
from src.config import config
from src.utils import load_data, transform_choices, save_transformed_data
from src.agent.graph import app
from src.client import RateLimitException

LOG_FILE = "inference_log.jsonl"
DETAIL_LOG_FILE = "inference_detail.log"
EMERGENCY_CSV = "submission_emergency.csv"
TEST_FILE="test.json"
OUTPUT_FILE="submission_test.csv"

def load_processed_qids(log_path: str) -> set:
    """Load already processed question IDs from log file."""
    processed = set()
    if os.path.exists(log_path):
        with open(log_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    record = json.loads(line.strip())
                    processed.add(record.get('qid'))
                except:
                    pass
    return processed

def append_to_log(log_path: str, record: dict):
    """Append a processed record to the log file."""
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def append_detail_log(log_path: str, message: str):
    """Append detailed log message to file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(f"[{timestamp}] {message}\n")

def consolidate_log_to_csv(log_path: str, csv_path: str):
    """Read all log entries and create a submission CSV."""
    results = []
    if os.path.exists(log_path):
        with open(log_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    record = json.loads(line.strip())
                    results.append({
                        'qid': record.get('qid'),
                        'answer': record.get('answer', 'A')
                    })
                except:
                    pass
    if results:
        df = pd.DataFrame(results)
        # Sort by qid to ensure consistent order (1, 2, 3, 4...)
        # regardless of execution order
        df = df.sort_values(by='qid')
        df.to_csv(csv_path, index=False)
    return len(results)

import argparse
import time

def get_seconds_until_next_hour():
    """Calculate seconds to wait until the start of the next hour."""
    now = datetime.now()
    # Target next hour :00
    next_hour = now.replace(second=0, microsecond=0, minute=0).timestamp() + 3600
    current_time = now.timestamp()
    wait_seconds = next_hour - current_time
    return max(0, wait_seconds) + 5  # +5s buffer

def main():
    parser = argparse.ArgumentParser(description="Run inference with options.")
    parser.add_argument("--auto", action="store_true", help="Auto mode: waits for hourly quota reset on rate limit.")
    args = parser.parse_args()

    # Determine input file path
    input_file = os.path.join(config.DATA_DIR, TEST_FILE)
    
    # Fallbacks for testing
    if not os.path.exists(input_file):
        for fallback in ["public_test.json", "test.json", "private_test.json"]:
            fallback_path = os.path.join(config.DATA_DIR, fallback)
            if os.path.exists(fallback_path):
                input_file = fallback_path
                break
    
    if not os.path.exists(input_file):
        print(f"[ERROR] No input file found in {config.DATA_DIR}")
        return 1
    
    output_file = os.path.join(config.OUTPUT_DIR, OUTPUT_FILE)
    log_file = os.path.join(config.OUTPUT_DIR, LOG_FILE)
    detail_log = os.path.join(config.OUTPUT_DIR, DETAIL_LOG_FILE)
    emergency_csv = os.path.join(config.OUTPUT_DIR, EMERGENCY_CSV)
    
    # Ensure output directory exists
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    print(f"Loading data from {input_file}...")
    
    # Step 1: Load original data
    raw_data = load_data(input_file)
    
    # Step 2: Transform choices to "A. choice", "B. choice" format
    transformed_data = transform_choices(raw_data)
    
    # Step 3: Save transformed data
    save_transformed_data(transformed_data, input_file)
    
    # Load already processed questions for resume
    processed_qids = load_processed_qids(log_file)
    
    # Filter to only unprocessed questions
    remaining_data = [item for item in transformed_data if item.get('qid') not in processed_qids]
    
    total = len(transformed_data)
    done = len(processed_qids)
    remaining_count = len(remaining_data)
    
    print(f"Total: {total} | Done: {done} | Remaining: {remaining_count}")
    
    if remaining_count == 0:
        print("All questions already processed!")
        consolidate_log_to_csv(log_file, output_file)
        print(f"Submission saved to {output_file}")
        return 0
    
    print(f"Processing {remaining_count} questions...")
    append_detail_log(detail_log, f"=== Starting inference session (Auto: {args.auto}) ===")
    
    # Use index to manually control loop for retries
    idx = 0
    try:
        while idx < len(remaining_data):
            item = remaining_data[idx]
            qid = item.get("qid")
            question = item.get("question", "")
            choices = item.get("choices", [])
            
            # Check for stop signal file
            if os.path.exists("STOP_AUTO"):
                print("\n[STOP] Found STOP_AUTO file. Stopping gracefully...")
                append_detail_log(detail_log, "Stopped by STOP_AUTO file.")
                break
            
            # Initial State
            initial_state = {
                "question": question,
                "qid": qid,
                "choices": choices,
                "category": "",
                "context": "",
                "answer": "",
                "reasoning": ""
            }
            
            start_time = time.time()
            
            try:
                # Invoke Agent
                final_state = app.invoke(initial_state)
                
                # Calculate duration
                duration = time.time() - start_time
                
                record = {
                    "qid": qid,
                    "answer": final_state["answer"],
                    "category": final_state.get("category", ""),
                    "reasoning": final_state.get("reasoning", "")[:200],
                    "time_taken": round(duration, 2)  # Log duration
                }
                
                # Save immediately to log
                append_to_log(log_file, record)
                
                # Detailed log to file
                category = final_state.get('category', '')
                append_detail_log(detail_log, f"{qid}: {final_state['answer']} ({category}) - {duration:.2f}s")
                
                # Simple console output
                current = done + idx + 1
                print(f"[{current}/{total}] {qid} -> {final_state['answer']} ({category}) [{duration:.2f}s]")
                
                # Success, move to next
                idx += 1
                
            except RateLimitException as e:
                # RATE LIMIT HANDLING
                print(f"\n{'='*50}")
                print(f"[RATE LIMIT] Quota exceeded at {qid}")
                print(f"{'='*50}")
                
                append_detail_log(detail_log, f"RATE LIMIT at {qid}: {e}")
                
                if args.auto:
                    wait_seconds = get_seconds_until_next_hour()
                    wait_minutes = wait_seconds / 60
                    print(f"⚠️ Auto Mode: Waiting {wait_minutes:.1f} minutes until next hour quota reset...")
                    print(f"Resuming at {datetime.fromtimestamp(time.time() + wait_seconds).strftime('%H:%M:%S')}")
                    
                    time.sleep(wait_seconds)
                    print("♻️ Quota reset! Resuming...")
                    append_detail_log(detail_log, "Resuming after rate limit wait.")
                    # Do NOT increment idx, retry same question
                    continue
                else:
                    print("Manual mode: Exiting. Use --auto to wait automatically.")
                    break
                
            except Exception as e:
                duration = time.time() - start_time
                error_msg = str(e)
                append_detail_log(detail_log, f"ERROR at {qid}: {error_msg}")
                print(f"❌ Error at {qid}: {error_msg}")
                
                # Log with fallback answer 'C'
                record = {
                    "qid": qid,
                    "answer": "C",
                    "category": "error",
                    "reasoning": error_msg[:200],
                    "time_taken": round(duration, 2)
                }
                append_to_log(log_file, record)
                current = done + idx + 1
                print(f"[{current}/{total}] {qid} -> C (error) [{duration:.2f}s]")
                
                # Move to next despite error
                idx += 1

    except KeyboardInterrupt:
        print("\n\n⚠️ KeyboardInterrupt Detected (Ctrl+C). Stopping gracefully...")
        append_detail_log(detail_log, "Stopped by KeyboardInterrupt.")
    
    # CONSOLIDATE RESULTS
    count = consolidate_log_to_csv(log_file, output_file)
    print(f"\nDone/Stopped! {count} answers -> {output_file}")
    
    if os.path.exists("STOP_AUTO"):
        os.remove("STOP_AUTO")
        print("Removed STOP_AUTO file.")
        
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
