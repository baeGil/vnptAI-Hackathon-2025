import sys
import os
import json
import pandas as pd
from datetime import datetime
from src.config import config
from src.utils import load_data, transform_choices, save_transformed_data
from src.utils import load_data, transform_choices, save_transformed_data, save_submission
from src.agent.graph import app
from src.client import RateLimitException

LOG_FILE = "submission_log.jsonl"
DETAIL_LOG_FILE = "inference_detail.log"
EMERGENCY_CSV = "submission_emergency.csv"
OUTPUT_FILE = "submission.csv"
TIME_FILE = "submission_time.csv"
TEST_FILE="test.json"

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

def save_time_csv(log_path: str, csv_path: str):
    """Read all log entries and create a request time CSV."""
    results = []
    if os.path.exists(log_path):
        with open(log_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    record = json.loads(line.strip())
                    # Handle both time keys (success vs error)
                    t = record.get('time')
                    if t is None:
                        t = record.get('time_taken', 0)
                        
                    results.append({
                        'qid': record.get('qid'),
                        'answer': record.get('answer', 'A'),
                        'time': t
                    })
                except:
                    pass
    if results:
        df = pd.DataFrame(results)
        df = df.sort_values(by='qid')
        # Ensure correct column order: qid, answer, time
        df = df[['qid', 'answer', 'time']]
        df.to_csv(csv_path, index=False)

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

    # Determine input file path - BTC requires /code/private_test.json
    # We check for mandatory path first, then fallback for local development
    mandatory_path = "/code/private_test.json"
    
    if os.path.isfile(mandatory_path):
        input_file = mandatory_path
        print(f"[INFO] Using mandatory input: {input_file}")
    else:
        input_file = os.path.join(config.DATA_DIR, TEST_FILE)
        # Fallbacks for testing
        if not os.path.exists(input_file):
            for fallback in ["private_test.json", "public_test.json", "test.json"]:
                fallback_path = os.path.join(config.DATA_DIR, fallback)
                if os.path.exists(fallback_path):
                    input_file = fallback_path
                    break
        print(f"[INFO] Using fallback input: {input_file}")
    
    if not os.path.exists(input_file):
        print(f"[ERROR] No input file found. Expected {mandatory_path} or file in {config.DATA_DIR}")
        return 1
    
    output_file = os.path.join(config.OUTPUT_DIR, OUTPUT_FILE)
    time_file = os.path.join(config.OUTPUT_DIR, TIME_FILE)
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
    # remaining_data = [item for item in transformed_data if item.get('qid') not in processed_qids] # This line is removed by the new logic
    
    total = len(transformed_data)
    done = len(processed_qids)
    # remaining_count = len(remaining_data) # This line is removed by the new logic
    
    print(f"Total: {total} | Done: {done}")
    
    # if remaining_count == 0: # This check is removed by the new logic
    #     print("All questions already processed!")
    #     consolidate_log_to_csv(log_file, output_file)
    #     print(f"Submission saved to {output_file}")
    #     return 0
    
    print(f"Processing questions...")
    append_detail_log(detail_log, f"=== Starting inference session (Auto: {args.auto}) ===")
    
    # Use index to manually control loop for retries
    idx = 0
    outputs = []
    time_outputs = []
    
    try:
        for i, item in enumerate(transformed_data): # Changed from remaining_data to transformed_data
            qid = item.get('qid', f'unknown_{i}')
            
            # Check for stop signal file
            if os.path.exists("STOP_AUTO"):
                print("\n[STOP] Found STOP_AUTO file. Stopping gracefully...")
                append_detail_log(detail_log, "Stopped by STOP_AUTO file.")
                break

            # Skip if already processed
            if qid in processed_qids:
                print(f"[SKIP] {qid} already processed.")
                continue

            question = item.get('question', '')
            choices = item.get('choices', [])
            
            initial_state = { # Renamed from inputs to initial_state for consistency with original code
                "question": question,
                "choices": choices,
                "qid": qid
            }
            
            print(f"[{i+1}/{total}] {qid} processing...", end=" ", flush=True)
            
            # Track timing for BTC requirement
            start_time = time.time()
            try:
                # Persistent Retry Logic
                MAX_RETRIES = 5
                RETRY_DELAY = 30  # seconds
                
                invoke_result = None
                
                for attempt in range(MAX_RETRIES + 1):
                    try:
                        # Invoke Agent
                        invoke_result = app.invoke(initial_state)
                        break  # Success
                    except RateLimitException as e:
                        if attempt < MAX_RETRIES:
                            print(f"\n⚠️ Rate Limit (Attempt {attempt + 1}/{MAX_RETRIES}) - Waiting {RETRY_DELAY}s...")
                            time.sleep(RETRY_DELAY)
                            continue
                        else:
                            # Propagate after max retries to hit the hourly wait logic
                            raise e

                final_state = invoke_result
                end_time = time.time()
                elapsed = end_time - start_time
                
                answer = final_state.get('answer', 'A')
                category = final_state.get('category', 'unknown')
                print(f"-> {answer} ({category}) [{elapsed:.2f}s]")
                
                # Log detail (This is our source of truth for all results)
                log_entry = {
                    "qid": qid,
                    "answer": answer,
                    "category": category,
                    "reasoning": final_state.get("reasoning", "")[:200],
                    "time": elapsed,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                append_to_log(log_file, log_entry)

            except RateLimitException as e:
                print(f"\n{'='*50}")
                print(f"[RATE LIMIT] Quota exceeded at {qid} after {MAX_RETRIES} internal retries.")
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
                print(f"Error at {qid}: {error_msg}")
                
                # Log with fallback answer 'C' but DO NOT save to inference_log.jsonl
                # This ensures we retry this question next time
                record = {
                    "qid": qid,
                    "answer": "C",
                    "category": "error",
                    "reasoning": error_msg[:200],
                    "time_taken": round(duration, 2)
                }
                # append_to_log(log_file, record)  <-- DISABLED per user request
                current = done + idx + 1
                print(f"[{current}/{total}] {qid} -> C (error - NOT SAVED) [{duration:.2f}s]")
                
                # Move to next despite error
                idx += 1

    except KeyboardInterrupt:
        print("\n\nKeyboardInterrupt Detected (Ctrl+C). Stopping gracefully...")
        append_detail_log(detail_log, "Stopped by KeyboardInterrupt.")
    
    # CONSOLIDATE RESULTS
    count = consolidate_log_to_csv(log_file, output_file)
    save_time_csv(log_file, time_file)
    print(f"\nDone/Stopped! {count} answers -> {output_file} & {time_file}")
    
    if os.path.exists("STOP_AUTO"):
        os.remove("STOP_AUTO")
        print("Removed STOP_AUTO file.")
        
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)