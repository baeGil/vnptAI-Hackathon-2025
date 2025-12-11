#!/usr/bin/env python3
"""
Compare ground truth answers with predicted answers.
Usage: python evaluate.py <ground_truth.json> <predictions.jsonl> <category_ground_truth.csv>
"""
import json
import sys
import csv

# ANSI color codes
GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"
TICK = f"{GREEN}✓{RESET}"
CROSS = f"{RED}✗{RESET}"

def load_ground_truth(filepath: str) -> dict:
    """Load ground truth from JSON file with 'answer' field."""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return {item['qid']: item.get('answer', '') for item in data}

def load_category_ground_truth(filepath: str) -> dict:
    """Load category ground truth from CSV file."""
    mapping = {}
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if 'qid' in row and 'category' in row:
                    mapping[row['qid']] = row['category'].strip()
    except Exception as e:
        print(f"Warning: Could not load categories from {filepath}: {e}")
    return mapping

def load_predictions(filepath: str) -> dict:
    """Load predictions from JSONL file."""
    predictions = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                record = json.loads(line.strip())
                predictions[record['qid']] = {
                    'answer': record.get('answer', ''),
                    'category': record.get('category', '')
                }
            except:
                pass
    return predictions

def evaluate(gt_path: str, pred_path: str, cat_path: str):
    """Compare ground truth with predictions."""
    ground_truth = load_ground_truth(gt_path)
    cat_ground_truth = load_category_ground_truth(cat_path)
    predictions = load_predictions(pred_path)
    
    # Answer stats
    correct = 0
    wrong = 0
    missing = 0

    # Category stats
    cat_correct = 0
    cat_wrong = 0
    cat_missing = 0
    
    print(f"\n{'='*50}")
    print(f"Comparing: {gt_path} vs {pred_path}")
    print(f"Categories: {cat_path}")
    print(f"{'='*50}\n")
    
    # Sort by qid
    all_qids = sorted(ground_truth.keys())
    
    for qid in all_qids:
        gt_answer = ground_truth[qid]
        gt_cat = cat_ground_truth.get(qid, "Unknown")
        
        pred_data = predictions.get(qid)
        pred_answer = pred_data.get('answer') if pred_data else None
        pred_cat = pred_data.get('category') if pred_data else None
        
        # Check Answer
        ans_str = ""
        if pred_answer is None:
            ans_str = f"GT={gt_answer} | Pred=? (missing)"
            missing += 1
        elif gt_answer.upper() == pred_answer.upper():
            ans_str = f"{TICK} Ans={gt_answer}"
            correct += 1
        else:
            ans_str = f"{CROSS} Ans: GT={gt_answer} | Pred={pred_answer}"
            wrong += 1
            
        # Check Category
        cat_str = ""
        if gt_cat != "Unknown":
            if pred_cat is None:
                cat_str = f" | Cat=? (missing)"
                cat_missing += 1
            elif gt_cat.lower() == pred_cat.lower():
                cat_str = f" | {TICK} Cat={gt_cat}"
                cat_correct += 1
            else:
                cat_str = f" | {CROSS} Cat: GT={gt_cat} | Pred={pred_cat}"
                cat_wrong += 1
        
        print(f"  {qid}: {ans_str}{cat_str}")
    
    total = correct + wrong + missing
    evaluated = correct + wrong
    accuracy = (correct / evaluated * 100) if evaluated > 0 else 0
    
    cat_evaluated = cat_correct + cat_wrong
    cat_accuracy = (cat_correct / cat_evaluated * 100) if cat_evaluated > 0 else 0
    
    print(f"\n{'='*50}")
    print(f"RESULTS")
    print(f"{'='*50}")
    print(f"Total questions: {total}")
    print("-" * 20)
    print("ANSWER ACCURACY:")
    print(f"Evaluated: {evaluated}")
    print(f"Missing: {missing}")
    print(f"{GREEN}Correct: {correct}{RESET}")
    print(f"{RED}Wrong: {wrong}{RESET}")
    print(f"Accuracy: {accuracy:.2f}%")
    print("-" * 20)
    print("CATEGORY ACCURACY:")
    print(f"Evaluated: {cat_evaluated}")
    print(f"Missing: {cat_missing}")
    print(f"{GREEN}Correct: {cat_correct}{RESET}")
    print(f"{RED}Wrong: {cat_wrong}{RESET}")
    print(f"Accuracy: {cat_accuracy:.2f}%")
    print(f"{'='*50}\n") 

# Thực hiện đánh giá:
gt_path = "./data/val.json"
pred_path = "./output/inference_log.jsonl"
cat_path = "./data/val_category.csv"
evaluate(gt_path, pred_path, cat_path)