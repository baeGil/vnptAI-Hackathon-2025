import pandas as pd
import json
import os

### Load input data 
def load_data(file_path: str) -> list:
    """Load data from JSON or CSV file."""
    if file_path.endswith('.json'):
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    elif file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
        # If column choices is a string representation of a list, parse it
        if 'choices' in df.columns and isinstance(df['choices'].iloc[0], str):
            df['choices'] = df['choices'].apply(eval)
        return df.to_dict(orient='records')
    return []

### Transform input data 
def transform_choices(data: list) -> list:
    """
    Transform choices to include letter prefixes (A., B., C., ...).
    Example:
    Input:  {"choices": ["Môi trường", "Kinh tế", "Văn hóa", "Quốc phòng an ninh"]}
    Output: {"choices": ["A. Môi trường", "B. Kinh tế", "C. Văn hóa", "D. Quốc phòng an ninh"]}
    """
    transformed = []
    for item in data:
        qid = item.get('qid', item.get('id', ''))
        question = item.get('question', '')
        choices = item.get('choices', [])
        
        # Map choices to include letter prefixes (A., B., C., ...)
        mapped_choices = []
        for i, choice in enumerate(choices):
            letter = chr(ord('A') + i)
            # Check if the answer already starts with a Latin letter
            if not choice.strip().startswith(f"{letter}.") and not choice.strip().startswith(f"{letter} "):
                mapped_choices.append(f"{letter}. {choice}")
            else:
                mapped_choices.append(choice)
        
        transformed.append({
            'qid': qid,
            'question': question,
            'choices': mapped_choices
        })
    
    return transformed

### Save transformed data to a new JSON file
def save_transformed_data(data: list, original_path: str) -> str:
    """
    Save transformed data to a new JSON file.
    Returns the path to the new file.
    """
    # Create new file name
    base, ext = os.path.splitext(original_path)
    new_path = f"{base}_transformed.json"
    
    # Save as json
    with open(new_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"[Utils] Saved transformed data to: {new_path}")
    
    return new_path

### Load transformed data
def load_transformed_data(file_path: str) -> list:
    """Load transformed data from CSV (choices are JSON strings)."""
    df = pd.read_csv(file_path, encoding='utf-8')
    df['choices'] = df['choices'].apply(json.loads)
    return df.to_dict(orient='records')

### Save submission results
def save_submission(results: list, file_path: str):
    """Save results to submission CSV."""
    df = pd.DataFrame(results)
    df.to_csv(file_path, index=False)
    print(f"[Utils] Saved submission to: {file_path}")