import os
import json
from pathlib import Path
import argparse

#<your image data base path>
#-image
#--original_image.jpg
#-prompt
#--original_image_prompt.txt
#--edit_image_prompt.txt
#-splits
#--1.000_000001000.jsonl


def parse_args():
    parser = argparse.ArgumentParser(description="Create training JSONL splits for EditInfinity single-image finetuning.")
    parser.add_argument("--base_path", "-b", type=str, required=True,
                        help="Root directory that contains multiple sample folders, each with 'image/' and 'prompt/' subfolders.")
    parser.add_argument("--entries", "-n", type=int, default=1000,
                        help="Number of duplicated entries to write into the JSONL file per sample directory (default: 1000).")
    return parser.parse_args()

args = parse_args()
base_path = args.base_path
NUM_ENTRIES = args.entries

if not os.path.isdir(base_path):
    raise FileNotFoundError(f"Base path not found or not a directory: {base_path}")

def create_jsonl_entry(directory):
    """Create one JSONL entry for a single directory."""
    # Build image_path
    image_path = os.path.join(directory, "image", "original_image.jpg") # The image is named "original_image.jpg"
    
    # Read the content of original_image_prompt.txt
    prompt_file = os.path.join(directory, "prompt", "original_image_prompt.txt") # The prompt file is named "original_image_prompt.txt"
    try:
        with open(prompt_file, 'r') as f:
            caption = f.read().strip()
    except FileNotFoundError:
        print(f"Warning: Prompt file not found in {directory}")
        return None
    
    # Construct a standard entry
    entry = {
        "image_path": image_path,
        "h_div_w": 1.0,
        "long_caption": caption,
        "long_caption_type": "caption-InternVL2.0",
        "text": caption,
        "short_caption_type": "blip2_caption"
    }
    return entry

def process_directory(directory):
    """Process a single directory to produce its JSONL split file."""
    # Create the splits folder
    splits_dir = os.path.join(directory, "splits")
    os.makedirs(splits_dir, exist_ok=True)
    
    # Clear all .jsonl files under the splits directory
    for file in os.listdir(splits_dir):
        if file.endswith('.jsonl'):
            os.remove(os.path.join(splits_dir, file))
            print(f"Cleared existing: {os.path.join(splits_dir, file)}")
    
    # Generate the JSONL entry
    entry = create_jsonl_entry(directory)
    if entry is None:
        return
    
    # Create the JSONL file and write duplicated lines
    jsonl_file = os.path.join(splits_dir, "1.000_000001000.jsonl")
    with open(jsonl_file, 'w') as f:
        for _ in range(NUM_ENTRIES):
            f.write(json.dumps(entry) + "\n")

    print(f"Created: {jsonl_file} with {NUM_ENTRIES} entries")

# Traverse all directories under base_path
for root, dirs, _ in os.walk(base_path):
    # Check if both 'prompt' and 'image' folders exist (ensure it's a target directory)
    if "prompt" in dirs and "image" in dirs:
        process_directory(root)

print("All directories processed successfully.")