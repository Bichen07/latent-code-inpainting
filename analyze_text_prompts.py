# analyze_text_prompts.py
import pandas as pd
from collections import Counter
import re
import os

def analyze_text_prompts(csv_filepath, output_dir):
    """
    Reads a CSV file, extracts and tokenizes text prompts,
    counts word frequencies, and saves the sorted results to a text file.

    Args:
        csv_filepath (str): Path to the input CSV file.
        output_dir (str): Directory where the output file will be saved.
    """
    try:
        df = pd.read_csv(csv_filepath)
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_filepath}")
        return
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    if 'text_prompt' not in df.columns:
        print("Error: 'text_prompt' column not found in the CSV file.")
        return

    all_words = []
    # Use tqdm if you have a large number of prompts for a progress bar
    # from tqdm import tqdm
    # for prompt in tqdm(df['text_prompt'], desc="Processing text prompts"):
    for prompt in df['text_prompt']:
        if pd.isna(prompt): # Handle potential NaN prompts
            continue
        # Tokenize words: convert to lowercase, remove punctuation, split by spaces
        # This regex keeps only alphanumeric characters and converts to lowercase
        words = re.findall(r'\b\w+\b', str(prompt).lower())
        all_words.extend(words)

    word_counts = Counter(all_words)

    # Sort words by frequency in descending order
    sorted_word_counts = sorted(word_counts.items(), key=lambda item: item[1], reverse=True)

    output_filename = "word_frequencies.txt"
    output_filepath = os.path.join(output_dir, output_filename)

    with open(output_filepath, 'w', encoding='utf-8') as f:
        f.write("Word Frequencies (sorted by count, descending):\n")
        f.write("--------------------------------------------------\n")
        for word, count in sorted_word_counts:
            f.write(f"{word}: {count}\n")

    print(f"\nWord frequencies saved to: {output_filepath}")
    print(f"Total unique words found: {len(sorted_word_counts)}")

if __name__ == "__main__":
    # Define the base directory where the script will be created and output saved
    base_dir = "/home/carlos11/Downloads/code/SecondSemester/DLP/final/latent-code-inpainting/"

    # Define the path to your input CSV file
    csv_file = os.path.join(base_dir, "annotations", "generated_annotations.csv")
    
    # Define the output directory (same as base_dir in this case)
    output_directory = "/home/carlos11/Downloads/code/SecondSemester/DLP/final/latent-code-inpainting/annotations/"

    # Create the output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    print(f"Starting analysis for CSV: {csv_file}")
    analyze_text_prompts(csv_file, output_directory)