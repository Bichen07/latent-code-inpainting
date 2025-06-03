import torch
import clip

def check_clip_vocabulary_and_embeddings(keywords, model_name="ViT-B/32"):
    """
    Checks if keywords are in CLIP's vocabulary and generates their embeddings.

    Args:
        keywords (list of str): A list of keywords or short phrases to check.
        model_name (str): The CLIP model to use (e.g., "ViT-B/32").

    Returns:
        tuple: (bool, torch.Tensor or None)
               - True if all keywords can be tokenized, False otherwise.
               - A tensor of text embeddings if successful, None otherwise.
    """
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load(model_name, device=device)
        print(f"CLIP model '{model_name}' loaded successfully on {device}.")

        # Tokenize the keywords
        # clip.tokenize() expects a list of strings, even for a single string.
        text_tokens = clip.tokenize(keywords).to(device)
        print(f"\nSuccessfully tokenized keywords: {keywords}")
        # print(f"Tokenized output (shape: {text_tokens.shape}):\n{text_tokens}")

        # Generate text embeddings
        with torch.no_grad(): # We don't need to track gradients for this
            text_features = model.encode_text(text_tokens)
        
        print(f"\nGenerated text embeddings (shape: {text_features.shape}).")
        # For ViT-B/32, the embedding dimension should be 512.
        # For example, if you input 5 keywords, shape will be (5, 512).
        
        # You can optionally print the first few elements of each embedding
        # for i, keyword in enumerate(keywords):
        #     print(f"Embedding for '{keyword}' (first 5 elements): {text_features[i, :5]}")
            
        return True, text_features

    except Exception as e:
        print(f"An error occurred: {e}")
        return False, None

if __name__ == '__main__':
    # --- Suggested Keywords for "football field" context ---
    # You and your friend can pick from this list or add your own.
    # Keep them relatively simple and distinct for a start.
    suggested_keywords = [
        "football field",       # 足球場
        "green grass",          # 綠色草地
        "stadium",              # 體育場，球場
        "player",               # 球員 (單數)
        "players",              # 球員 (複數)
        "audience",             # 觀眾 (通常指特定場合的聽眾/觀眾)
        "soccer ball",          # 足球
        "football",             # 足球 (可以指運動本身或球)
        "empty field",          # 空曠的場地
        "sports arena",         # 體育館，運動場地
        "jersey",               # 球衣
        "scoreboard",           # 計分板
        "lines on field"        # 場地上的線條
    ]

    print("--- Checking Suggested Keywords ---")
    success, embeddings = check_clip_vocabulary_and_embeddings(suggested_keywords)

    if success:
        print("\nAll suggested keywords are usable with CLIP and embeddings generated.")
    else:
        print("\nSome suggested keywords might have issues or an error occurred during processing.")
    
    # --- Example of checking a custom list (your potential choices) ---
    # my_chosen_keywords = ["football field", "player", "soccer ball", "green grass", "goal post"]
    # print(f"\n--- Checking My Chosen Keywords: {my_chosen_keywords} ---")
    # success_custom, embeddings_custom = check_clip_vocabulary_and_embeddings(my_chosen_keywords)
    # if success_custom:
    #     print("Custom keywords successfully processed.")