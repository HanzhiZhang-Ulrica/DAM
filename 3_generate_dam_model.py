import os
import re
import pickle
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from dam_attention import DamLlamaForCausalLM  # Import the full model

MAX_LENGTH = 1024
THRESHOLD = 0.1
MATCH_THRESHOLD = 0.9

MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
CACHE_DIR_MODEL = "../model"
MATCHED_PATTERNS_FILE = f"../intermediate_data_Llama-3.2-3B-Instruct/matched_patterns_positions_{MAX_LENGTH}_{THRESHOLD}_{MATCH_THRESHOLD}.pkl"
TRUE_MASK_DIR = f"../intermediate_data_Llama-3.2-3B-Instruct/true_mask_data_{MAX_LENGTH}_{THRESHOLD}_{MATCH_THRESHOLD}"
EDITED_MODEL_PATH = f"../DAM_3B_triton_{MAX_LENGTH}_{THRESHOLD}_{MATCH_THRESHOLD}"


def load_matched_positions(filepath):
    """Load matched positions for efficient attention from a pickle file."""
    if os.path.exists(filepath):
        with open(filepath, "rb") as f:
            matched_positions = pickle.load(f)
    else:
        print(f"Warning: {filepath} not found! Using empty matched positions.")
        return {}

    print(f"Loaded matched positions for {len(matched_positions)} layers.")
    
    # # Debugging print:
    # for layer, heads in matched_positions.items():
    #     for head, pos in heads.items():
    #         print(f"Matched Positions - Layer {layer}, Head {head}: {pos}")

    return matched_positions


def load_true_mask(directory):
    """Load true mask data from multiple `.pt` files in a directory."""
    true_mask = {}
    if not os.path.exists(directory):
        print(f"Warning: {directory} not found! Using only matched positions.")
        return true_mask

    for file in os.listdir(directory):
        match = re.match(r"layer_(\d+)_head_(\d+)\.pt", file)
        if match:
            layer_idx = int(match.group(1))
            head_idx = int(match.group(2))
            file_path = os.path.join(directory, file)

            mask_tensor = torch.load(file_path)
            mask_data = {"query": mask_tensor, "key": mask_tensor}  

            if layer_idx not in true_mask:
                true_mask[layer_idx] = {}
            true_mask[layer_idx][head_idx] = mask_data  

    print(f"Loaded true mask data for {len(true_mask)} layers from {directory}.")
    return true_mask


def generate_dam_model():
    print("Loading original model & tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, cache_dir=CACHE_DIR_MODEL, torch_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR_MODEL)

    print("Loading matched positions...")
    matched_positions = load_matched_positions(MATCHED_PATTERNS_FILE)

    print("Loading true mask...")
    true_mask = load_true_mask(TRUE_MASK_DIR)

    print("Replacing attention modules with efficient Triton-based indexed sparse attention...")
    model = DamLlamaForCausalLM(config=model.config, matched_positions=matched_positions, true_mask=true_mask)

    print("Validating model on CUDA...")
    model.to("cuda")
    input_ids = tokenizer("Test input", return_tensors="pt").input_ids.to("cuda")
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
        print(f"Validation successful! Output shape: {outputs.logits.shape}")

    print("Saving modified model...")
    os.makedirs(EDITED_MODEL_PATH, exist_ok=True)

    with open(os.path.join(EDITED_MODEL_PATH, "matched_positions.pkl"), "wb") as f:
        pickle.dump(matched_positions, f, protocol=pickle.HIGHEST_PROTOCOL)

    true_mask_path = os.path.join(EDITED_MODEL_PATH, "true_mask_data")
    os.makedirs(true_mask_path, exist_ok=True)

    for layer_idx, heads in true_mask.items():
        for head_idx, mask_data in heads.items():
            torch.save(mask_data["query"], os.path.join(true_mask_path, f"layer_{layer_idx}_head_{head_idx}.pt"))

    model.config.save_pretrained(EDITED_MODEL_PATH)
    model.save_pretrained(EDITED_MODEL_PATH)
    tokenizer.save_pretrained(EDITED_MODEL_PATH)

    os.system(f"cp dam_attention.py {EDITED_MODEL_PATH}/dam_attention.py")

    print(f"DAM model saved to {EDITED_MODEL_PATH}")


if __name__ == "__main__":
    generate_dam_model()
