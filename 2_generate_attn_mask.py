import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

NORM_METHOD = "box-cox"
MAX_LENGTH = 512
THRESHOLD = 0.3
MATCH_THRESHOLD = 0.8

INPUT_DIR_DATA = f"../intermediate_data_Llama-3.2-3B-Instruct/attn_maps_data_{NORM_METHOD}_{MAX_LENGTH}"

OUTPUT_DIR_SPARSE_DATA = f"../intermediate_data_Llama-3.2-3B-Instruct/sparse_maps_data_{MAX_LENGTH}"
OUTPUT_DIR_SPARSE_PNG = f"../intermediate_data_Llama-3.2-3B-Instruct/sparse_maps_png_{MAX_LENGTH}_{NORM_METHOD}"

OUTPUT_DIR_DATA_MASKS = f"../intermediate_data_Llama-3.2-3B-Instruct/true_mask_data_{MAX_LENGTH}_{THRESHOLD}_{MATCH_THRESHOLD}"
OUTPUT_DIR_PNG_MASKS = f"../intermediate_data_Llama-3.2-3B-Instruct/true_mask_png_{MAX_LENGTH}_{NORM_METHOD}_{THRESHOLD}_{MATCH_THRESHOLD}"

OUTPUT_DIR_FFT_DATA_ATTENTION_MAPS = f"../intermediate_data_Llama-3.2-3B-Instruct/appro_mask_data_{MAX_LENGTH}_{THRESHOLD}_{MATCH_THRESHOLD}"
OUTPUT_DIR_FFT_PNG_ATTENTION_MAPS = f"../intermediate_data_Llama-3.2-3B-Instruct/appro_mask_png_{MAX_LENGTH}_{NORM_METHOD}_{THRESHOLD}_{MATCH_THRESHOLD}"

MATCHED_PATTERNS_FILE = f"../intermediate_data_Llama-3.2-3B-Instruct/matched_patterns_positions_{MAX_LENGTH}_{THRESHOLD}_{MATCH_THRESHOLD}.pkl"

class AttentionMaskGenerator:
    def __init__(self, threshold=THRESHOLD, match_threshold=MATCH_THRESHOLD):
        self.match_threshold = match_threshold
        self.threshold = threshold

    def load_attention_maps(self, input_dir):
        attention_maps = {}
        attention_files = [f for f in os.listdir(input_dir) if f.endswith('.pt')]
        for attention_file in tqdm(attention_files, desc="Loading attention maps"):
            attention_map = torch.load(
                os.path.join(input_dir, attention_file)
            )
            attention_maps[attention_file] = attention_map.to(torch.float32).numpy()
        return attention_maps

    def generate_pattern_pool(self, size):
        pattern_pool = {}

        # Diagonal patterns
        for row in range(size):
            pattern = torch.zeros((size, size), dtype=torch.float32)
            i, j = row, 0  # Start from (row, 0)
            while i < size and j < size:
                pattern[i, j] = 1
                i += 1
                j += 1
            pattern_pool[f'diagonal_row_{row}'] = pattern

        # Vertical patterns
        for col in range(size):
            pattern = torch.zeros((size, size), dtype=torch.float32)
            start_row = col  # Start filling from the diagonal position (col, col)
            pattern[start_row:, col] = 1  # Fill from start_row to the bottom
            pattern_pool[f'vertical_col_{col}'] = pattern

        return pattern_pool
    def generate_sparse_maps(self, attention_maps):
        sparse_maps = {}
        for file_name, attention_map in tqdm(attention_maps.items(), desc="Generating sparse maps"):
            sparse_map = attention_map.copy()
            sparse_map[sparse_map < self.threshold] = 0
            sparse_maps[file_name] = sparse_map
        return sparse_maps


    def generate_attention_masks(self, attention_maps):
        attention_masks = {}
        for file_name, attention_map in tqdm(attention_maps.items(), desc="Generating attention masks"):
            mask = attention_map.copy()
            mask[mask < self.threshold] = 0
            mask[mask >= self.threshold] = 1
            attention_masks[file_name] = mask  
        return attention_masks
    
    def generate_attention_masks_with_patterns(self, attention_masks, pattern_pool):
        attention_masks_with_patterns = {}
        matched_positions = {}

        for file_name, mask in tqdm(attention_masks.items(), desc="Generating attention masks with patterns"):
            mask_tensor = mask if isinstance(mask, torch.Tensor) else torch.tensor(mask, dtype=torch.float32)

            matched_patterns = torch.zeros_like(mask_tensor)
            matched_positions[file_name] = {'diagonal_row': [], 'vertical_col': []}

            for pattern_name, pattern in pattern_pool.items():
                pattern_tensor = pattern if isinstance(pattern, torch.Tensor) else torch.tensor(pattern, dtype=torch.float32)

                # Compute the element-wise matching positions
                matching_positions = mask_tensor * pattern_tensor

                # Calculate the similarity score
                similarity_score = matching_positions.sum() / pattern_tensor.sum()

                if similarity_score >= self.match_threshold:
                    # Update the matched patterns
                    matched_patterns += pattern_tensor

                    # Record matched pattern positions
                    if pattern_name.startswith('diagonal_row_'):
                        row_num = int(pattern_name.split('_')[-1])
                        matched_positions[file_name]['diagonal_row'].append(row_num)
                    elif pattern_name.startswith('vertical_col_'):
                        col_num = int(pattern_name.split('_')[-1])
                        matched_positions[file_name]['vertical_col'].append(col_num)

            # Create the final mask indicating matched patterns
            final_mask = (matched_patterns >= 1).float()

            attention_masks_with_patterns[file_name] = final_mask

        # Save the matched positions dictionary to a pickle file
        with open(MATCHED_PATTERNS_FILE, 'wb') as f:
            pickle.dump(matched_positions, f)

        return attention_masks_with_patterns

    def save_data(self, data_dict, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        for name, data in tqdm(data_dict.items(), desc="Saving data"):
            if isinstance(data, np.ndarray):
                tensor = torch.from_numpy(data)
            else:
                tensor = data
            torch.save(tensor.float(), os.path.join(output_dir, name))

    def save_images(self, data_dict, output_dir, title_suffix=''):
        os.makedirs(output_dir, exist_ok=True)
        for name, data in tqdm(data_dict.items(), desc="Saving images"):
            if isinstance(data, torch.Tensor):
                array = data.numpy()
            else:
                array = data
            plt.figure(figsize=(8, 8))
            sns.heatmap(
                array, 
                cmap="tab20c",
                square=True,  # Keep squares consistent
                cbar=False,   # Remove color bar if unnecessary
                linewidths=0.5,  # Remove grid lines
                linecolor='white',  # Set grid line color
                xticklabels=False,  # Remove x labels
                yticklabels=False   # Remove y labels
            )
            plt.axis('off')
            plt.savefig(
                os.path.join(output_dir, name.replace('.pt', '.png')),  # Save as PNG
                bbox_inches='tight',  # Remove white borders
                pad_inches=0  # Optional: minimize padding
                )
            plt.close()

if __name__ == "__main__":
    generator = AttentionMaskGenerator()

    attention_maps = generator.load_attention_maps(INPUT_DIR_DATA)

    pattern_pool = generator.generate_pattern_pool(MAX_LENGTH)

    sparse_maps = generator.generate_sparse_maps(attention_maps)
    generator.save_data(sparse_maps, OUTPUT_DIR_SPARSE_DATA)
    generator.save_images(sparse_maps, OUTPUT_DIR_SPARSE_PNG, title_suffix='Sparse Map')

    attention_masks = generator.generate_attention_masks(sparse_maps)

    generator.save_data(attention_masks, OUTPUT_DIR_DATA_MASKS)
    generator.save_images(attention_masks, OUTPUT_DIR_PNG_MASKS, title_suffix='True Mask')

    attention_masks_patterns = generator.generate_attention_masks_with_patterns(attention_masks, pattern_pool)
    generator.save_data(attention_masks_patterns, OUTPUT_DIR_FFT_DATA_ATTENTION_MAPS)
    generator.save_images(attention_masks_patterns, OUTPUT_DIR_FFT_PNG_ATTENTION_MAPS, title_suffix='Appro Mask')
