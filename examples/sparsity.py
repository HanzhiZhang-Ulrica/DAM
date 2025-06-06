import torch
from transformers import AutoModelForCausalLM, LlamaForCausalLM, LlamaConfig, cached_file, WEIGHTS_NAME

# Define model names and cache directory
original_model_name = "meta-llama/Llama-3.2-3B-Instruct"
sparse_model_name = "HanzhiZhang/DAM"
cache_dir = '../model'  
device_map = 'auto'    

# Load configurations
original_config = LlamaConfig.from_pretrained(
    original_model_name,
    cache_dir=cache_dir
)
sparse_config = LlamaConfig.from_pretrained(
    sparse_model_name,
    cache_dir=cache_dir
)

# Inspect the sparse model's parameter shapes
model_file = cached_file(sparse_model_name, WEIGHTS_NAME, cache_dir=cache_dir)
state_dict = torch.load(model_file, map_location="cpu")

print("Sparse Model Parameter Shapes:")
for name, param in state_dict.items():
    print(f"{name}: {param.shape}")

# Adjust the sparse model configuration based on the state_dict
# For example, if you find that 'model.embed_tokens.weight' has shape [512, 2048]:
sparse_config.vocab_size = 512  # Adjust to match the actual vocab size

# Adjust other configurations if necessary
sparse_config.hidden_size = 2048  # As previously determined
sparse_config.num_attention_heads = sparse_config.hidden_size // sparse_config.head_dim  # 2048 // 128 = 16
sparse_config.intermediate_size = sparse_config.hidden_size * 4  # 2048 * 4 = 8192

# Print configurations
print("Original Model Configuration:")
print(original_config)
print("\nAdjusted Sparse Model Configuration:")
print(sparse_config)

# Load the original and sparse models
original_model = LlamaForCausalLM.from_pretrained(
    original_model_name,
    cache_dir=cache_dir,
    device_map=device_map
)

# Load the sparse model with adjusted configuration
sparse_model = LlamaForCausalLM.from_pretrained(
    sparse_model_name,
    cache_dir=cache_dir,
    device_map=device_map,
    config=sparse_config,
    ignore_mismatched_sizes=True  # Use if necessary
)

def calculate_sparsity(model):
    total_params = 0
    zero_params = 0

    for param in model.parameters():
        total_params += param.numel()
        zero_params += torch.sum(param == 0).item()

    sparsity = (zero_params / total_params) * 100  # Percentage
    return sparsity

# Calculate sparsity of the sparse model
sparsity_percentage = calculate_sparsity(sparse_model)
print(f"The sparsity of the model is {sparsity_percentage:.2f}%")
