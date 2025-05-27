import os
import torch
import threading
import queue
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader

# MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
DATASET_NAME = "alexfabbri/multi_news"
CACHE_DIR_MODEL = "../model"
CACHE_DIR_DATASET = "../dataset"
NORM_METHOD = "box-cox"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LENGTH = 1024
BATCH_SIZE = 8
ATTN_DATA_DIR = f"../intermediate_data_Llama-3.2-3B-Instruct/attn_maps_data_{NORM_METHOD}_{MAX_LENGTH}"
ATTN_PNG_DIR = f"../intermediate_data_Llama-3.2-3B-Instruct/attn_maps_png_{NORM_METHOD}_{MAX_LENGTH}"

torch.backends.cudnn.benchmark = True

class AttentionProcessor:
    def __init__(self, model_name, dataset_name, cache_dir_model, cache_dir_dataset, device, max_length):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.cache_dir_model = cache_dir_model
        self.cache_dir_dataset = cache_dir_dataset
        self.device = device
        self.max_length = max_length
        self.model, self.tokenizer = self.load_model_and_tokenizer()
        self.dataset = self.load_dataset()
        self.sort_dataset_by_length()
        self.collate_fn = self._create_collate_fn()

    def load_model_and_tokenizer(self):
        print(f"Loading model and tokenizer on device: {self.device}")
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir_model,
            device_map='auto',
            attn_implementation="eager",
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        ).eval()
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=self.cache_dir_model)
        if tokenizer.pad_token is None:
            if tokenizer.eos_token:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                model.resize_token_embeddings(len(tokenizer))
        return model, tokenizer

    def _create_collate_fn(self):
        def collate_fn(batch):
            documents = [item['document'] for item in batch]
            inputs = self.tokenizer(
                documents,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=self.max_length
            )
            return inputs
        return collate_fn

    def load_dataset(self):
        print("Loading dataset")
        dataset = load_dataset(
            self.dataset_name,
            cache_dir=self.cache_dir_dataset,
            split="train"
        )
        print("Tokenizing and computing lengths of documents")
        dataset = dataset.map(
            lambda batch: self.tokenizer(
                batch['document'],
                truncation=True,
                max_length=self.max_length,
                return_length=True,
            ),
            batched=True,
            batch_size=1000
        )
        dataset = dataset.map(
            lambda x: {'length': x['length']},
            batched=True
        )
        return dataset

    def sort_dataset_by_length(self):
        print("Sorting dataset by document length")
        self.dataset = self.dataset.sort("length", reverse=False)

    def compute_average_attention_maps(self, attention_sums, valid_counts):
        avg_attention_maps = []
        for attn_sum, count in zip(attention_sums, valid_counts):
            avg_attention = attn_sum / (count + 1e-10)
            avg_attention_maps.append(avg_attention)
        return avg_attention_maps

    def normalize_attention_maps(self, attention_sums, valid_counts, method, lam):
        if method == "average":
            return self.compute_average_attention_maps(attention_sums, valid_counts)
        elif method == "raw_sum":
            return attention_sums
        elif method == "log":
            normalized_attention_maps = []
            for attn_sum in attention_sums:
                data = attn_sum + 1
                transformed = torch.log(data)
                min_val = transformed.min()
                if min_val < 0:
                    transformed -= min_val
                normalized_attention_maps.append(transformed)
            return normalized_attention_maps
        elif method == "box-cox":
            normalized_attention_maps = []
            for attn_sum, count in zip(attention_sums, valid_counts):
                data = attn_sum / (count + 1e-10)
                data = data.clamp(min=1e-10)
                if lam != 0:
                    transformed = (data ** lam - 1) / lam
                else:
                    transformed = torch.log(data)
                min_val = transformed.min()
                if min_val < 0:
                    transformed -= min_val
                normalized_attention_maps.append(transformed)
            return normalized_attention_maps
        elif method == "yeo-johnson":
            normalized_attention_maps = []
            for attn_sum, count in zip(attention_sums, valid_counts):
                data = attn_sum / (count + 1e-10)
                x = data
                transformed = torch.zeros_like(x)
                pos = x >= 0
                neg = x < 0
                if lam != 0:
                    transformed[pos] = ((x[pos] + 1) ** lam - 1) / lam
                else:
                    transformed[pos] = torch.log(x[pos] + 1)
                if lam != 2:
                    transformed[neg] = -((-x[neg] + 1) ** (2 - lam) - 1) / (2 - lam)
                else:
                    transformed[neg] = -torch.log(-x[neg] + 1)
                min_val = transformed.min()
                if min_val < 0:
                    transformed -= min_val
                normalized_attention_maps.append(transformed)
            return normalized_attention_maps
        elif method == "sqrt":
            normalized_attention_maps = []
            for attn_sum in attention_sums:
                data = attn_sum.clamp(min=0)
                transformed = torch.sqrt(data)
                normalized_attention_maps.append(transformed)
            return normalized_attention_maps
        elif method == "min-max":
            normalized_attention_maps = []
            for attn_sum in attention_sums:
                data = attn_sum
                min_val = data.min()
                max_val = data.max()
                if max_val - min_val > 0:
                    transformed = (data - min_val) / (max_val - min_val)
                else:
                    transformed = data * 0
                normalized_attention_maps.append(transformed)
            return normalized_attention_maps
        elif method == "arcsinh":
            normalized_attention_maps = []
            for attn_sum in attention_sums:
                data = attn_sum
                transformed = torch.asinh(data)
                normalized_attention_maps.append(transformed)
            return normalized_attention_maps
        elif method == "z-score":
            normalized_attention_maps = []
            for attn_sum in attention_sums:
                data = attn_sum
                mean = data.mean()
                std = data.std()
                if std > 0:
                    transformed = (data - mean) / std
                else:
                    transformed = data * 0
                normalized_attention_maps.append(transformed)
            return normalized_attention_maps
        else:
            raise ValueError(f"Unknown normalization method: {method}")

    def process_documents(self, batch_size, normalization_method, lam):
        print(f"Processing documents in batches on device: {self.device}")
        num_layers = self.model.config.num_hidden_layers
        num_heads = self.model.config.num_attention_heads
        cumulative_attention_sums = [
            torch.zeros((num_heads, self.max_length, self.max_length), dtype=torch.float32)
            for _ in range(num_layers)
        ]
        cumulative_valid_token_counts = [
            torch.zeros((1, self.max_length, self.max_length), dtype=torch.float32)
            for _ in range(num_layers)
        ]
        dataloader = DataLoader(self.dataset, batch_size=batch_size, collate_fn=self.collate_fn, pin_memory=True)
        data_queue = queue.Queue(maxsize=10)
        total_batches = len(dataloader)
        pbar = tqdm(total=total_batches, desc="Processing document batches")

        def data_loader_thread(dataloader, data_queue, device):
            stream = torch.cuda.Stream(device=device)
            for batch in dataloader:
                with torch.cuda.stream(stream):
                    batch = {k: v.pin_memory() for k, v in batch.items()}
                    batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                event = torch.cuda.Event()
                event.record(stream)
                data_queue.put((batch, event))
            data_queue.put(None)

        loader_thread = threading.Thread(target=data_loader_thread, args=(dataloader, data_queue, self.device))
        loader_thread.start()

        while True:
            item = data_queue.get()
            if item is None:
                break
            batch, event = item
            event.wait()
            batch_attention_sums, batch_valid_token_counts = self._process_batch(batch)

            for idx in range(len(cumulative_attention_sums)):
                attn_sum = batch_attention_sums[idx]
                valid_count = batch_valid_token_counts[idx]
                seq_len = attn_sum.size(-1)
                if seq_len < self.max_length:
                    padding = self.max_length - seq_len
                    attn_sum = torch.nn.functional.pad(attn_sum, (0, padding, 0, padding), "constant", 0)
                    valid_count = torch.nn.functional.pad(valid_count, (0, padding, 0, padding), "constant", 0)
                cumulative_attention_sums[idx] += attn_sum.cpu()
                cumulative_valid_token_counts[idx] += valid_count.cpu()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            pbar.update(1)

        pbar.close()
        loader_thread.join()

        normalized_attention_maps = self.normalize_attention_maps(
            cumulative_attention_sums,
            cumulative_valid_token_counts,
            method=normalization_method,
            lam=lam
        )
        return normalized_attention_maps

    def _process_batch(self, batch):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True
            )
            attentions = outputs.attentions
        batch_attention_sums = []
        batch_valid_token_counts = []

        for layer_attention in attentions:
            seq_len = layer_attention.size(-1)
            valid_mask = attention_mask.unsqueeze(1).unsqueeze(2).float()
            valid_mask = valid_mask * valid_mask.transpose(-1, -2)
            weighted_attention = layer_attention * valid_mask
            attn_sum = weighted_attention.sum(dim=0)
            valid_count = valid_mask.sum(dim=0)
            batch_attention_sums.append(attn_sum.cpu())
            batch_valid_token_counts.append(valid_count.cpu())

        del batch, attentions, outputs, input_ids, attention_mask, layer_attention, valid_mask, weighted_attention
        torch.cuda.empty_cache()
        return batch_attention_sums, batch_valid_token_counts

class AttentionVisualizer:
    def __init__(self, output_dir_data, output_dir_png):
        self.output_dir_data = output_dir_data
        self.output_dir_png = output_dir_png
        os.makedirs(self.output_dir_data, exist_ok=True)
        os.makedirs(self.output_dir_png, exist_ok=True)

    def save_attention_maps(self, attention_maps):
        num_layers = len(attention_maps)
        num_heads = attention_maps[0].size(0)
        for layer_idx in tqdm(range(num_layers), desc="Saving attention maps"):
            for head_idx in range(num_heads):
                attention_map = attention_maps[layer_idx][head_idx]
                
                # Save as .pt file
                torch.save(
                    attention_map.to(torch.float32),
                    os.path.join(
                        self.output_dir_data,
                        f"layer_{layer_idx + 1}_head_{head_idx + 1}.pt"
                    )
                )
                
                # Save as .txt file
                txt_file_path = os.path.join(
                    self.output_dir_data,
                    f"layer_{layer_idx + 1}_head_{head_idx + 1}.txt"
                )
                
                with open(txt_file_path, "w") as f:
                    for row in attention_map.tolist():  # Convert tensor to list for easier writing
                        f.write(" ".join(map(str, row)) + "\n")

    def plot_attention_maps(self, attention_maps):
        num_layers = len(attention_maps)
        num_heads = attention_maps[0].size(0)
        for layer_idx in tqdm(range(num_layers), desc="Plotting attention maps"):
            for head_idx in range(num_heads):
                attention_map = attention_maps[layer_idx][head_idx]
                attention_numpy = attention_map.to(torch.float32).numpy()
                plt.figure(figsize=(32, 32))
                sns.heatmap(
                    attention_numpy, 
                    cmap="tab20c",
                    square=True,  # Keep squares consistent
                    cbar=False,   # Remove color bar if unnecessary
                    linewidths=1,  # Remove grid lines
                    linecolor='white',  # Set grid line color
                    xticklabels=False,  # Remove x labels
                    yticklabels=False   # Remove y labels
                )

                # Optionally, remove the axes spines for a cleaner look
                plt.axis('off')
                plt.savefig(
                    os.path.join(
                        self.output_dir_png,
                        f"layer_{layer_idx + 1}_head_{head_idx + 1}.png"
                    ),
                    bbox_inches='tight',  # Remove white borders
                    pad_inches=0  # Optional: minimize padding
                )
                plt.close()

if __name__ == "__main__":
    processor = AttentionProcessor(
        MODEL_NAME,
        DATASET_NAME,
        CACHE_DIR_MODEL,
        CACHE_DIR_DATASET,
        DEVICE,
        max_length=MAX_LENGTH
    )
    visualizer = AttentionVisualizer(ATTN_DATA_DIR, ATTN_PNG_DIR)
    normalization_method = NORM_METHOD
    lam = 0.5
    attention_maps = processor.process_documents(
        batch_size=BATCH_SIZE,
        normalization_method=normalization_method,
        lam=lam
    )
    visualizer.save_attention_maps(attention_maps)
    visualizer.plot_attention_maps(attention_maps)