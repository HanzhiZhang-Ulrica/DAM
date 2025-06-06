import os
import torch
import triton
import triton.language as tl
import pickle
from torch.autograd import Function
from transformers.models.llama.modeling_llama import LlamaForCausalLM, LlamaAttention

MAX_QUERIES = 1024
MAX_KEYS = 1024


@triton.jit
def efficient_indexed_sparse_attention_kernel(
    Q_ptr, K_ptr, V_ptr, Out_ptr,
    q_indices_ptr, k_indices_ptr,
    stride_qz, stride_qh, stride_qs, stride_qd,
    stride_kz, stride_kh, stride_ks, stride_kd,
    stride_vz, stride_vh, stride_vs, stride_vd,
    stride_oz, stride_oh, stride_os, stride_od,
    num_queries: tl.constexpr,
    num_keys: tl.constexpr,
    head_dim: tl.constexpr
):
    pid = tl.program_id(0)
    if pid >= num_queries:
        return

    q_idx = tl.load(q_indices_ptr + pid)
    Q = Q_ptr + q_idx * stride_qs
    Out = Out_ptr + q_idx * stride_os

    head_dim_sqrt = tl.sqrt(float(head_dim))
    max_score = float('-inf')

    for k in range(num_keys):
        k_idx = tl.load(k_indices_ptr + k)
        K = K_ptr + k_idx * stride_ks
        score = 0.0
        for d in range(head_dim):
            q_val = tl.load(Q + d * stride_qd)
            k_val = tl.load(K + d * stride_kd)
            score += q_val * k_val
        score /= head_dim_sqrt
        max_score = tl.maximum(max_score, score)

    sum_exp = 0.0
    for k in range(num_keys):
        k_idx = tl.load(k_indices_ptr + k)
        K = K_ptr + k_idx * stride_ks
        score = 0.0
        for d in range(head_dim):
            q_val = tl.load(Q + d * stride_qd)
            k_val = tl.load(K + d * stride_kd)
            score += q_val * k_val
        score /= head_dim_sqrt
        sum_exp += tl.exp(score - max_score)

    for d in range(head_dim):
        context = 0.0
        for k in range(num_keys):
            k_idx = tl.load(k_indices_ptr + k)
            K = K_ptr + k_idx * stride_ks
            V = V_ptr + k_idx * stride_vs + d * stride_vd
            v_val = tl.load(V)

            score = 0.0
            for d_inner in range(head_dim):
                q_val = tl.load(Q + d_inner * stride_qd)
                k_val = tl.load(K + d_inner * stride_kd)
                score += q_val * k_val
            score /= head_dim_sqrt
            attn_prob = tl.exp(score - max_score) / sum_exp
            context += attn_prob * v_val

        tl.store(Out + d * stride_od, context)


class EfficientIndexedSparseAttentionFunction(Function):
    @staticmethod
    def forward(ctx, Q, K, V, query_indices, key_indices):
        bsz, num_heads, seq_len, head_dim = Q.size()
        device = Q.device

        Out = torch.zeros_like(Q)

        query_indices = query_indices[query_indices < seq_len]
        key_indices = key_indices[key_indices < seq_len]

        num_queries = len(query_indices)
        num_keys = len(key_indices)

        if num_queries == 0 or num_keys == 0:
            return Out

        print(f"Debug: EfficientIndexedSparseAttention running | Queries: {num_queries}, Keys: {num_keys}")

        q_indices_padded = torch.zeros((MAX_QUERIES,), device=device, dtype=torch.int32)
        k_indices_padded = torch.zeros((MAX_KEYS,), device=device, dtype=torch.int32)

        q_indices_padded[:num_queries] = query_indices
        k_indices_padded[:num_keys] = key_indices

        grid = (num_queries,)

        efficient_indexed_sparse_attention_kernel[grid](
            Q, K, V, Out,
            q_indices_padded, k_indices_padded,
            Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
            K.stride(0), K.stride(1), K.stride(2), K.stride(3),
            V.stride(0), V.stride(1), V.stride(2), V.stride(3),
            Out.stride(0), Out.stride(1), Out.stride(2), Out.stride(3),
            num_queries=num_queries,
            num_keys=num_keys,
            head_dim=head_dim
        )

        return Out


class DamLlamaAttention(LlamaAttention):
    def __init__(self, config, matched_positions, true_mask, layer_idx):
        super().__init__(config, layer_idx)
        self.matched_positions = matched_positions
        self.true_mask = true_mask  
        self.layer_idx = layer_idx
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // self.num_heads

    def forward(self, hidden_states, **kwargs):
        bsz, seq_len, _ = hidden_states.size()
        device = hidden_states.device

        query_states = self.q_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        key_states = key_states.repeat_interleave(self.num_heads // self.num_key_value_heads, dim=1)
        value_states = value_states.repeat_interleave(self.num_heads // self.num_key_value_heads, dim=1)

        context_layer = torch.zeros_like(query_states, device=device)

        print(f"\n🔹 [Layer {self.layer_idx}] Processing Attention | SeqLen={seq_len} | Batch={bsz}")

        for head_idx in range(self.num_heads):
            print(f"  🔹 [Head {head_idx}] Processing...")

            # ✅ Check if true_mask exists and has values
            if seq_len <= MAX_QUERIES and self.layer_idx in self.true_mask and head_idx in self.true_mask[self.layer_idx]:
                # ✅ Use precomputed true mask
                query_positions = self.true_mask[self.layer_idx][head_idx]["query"]
                key_positions = self.true_mask[self.layer_idx][head_idx]["key"]
                print(f"    ✅ Using precomputed true mask positions. Query: {query_positions.numel()}, Key: {key_positions.numel()}")
                
            else:
                # ✅ Start with True Mask for positions ≤ MAX_QUERIES
                query_positions = []
                key_positions = []

                if self.layer_idx in self.true_mask and head_idx in self.true_mask[self.layer_idx]:
                    query_positions = self.true_mask[self.layer_idx][head_idx]["query"].tolist()
                    key_positions = self.true_mask[self.layer_idx][head_idx]["key"].tolist()

                # ✅ Extend with Matched Positions for seq_len > MAX_QUERIES
                positions = self.matched_positions.get(self.layer_idx, {}).get(head_idx, {})
                extended_query_positions = sorted(set(positions.get('diagonal_row', []) + positions.get('vertical_col', [])))
                extended_key_positions = sorted(set(positions.get('vertical_col', []) + positions.get('diagonal_row', [])))

                query_positions += [pos for pos in extended_query_positions if pos >= MAX_QUERIES]
                key_positions += [pos for pos in extended_key_positions if pos >= MAX_QUERIES]

                print(f"    🔹 Extended query positions: {len(query_positions)}, key positions: {len(key_positions)}")

            # 🔹 Ensure positions are Tensors and Flatten
            if isinstance(query_positions, list):
                query_positions = torch.tensor(query_positions, dtype=torch.int32, device=device)
            if isinstance(key_positions, list):
                key_positions = torch.tensor(key_positions, dtype=torch.int32, device=device)

            # 🔹 Ensure indices are within allowed limits
            query_positions = query_positions.view(-1)[:MAX_QUERIES]  # Flatten & Truncate
            key_positions = key_positions.view(-1)[:MAX_KEYS]  # Flatten & Truncate

            # 🔹 Convert indices to LONG (Fix IndexError)
            query_positions = query_positions.to(torch.long)
            key_positions = key_positions.to(torch.long)

            if query_positions.numel() == 0 or key_positions.numel() == 0:
                print(f"⚠️ Skipping head {head_idx} in Layer {self.layer_idx} due to empty positions")
                continue

            print(f"    🔹 Query positions (First 10): {query_positions[:10]}")
            print(f"    🔹 Key positions (First 10): {key_positions[:10]}")

            # Convert to tensors (ensure valid type)
            q_indices_tensor = query_positions.clone().detach()
            k_indices_tensor = key_positions.clone().detach()

            print(f"    🔹 Running Efficient Indexed Sparse Attention for Head {head_idx}")

            context_sub = EfficientIndexedSparseAttentionFunction.apply(
                query_states[:, head_idx:head_idx+1, :, :],
                key_states[:, head_idx:head_idx+1, :, :],
                value_states[:, head_idx:head_idx+1, :, :],
                q_indices_tensor,
                k_indices_tensor
            )

            # 🔹 Fix: Ensure query_positions are LONG before using them for indexing
            context_layer[:, head_idx, query_positions, :] = context_sub[:, 0, query_positions, :]



        context_layer = context_layer.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        output = self.o_proj(context_layer)

        return output, None


class DamLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config, matched_positions=None, true_mask=None):
        super().__init__(config)
        self.matched_positions = matched_positions if matched_positions is not None else {}
        self.true_mask = true_mask if true_mask is not None else {}

        for layer_idx, layer in enumerate(self.model.layers):
            layer.self_attn = DamLlamaAttention(config, self.matched_positions, self.true_mask, layer_idx)
