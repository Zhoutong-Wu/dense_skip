import torch
import json
import os

def compress_value_projection(W_V, W_O, rank):
    """
    Approximate M = W_V @ W_O with rank-r factors W_V' and W_O'.
    W_V: [d × D], W_O: [D × d], rank <= D
    Returns:
      W_V_new: [d × rank],
      W_O_new: [rank × d]
    """
    M = W_V @ W_O                        
    U, S, V = torch.svd_lowrank(M, q=rank, niter=2)

    # Truncate
    U_r = U[:, :rank]                   
    S_r = S[:rank]                       
    V_r = V[:, :rank]                    

    # Form factors
    sqrt_S = torch.sqrt(S_r)           
    W_V_new = U_r * sqrt_S.unsqueeze(0) 
    # W_O_new should be [r×d]: take V_r^T
    W_O_new = (V_r * sqrt_S.unsqueeze(0)).T  

    return W_V_new, W_O_new

def merge_configs(original_config_path, skipv1_config):
    """
    Merge the original config and SkipV1 config.

    Args:
        original_config_path (str): Path to the original summary.json file.
        skipv1_config (dict): SkipV1 config dictionary.

    Returns:
        dict: Merged config dictionary.
    """
    with open(original_config_path, "r") as f:
        original_config = json.load(f)
    original_config["args"].update(skipv1_config)
    return original_config


def convert_gpt2_checkpoint_to_skipv1_svd(gpt2_checkpoint, config):
    """
    Convert a standard GPT-2 checkpoint to a SkipV1 checkpoint using SVD decomposition.

    Args:
        gpt2_checkpoint (dict): Standard GPT-2 checkpoint.
        config (dict): Model config, including n_head, n_embd, num_skip_head, etc.

    Returns:
        dict: Converted SkipV1 checkpoint.
    """
    skipv1_checkpoint = {}
    num_skip_heads = config["num_skip_head"]
    split_num = config["n_head"] - num_skip_heads
    head_dim = config["n_embd"] // config["n_head"]

    for key, value in gpt2_checkpoint.items():
        if "attn.c_attn.weight" in key:
            value = value.T  # transpose weights
            # Split Q, K, V weights
            q_weight, k_weight, v_weight = value.split(config["n_embd"], dim=1)

            if "transformer.h.0" in key:
                new_v_weight = v_weight
            else:
                # Get corresponding W_O
                w_o_key = key.replace("attn.c_attn.weight", "attn.c_proj.weight")
                w_o_weight = gpt2_checkpoint[w_o_key].T
                split_dim = config["n_embd"] // 2
                new_v_weight, W_O_new = compress_value_projection(v_weight, w_o_weight, v_weight.shape[1] // 2)
                new_w_o_weight = torch.zeros_like(w_o_weight)
                new_w_o_weight[:split_dim, :] = W_O_new
                new_w_o_weight = new_w_o_weight.T

                layer_idx = key.split(".")[2]
                skipv1_checkpoint[f"transformer.h.{layer_idx}.attn.c_proj.weight"] = new_w_o_weight

            layer_idx = key.split(".")[2]
            q_weight = q_weight.T
            k_weight = k_weight.T
            new_v_weight = new_v_weight.T

            skipv1_checkpoint[f"transformer.h.{layer_idx}.attn.q_proj.weight"] = q_weight
            skipv1_checkpoint[f"transformer.h.{layer_idx}.attn.k_proj.weight"] = k_weight
            skipv1_checkpoint[f"transformer.h.{layer_idx}.attn.v_proj.weight"] = new_v_weight
        else:
            skipv1_checkpoint[key] = value

    return skipv1_checkpoint

def convert_gpt2_checkpoint_to_skipv1_avg(gpt2_checkpoint, config):
    """
    Convert a standard GPT-2 checkpoint to a SkipV1 checkpoint using mean pooling for V.

    Args:
        gpt2_checkpoint (dict): Standard GPT-2 checkpoint.
        config (dict): Model config, including n_head, n_embd, num_skip_head, etc.

    Returns:
        dict: Converted SkipV1 checkpoint.
    """
    skipv1_checkpoint = {}
    num_skip_heads = config["num_skip_head"]
    split_num = config["n_head"] - num_skip_heads
    head_dim = config["n_embd"] // config["n_head"]

    for key, value in gpt2_checkpoint.items():
        if "attn.c_attn.weight" in key:
            value = value.T
            q_weight, k_weight, v_weight = value.split(config["n_embd"], dim=1)

            if "transformer.h.0" in key:
                new_v_weight = v_weight
            else:
                # For layers after the first, average V weights in pairs
                v_weight = v_weight.view(config["n_embd"], config["n_head"], head_dim)
                new_v_weight = (v_weight[:, :split_num, :] + v_weight[:, split_num:, :]) / 2
                new_v_weight = new_v_weight.view(config["n_embd"], split_num * head_dim)

            layer_idx = key.split(".")[2]
            q_weight = q_weight.T
            k_weight = k_weight.T
            new_v_weight = new_v_weight.T

            skipv1_checkpoint[f"transformer.h.{layer_idx}.attn.q_proj.weight"] = q_weight
            skipv1_checkpoint[f"transformer.h.{layer_idx}.attn.k_proj.weight"] = k_weight
            skipv1_checkpoint[f"transformer.h.{layer_idx}.attn.v_proj.weight"] = new_v_weight
        elif "attn.c_proj.weight" in key:
            layer_idx = key.split(".")[2]
            skipv1_checkpoint[f"transformer.h.{layer_idx}.attn.c_proj.weight"] = value
        else:
            skipv1_checkpoint[key] = value

    return skipv1_checkpoint

def convert_gpt2_checkpoint_to_skipv1_topk(gpt2_checkpoint, config):
    """
    Convert a standard GPT-2 checkpoint to a SkipV1 checkpoint by selecting top-k V columns and W_O rows by norm product.

    Args:
        gpt2_checkpoint (dict): Standard GPT-2 checkpoint.
        config (dict): Model config, including n_head, n_embd, num_skip_head, etc.

    Returns:
        dict: Converted SkipV1 checkpoint.
    """
    skipv1_checkpoint = {}
    num_skip_heads = config["num_skip_head"]
    split_num = config["n_head"] - num_skip_heads
    head_dim = config["n_embd"] // config["n_head"]

    for key, value in gpt2_checkpoint.items():
        if "attn.c_attn.weight" in key:
            value = value.T
            q_weight, k_weight, v_weight = value.split(config["n_embd"], dim=1)

            if "transformer.h.0" in key:
                new_v_weight = v_weight
            else:
                w_o_key = key.replace("attn.c_attn.weight", "attn.c_proj.weight")
                w_o_weight = gpt2_checkpoint[w_o_key].T

                v_norms = torch.norm(v_weight, dim=0)
                o_norms = torch.norm(w_o_weight, dim=1)
                contribution_scores = v_norms * o_norms

                top_indices = torch.topk(contribution_scores, split_num * head_dim, largest=True).indices
                new_v_weight = v_weight[:, top_indices]

                new_w_o_weight = torch.zeros_like(w_o_weight)
                for new_idx, old_idx in enumerate(top_indices):
                    new_w_o_weight[new_idx, :] = w_o_weight[old_idx, :]
                new_w_o_weight = new_w_o_weight.T

                layer_idx = key.split(".")[2]
                skipv1_checkpoint[f"transformer.h.{layer_idx}.attn.c_proj.weight"] = new_w_o_weight

            layer_idx = key.split(".")[2]
            q_weight = q_weight.T
            k_weight = k_weight.T
            new_v_weight = new_v_weight.T

            skipv1_checkpoint[f"transformer.h.{layer_idx}.attn.q_proj.weight"] = q_weight
            skipv1_checkpoint[f"transformer.h.{layer_idx}.attn.k_proj.weight"] = k_weight
            skipv1_checkpoint[f"transformer.h.{layer_idx}.attn.v_proj.weight"] = new_v_weight
        else:
            skipv1_checkpoint[key] = value

    return skipv1_checkpoint

def convert_gpt2_checkpoint_to_skipv1_topk1(gpt2_checkpoint, config):
    """
    Convert a standard GPT-2 checkpoint to a SkipV1 checkpoint by selecting top-k V columns and copying corresponding W_O rows.

    Args:
        gpt2_checkpoint (dict): Standard GPT-2 checkpoint.
        config (dict): Model config, including n_head, n_embd, num_skip_head, etc.

    Returns:
        dict: Converted SkipV1 checkpoint.
    """
    skipv1_checkpoint = {}
    num_skip_heads = config["num_skip_head"]
    split_num = config["n_head"] - num_skip_heads
    head_dim = config["n_embd"] // config["n_head"]

    for key, value in gpt2_checkpoint.items():
        if "attn.c_attn.weight" in key:
            value = value.T
            q_weight, k_weight, v_weight = value.split(config["n_embd"], dim=1)

            if "transformer.h.0" in key:
                new_v_weight = v_weight
            else:
                w_o_key = key.replace("attn.c_attn.weight", "attn.c_proj.weight")
                w_o_weight = gpt2_checkpoint[w_o_key].T

                v_norms = torch.norm(v_weight, dim=0)
                o_norms = torch.norm(w_o_weight, dim=1)
                contribution_scores = v_norms * o_norms

                top_indices = torch.topk(contribution_scores, split_num * head_dim, largest=True).indices
                new_v_weight = v_weight[:, top_indices]

                new_w_o_weight = w_o_weight.clone()
                for new_idx, old_idx in enumerate(top_indices):
                    new_w_o_weight[new_idx, :] = w_o_weight[old_idx, :]
                new_w_o_weight = new_w_o_weight.T

                layer_idx = key.split(".")[2]
                skipv1_checkpoint[f"transformer.h.{layer_idx}.attn.c_proj.weight"] = new_w_o_weight

            layer_idx = key.split(".")[2]
            q_weight = q_weight.T
            k_weight = k_weight.T
            new_v_weight = new_v_weight.T

            skipv1_checkpoint[f"transformer.h.{layer_idx}.attn.q_proj.weight"] = q_weight
            skipv1_checkpoint[f"transformer.h.{layer_idx}.attn.k_proj.weight"] = k_weight
            skipv1_checkpoint[f"transformer.h.{layer_idx}.attn.v_proj.weight"] = new_v_weight
        else:
            skipv1_checkpoint[key] = value

    return skipv1_checkpoint




if __name__ == "__main__":
    # Path to the standard GPT-2 checkpoint and config
    checkpoint_path = "/data2/ztwu/exp/base/o_base_lr0.001_bs30x4_seqlen1024/no_compile=True_seed=2/ckpt.pt"
    original_summary_path = "/data2/ztwu/exp/base/o_base_lr0.001_bs30x4_seqlen1024/no_compile=True_seed=2/summary.json"
    gpt2_checkpoint = torch.load(checkpoint_path)
    
    # Extract model weights
    if "model" in gpt2_checkpoint:
        model_weights = gpt2_checkpoint["model"]
    else:
        raise KeyError("The checkpoint does not contain a 'model' key with weights.")

    skipv1_config = {
        "model": "skipv1",
        "num_skip_head": 6,
        "num_skip_layer": 1,
        "positional_encoder": "rotary",
        "lm_cache": "none",
        "allow_cache_during_training": False,
    }
    
    merged_config = merge_configs(original_summary_path, skipv1_config)

    # Convert checkpoint
    class Config:
        def __init__(self, config_dict):
            for key, value in config_dict.items():
                setattr(self, key, value)
    
    config = Config(merged_config["args"])
    model_config = {
        "n_embd": config.n_embd,
        "n_head": config.n_head,
        "num_skip_head": config.num_skip_head,
        "n_layer": config.n_layer,
    }
    skipv1_checkpoint = convert_gpt2_checkpoint_to_skipv1_svd(model_weights, model_config)

    # Save the converted checkpoint
    output_dir = "/data2/ztwu/exp/skipv1/s_skipv1_lr0.001_bs30x4_seqlen1024/no_compile=True_seed=2"
    ckpt_path = f"{output_dir}/ckpt.pt"
    summary_path = f"{output_dir}/summary.json"

    final_checkpoint = {
        "model": skipv1_checkpoint,
        "optimizer": gpt2_checkpoint.get("optimizer", None),
        "scheduler": gpt2_checkpoint.get("scheduler", None),
        "itr": gpt2_checkpoint.get("itr", None),
    }
    os.makedirs(output_dir, exist_ok=True)
    torch.save(final_checkpoint, ckpt_path)

    with open(summary_path, "w") as f:
        json.dump(merged_config, f, indent=4)

    print(f"Converted checkpoint saved to: {ckpt_path}")
    print(f"Summary saved to: {summary_path}")