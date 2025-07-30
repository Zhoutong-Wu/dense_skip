import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import sys
import numpy as np
import torch
import inspect
import json
import copy
import argparse
import random
import wandb
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import config
import models
import distributed
from optim.utils import get_batch
from models.base import CausalSelfAttention
import matplotlib.ticker as ticker

def get_args():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--config_format', default='base', choices=config.registered_formats())
    parser.add_argument('--resume_ckpt', type=str, default=None, help="Path to the checkpoint to resume training from")

    args, rem_args = parser.parse_known_args()
    
    if args.resume_ckpt is not None:
        if os.path.isfile(args.resume_ckpt):
            args.resume_ckpt, args.resume_ckpt_filename = os.path.split(args.resume_ckpt)
        else:
            args.resume_ckpt_filename = "ckpt.pt"

        with open(os.path.join(args.resume_ckpt, "summary.json")) as f:
            summary = json.load(f)

        for k, v in summary['args'].items():
            if k == "config_format" and args.config_format is not None:
                continue
            if k not in ["device", "dtype"]:
                setattr(args, k, v)

    return config.parse_args_with_format(format=args.config_format, base_parser=parser, args=rem_args, namespace=args)


def patched_forward(self, x, pos_emb_closure, cache_context, start_index, mask=None, use_cache=False):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, C//self.n_head).transpose(1,2)
        k = k.view(B, T, self.n_head, C//self.n_head).transpose(1,2)
        v = v.view(B, T, self.n_head, C//self.n_head).transpose(1,2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / (k.size(-1)**0.5))
        mask_bool = torch.triu(torch.ones((T, T), device=x.device, dtype=torch.bool), diagonal=1)
        
        att = att.masked_fill(mask_bool, float('-inf'))
        att = F.softmax(att, dim=-1)
        self.last_attn = att.detach().cpu()
        out = att @ v
        out = out.transpose(1,2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(out))

def plot_token_cosine(model, data, args, layer_idx=-1, max_tokens=100):
    """
    1) Hooks into transformer.h[layer_idx] to grab its output hidden states.
    2) Runs one eval batch.
    3) Computes cosine similarity over the first max_tokens.
    4) Saves a heatmap.
    """
    # Register hook
    hidden = {}
    def hook_fn(_, __, out):
        hidden['hs'] = out.detach().cpu()  # (B, T, C)
    handle = model.transformer.h[layer_idx].register_forward_hook(hook_fn)

    model.eval()
    x, y = get_batch(data['val'], args.sequence_length, args.batch_size, device=args.device)
    with torch.no_grad():
        _ = model(x, targets=y)
    handle.remove()

    hs = hidden['hs']          # (B, T, C)
    h0 = hs[0, :max_tokens, :] # take first sample, first max_tokens
    
    # Cosine‐sim matrix
    sim = F.cosine_similarity(
        h0.unsqueeze(1),        # (T,1,C)
        h0.unsqueeze(0),        # (1,T,C)
        dim=-1                  # -> (T,T)
    )
    
    plt.figure(figsize=(8, 8))
    sns.heatmap(sim.numpy(), cmap="coolwarm", square=True,
                xticklabels=range(max_tokens), yticklabels=range(max_tokens))
    ax = plt.gca()

    # Show only every 10th tick
    n = max_tokens
    step = 10
    ax.set_xticks(np.arange(0, n, step))
    ax.set_yticks(np.arange(0, n, step))
    
    # Rotate + shrink labels
    ax.tick_params(axis='x', rotation=45, labelsize=8, pad=2)
    ax.tick_params(axis='y', rotation=0,  labelsize=8, pad=2)

    # Alternatively, use a MaxNLocator to auto‐decide ~10 ticks
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=10))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=10))

    # Then save
    plt.tight_layout(pad=2.0)
    plt.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.05)  # 手动调整边距
    plt.xlabel("Key token index", fontsize=14)
    plt.ylabel("Query token index", fontsize=14)
    plt.title(f"Token–Token Cosine Similarity\nLayer {layer_idx+1}, {max_tokens} tokens", fontsize=14, pad=12)
    os.makedirs(args.results_base_folder, exist_ok=True)
    plt.savefig(os.path.join(args.results_base_folder, f"token_cosine_layer{layer_idx+1}.pdf"), format='pdf')

def main(args, n_batches=1): 

    torch.backends.cuda.matmul.allow_tf32 = True # allows us to make sure we're able to use tensorfloat32 during training
    torch.backends.cudnn.allow_tf32 = True

    distributed_backend = distributed.make_backend_from_args(args)
    args = distributed_backend.get_adjusted_args_for_process(args)

    args.device = torch.device(args.device)
    torch.cuda.set_device(args.device)
    device_type = 'cuda' if 'cuda' in str(args.device) else 'cpu'
    
    print(f"Loading dataset from local files")
    
    train_data = np.memmap('/data2/ztwu/openwebtext/train.bin', dtype=np.uint16, mode='r')
    val_data = np.memmap('/data2/ztwu/openwebtext/val.bin', dtype=np.uint16, mode='r')
    data = {'train': train_data, 'val': val_data}
    
    print(f"Num training tokens: {len(data['train'])}")
    print(f"Num validation tokens: {len(data['val'])}")
    
    model = models.make_model_from_args(args).to(args.device) # todo: take care of initializing the model if args.use_pretrained != 'none'
    model = distributed_backend.transform_model(model)
    
    orig_forward = CausalSelfAttention.forward
    CausalSelfAttention.forward = patched_forward

    model.eval()
    n_layers = len(model.transformer.h)
    count = 0
    # For head similarity, accumulate mean attention per head per layer
    attention_mean = [None] * n_layers  # will store cumulative attention sum per head
    for l in range(n_layers):
        attention_mean[l] = None

    with torch.no_grad():
        for _ in tqdm(range(n_batches), desc="Processing Batches"):
            x, y = get_batch(data['val'], args.sequence_length, args.batch_size, device=args.device)
            _ = model(x, targets=y)
            # for each layer compute stats
            for l, blk in enumerate(model.transformer.h):
                attn = blk.attn.last_attn  # (B, nh, T, T)
                # accumulate mean attention per head
                head_mean = attn.mean(dim=0)  # (nh, T, T)
                if attention_mean[l] is None:
                    attention_mean[l] = head_mean.clone()
                else:
                    attention_mean[l] += head_mean
                del attn

            count += 1


    layers = np.arange(1, n_layers+1)
    os.makedirs(args.results_base_folder, exist_ok=True)

    # Head similarity heatmap for a selected layer (e.g., last layer)
    layer_idx = n_layers - 1
    layer_idx = 5
    head_attn = (attention_mean[layer_idx] / count).numpy()  # (nh, T, T)
    # Flatten each head to vector
    flat = head_attn.reshape(head_attn.shape[0], -1)  # (nh, T*T)
    sim = F.cosine_similarity(torch.tensor(flat).unsqueeze(1), torch.tensor(flat).unsqueeze(0), dim=-1)
    plt.figure(figsize=(6,5))
    #sns.heatmap(sim.numpy(), annot=True, cmap='coolwarm')
    sns.heatmap(sim.numpy(), cmap='coolwarm')
    plt.title(f'Head Similarity (Layer {layer_idx+1})')
    plt.savefig(os.path.join(args.results_base_folder, f'head_similarity_layer{layer_idx+1}.pdf'), format='pdf')

    # Restore original forward
    CausalSelfAttention.forward = orig_forward
    n_layers = len(model.transformer.h)
    plot_token_cosine(model, data, args,
                      layer_idx=n_layers-1, max_tokens=args.sequence_length)
    # plot_token_cosine(model, data, args,
    #                   layer_idx=6, max_tokens=args.sequence_length)


if __name__ == "__main__":
    args = get_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    print(f"Random seed set to: {args.seed}")
    main(args)