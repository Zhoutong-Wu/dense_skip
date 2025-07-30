import sys
from tqdm import tqdm
sys.path.append('/home/ztwu/lm-evaluation-harness')
import os
import sys
import numpy as np
import torch
import json
import argparse
import random
import logging

import config
import models

import torch.nn.functional as F
import lm_eval
from lm_eval.api.model import LM
from lm_eval.utils import setup_logging
from lm_eval import utils
from lm_eval.models.utils import (
    stop_sequences_criteria,
)

class Your_LM(LM):
    def __init__(self, model, batch_size=1):
        super().__init__()
        self.model = model  # Your model object
        self.model.eval()
        self.batch_size = batch_size
        self.tokenizer = model.tokenizer  # Get tokenizer from model
        self.lm_head = model.lm_head      # Get lm_head from model

        self.pad_token_id = 0  # Default padding token id
        
    def loglikelihood(self, requests):
        results = []
        tokenized_requests = []
        
        # Convert requests to tokens
        for instance in requests:
            context, continuation = instance.arguments[:2]
            ctx_tokens = self.tokenizer.encode(context)
            cont_tokens = self.tokenizer.encode(continuation)
            tokenized_requests.append((ctx_tokens, cont_tokens))

        # Batch processing
        for i in range(0, len(tokenized_requests), self.batch_size):
            batch = tokenized_requests[i:i+self.batch_size]
            ctx_lens = [len(ctx) for ctx, _ in batch]
            cont_lens = [len(cont) for _, cont in batch]

            # Merge context and continuation, pad to max length
            input_ids = [ctx + cont for ctx, cont in batch]
            max_len = max(len(ids) for ids in input_ids)
            input_ids_padded = torch.tensor([
                ids + [self.pad_token_id] * (max_len - len(ids)) 
                for ids in input_ids
            ]).to(self.lm_head.weight.device)

            # Build targets and attention mask
            targets = []
            attention_mask = []
            for ctx, cont in batch:
                total_len = len(ctx) + len(cont)
                tgt = [-1] * max_len
                tgt[len(ctx)-1 : len(ctx)+len(cont)-1] = cont
                targets.append(tgt)
                mask = [1] * total_len + [0] * (max_len - total_len)
                attention_mask.append(mask)

            targets = torch.tensor(targets).to(self.lm_head.weight.device)
            attention_mask = torch.tensor(attention_mask).to(self.lm_head.weight.device)

            # Forward pass
            self.model.eval()
            with torch.no_grad():
                outputs = self.model.forward(
                    idx=input_ids_padded, 
                    targets=targets, 
                    get_logits=True,
                    mask=attention_mask
                )
                logits = outputs['logits']

            # Compute logprobs
            for j, (ctx_tokens, cont_tokens) in enumerate(batch):
                ctx_len, cont_len = len(ctx_tokens), len(cont_tokens)
                start_idx = ctx_len - 1
                end_idx = start_idx + cont_len
                logits_j = logits[j, start_idx:end_idx, :]
                log_probs = F.log_softmax(logits_j, dim=-1)
                target_ids = torch.tensor(cont_tokens).to(logits_j.device)
                selected_log_probs = log_probs.gather(1, target_ids.unsqueeze(-1)).squeeze()
                loglikelihood = selected_log_probs.sum().item()
                greedy_tokens = logits_j.argmax(dim=-1)
                is_greedy = (greedy_tokens == target_ids).all().item()
                results.append((loglikelihood, is_greedy))
        
        return results

    def loglikelihood_rolling(self, requests):
        """
        Compute rolling loglikelihood for a list of requests.
        """
        results = []

        for instance in tqdm(requests, desc="Processing loglikelihood_rolling requests"):
            text = instance.arguments[0]
            tokens = self.tokenizer.encode(text, allowed_special={"<|endoftext|>"})

            # Rolling window logic
            rolling_token_windows = list(
                map(
                    lambda x: (None, x[0], x[1]),
                    utils.get_rolling_token_windows(
                        token_list=tokens,
                        prefix_token=self.tokenizer.bos_token_id,
                        max_seq_len=self.model.config.sequence_length,
                        context_len=1,
                    ),
                )
            )

            rolling_loss = 0.0
            for _, context_tokens, continuation_tokens in rolling_token_windows:
                input_ids = torch.tensor(context_tokens + continuation_tokens).unsqueeze(0).to(self.lm_head.weight.device)
                targets = torch.tensor([-1] * len(context_tokens) + continuation_tokens).unsqueeze(0).to(self.lm_head.weight.device)

                # Forward pass
                output = self.model.forward(idx=input_ids, targets=targets, get_logits=True)
                logits = output['logits']

                # Compute loglikelihood
                continuation_ids = torch.tensor(continuation_tokens, device=self.lm_head.weight.device).unsqueeze(0)
                logits_for_targets = logits[:, -len(continuation_tokens):, :]
                log_probs = torch.nn.functional.log_softmax(logits_for_targets, dim=-1)
                selected_log_probs = torch.gather(log_probs, 2, continuation_ids.unsqueeze(-1)).squeeze(-1)
                rolling_loss += selected_log_probs.sum().item()

            results.append(rolling_loss)

        return results

    def generate_until(self, requests):
        """
        Generate text until stop sequences are encountered.
        """
        results = []

        for instance in tqdm(requests, desc="Processing generate_until requests"):
            context, generation_kwargs = instance.arguments[:2]
            context_tokens = self.tokenizer.encode(context, allowed_special={"<|endoftext|>"})

            input_ids = torch.tensor(context_tokens, device=self.lm_head.weight.device).unsqueeze(0)

            generation_kwargs.setdefault("max_length", 256)
            stop_tokens = generation_kwargs.get("stop", [])

            output_ids = self.model.generate(
                input_ids=input_ids,
                max_length=generation_kwargs.get("max_length", 256),
                stopping_criteria=stop_sequences_criteria(
                    self.tokenizer,
                    generation_kwargs.get("stop", []),
                    input_ids.shape[1],
                    input_ids.shape[0],
                ),
                **generation_kwargs,
            )

            output_text = self.tokenizer.decode(output_ids[0].tolist(), skip_special_tokens=True)

            for stop_token in stop_tokens:
                if stop_token in output_text:
                    output_text = output_text.split(stop_token)[0]
                    break

            results.append(output_text)

        return results
    
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
            if k not in ["device", "dtype", "resume_ckpt"]:  # 排除 resume_ckpt
               setattr(args, k, v)

    return config.parse_args_with_format(format=args.config_format, base_parser=parser, args=rem_args, namespace=args)

def main(args):
    # initialize logging
    setup_logging("ERROR")
    model = models.make_model_from_args(args).to(args.device) # todo: take care of initializing the model if args.use_pretrained != 'none' 
    checkpoint = torch.load(os.path.join(args.resume_ckpt, args.resume_ckpt_filename), map_location=args.device)
    model.load_state_dict(checkpoint['model'], strict=False)
    print(f"Load state_dict from checkpoint {args.resume_ckpt}")
    
    # create your model (could be running finetuning with some custom modeling code)
    # instantiate an LM subclass that takes your initialized model and can run
    # - `Your_LM.loglikelihood()`
    # - `Your_LM.loglikelihood_rolling()`
    # - `Your_LM.generate_until()`
    
    model.eval()
    
    lm_obj = Your_LM(model=model, batch_size=16)

    # indexes all tasks from the `lm_eval/tasks` subdirectory.
    # Alternatively, you can set `TaskManager(include_path="path/to/my/custom/task/configs")`
    # to include a set of tasks in a separate directory.
    
    task_manager = lm_eval.tasks.TaskManager()

    # Setting `task_manager` to the one above is optional and should generally be done
    # if you want to include tasks from paths other than ones in `lm_eval/tasks`.
    # `simple_evaluate` will instantiate its own task_manager if it is set to None here.
    results = lm_eval.simple_evaluate( # call simple_evaluate
        model=lm_obj,
        tasks=["hellaswag", "winogrande", "arc_challenge", "arc_easy", "piqa", "openbookqa", "sciq"],
        num_fewshot=0,
        task_manager=task_manager,
        bootstrap_iters=0,
    )
    
    print(results["results"])
    
if __name__ == "__main__":
    args = get_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    print(f"Random seed set to: {args.seed}")
    main(args)