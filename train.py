import os
import math
import time
import glob
import random
from contextlib import nullcontext
import wandb

import numpy as np
import torch
import torch.nn.functional as F

from config import config as arch_config
from model import Transformer

# -----------------------------------------------------------------------------
# I/O
out_dir = "out"
init_from = "resume"  # 'scratch' | 'resume'
eval_interval = 500  # evaluate val loss every N iters
log_interval = 10  # print train loss every N iters
eval_iters = 100  # batches averaged per eval
always_save_checkpoint = True  # save even if val loss did not improve

# Logging
wandb_log = True
wandb_project = "pytorch-transformer"
wandb_run_name = "run1"

# data
dataset = arch_config["dataset_prefix"]
gradient_accumulation_steps = 8  # simulate larger batch via micro-steps
batch_size = 48  # micro-batch size per accumulation step
block_size = arch_config["max_seq_len"]

# model (read from arch_config so it stays in sync with the actual model)
vocab_size = arch_config["vocab_size"]

# adamw optimizer
max_lr = 6e-4  # keep this, no warmup needed mid-run
min_lr = 6e-5
max_iters = 150000
warmup_iters = 0  # you're already warmed up
lr_decay_iters = 150000  # = 70000 steps remaining #W#########################################################################################################
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0  # 0.0 disables clipping

# system
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = (
    "bfloat16"
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else "float32"
)
# -----------------------------------------------------------------------------

os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = "cuda" if "cuda" in device else "cpu"
ptdtype = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}[dtype]
ctx = (
    nullcontext()
    if device_type == "cpu"
    else torch.autocast(device_type=device_type, dtype=ptdtype)
)

# Discover shards written by prepare.py
data_dir = os.path.join("data", dataset)
train_shards = sorted(glob.glob(os.path.join(data_dir, "train_*.bin")))
val_shards = sorted(glob.glob(os.path.join(data_dir, "val_*.bin")))
assert train_shards, f"No train shards found in {data_dir}. Run prepare.py first."
assert val_shards, f"No val shards found in {data_dir}. Run prepare.py first."

print("─" * 52)
print(
    f"  model  │ N={arch_config['N']}, d_model={arch_config['d_model']}, h={arch_config['h']}, d_ff={arch_config['d_ff']}"
)
print(
    f"  data   │ block={block_size}, batch={batch_size}×{gradient_accumulation_steps} accum → {batch_size*gradient_accumulation_steps} eff, tokens/iter={gradient_accumulation_steps*batch_size*block_size:,}"
)
print(f"  optim  │ lr {max_lr}→{min_lr}, warmup={warmup_iters}, iters={max_iters:,}")
print(f"  system │ {device} | {dtype} | {len(train_shards)} train shard(s)")
print("─" * 52)


# -----------------------------------------------------------------------------
# Data loading


def get_batch(split: str):
    shards = train_shards if split == "train" else val_shards
    # Recreate memmap each call — avoids a numpy memory-leak with long-running loops
    data = np.memmap(random.choice(shards), dtype=np.uint16, mode="r")
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack(
        [torch.from_numpy(data[i : i + block_size].astype(np.int64)) for i in ix]
    )
    y = torch.stack(
        [
            torch.from_numpy(data[i + 1 : i + block_size + 1].astype(np.int64))
            for i in ix
        ]
    )
    if device_type == "cuda":
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(
            device, non_blocking=True
        )
    else:
        x, y = x.to(device), y.to(device)
    return x, y


# -----------------------------------------------------------------------------
# Learning rate schedule: linear warmup → cosine decay → floor


# https://www.desmos.com/calculator/x14yko4tce
def get_lr(it: int) -> float:
    if it < warmup_iters:
        return max_lr * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


# -----------------------------------------------------------------------------
# Loss estimation over many random batches for a stable reading


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits = model(X)
                loss = F.cross_entropy(logits.view(-1, vocab_size), Y.view(-1))
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# -----------------------------------------------------------------------------
# Model init

# Architecture params captured here so they're snapshotted into the checkpoint.
# On resume we rebuild from the checkpoint's model_args — not the current
# config.py — so old experiments load correctly even if config has changed.
model_args = {
    "vocab_size": arch_config["vocab_size"],
    "d_model": arch_config["d_model"],
    "d_ff": arch_config["d_ff"],
    "h": arch_config["h"],
    "N": arch_config["N"],
    "max_seq_len": arch_config["max_seq_len"],
}

iter_num = 0
best_val_loss = 1e9
checkpoint = None

if init_from == "resume":
    ckpt_path = os.path.join(out_dir, "ckpt.pt")
    assert os.path.exists(ckpt_path), f"No checkpoint found at {ckpt_path}"
    print(f"Resuming from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    model_args = checkpoint["model_args"]  # use saved arch, not current config.py
    model = Transformer(**model_args)
    model.load_state_dict(checkpoint["model"])
    iter_num = checkpoint["iter_num"]
    best_val_loss = checkpoint["best_val_loss"]
else:
    print("Initializing model from scratch")
    model = Transformer(**model_args)

model.to(device)
print(f"params={sum(p.numel() for p in model.parameters())}")
print(f"Starting at iter {iter_num}")

scaler = torch.amp.GradScaler(enabled=(dtype == "float16"))  # type: ignore[attr-defined]
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=max_lr,
    betas=(beta1, beta2),
    weight_decay=weight_decay,
)
if checkpoint is not None and "optimizer" in checkpoint:
    optimizer.load_state_dict(checkpoint["optimizer"])
checkpoint = None  # free memory

# -----------------------------------------------------------------------------
# Training loop

X, Y = get_batch("train")
t0 = time.time()

# Run logging
if wandb_log:
    wandb.init(
        project=wandb_project,
        name=wandb_run_name,
        config=model_args,
        resume="allow",
        id="9gnvj1e7",
    )

while True:
    # Set LR for this iteration
    lr = get_lr(iter_num)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    # Evaluate and checkpoint
    if iter_num % eval_interval == 0:
        losses = estimate_loss()
        print(
            f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
        )

        if wandb_log:
            wandb.log({"val/loss": losses["val"]}, step=iter_num)

        if losses["val"] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses["val"]
            if iter_num > 0:
                ckpt = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "iter_num": iter_num,
                    "best_val_loss": best_val_loss,
                    "model_args": model_args,
                    "config": {
                        "batch_size": batch_size,
                        "block_size": block_size,
                        "gradient_accumulation_steps": gradient_accumulation_steps,
                        "max_lr": max_lr,
                        "min_lr": min_lr,
                        "warmup_iters": warmup_iters,
                        "max_iters": max_iters,
                    },
                }
                print(f"saving checkpoint to {out_dir}/ckpt.pt")
                torch.save(ckpt, os.path.join(out_dir, "ckpt.pt"))

    # Forward / backward with gradient accumulation
    loss = torch.tensor(0.0)
    for micro_step in range(gradient_accumulation_steps):
        with ctx:
            logits = model(X)
            loss = F.cross_entropy(logits.view(-1, vocab_size), Y.view(-1))
            loss = loss / gradient_accumulation_steps
        # Prefetch next batch while GPU runs the backward pass
        X, Y = get_batch("train")
        scaler.scale(loss).backward()

    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0:
        # Undo the accumulation division to report the true batch loss
        lossf = loss.item() * gradient_accumulation_steps
        tok_per_sec = (gradient_accumulation_steps * batch_size * block_size) / dt
        print(
            f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, lr {lr:.2e}, tok_per_sec {(tok_per_sec / 1e3):.2f}k"
        )
        wandb.log(
            {"train/loss": lossf, "lr": lr, "tok_per_sec": tok_per_sec}, step=iter_num
        )

    iter_num += 1
    if iter_num >= max_iters:
        break
