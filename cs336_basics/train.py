from cs336_basics.BPETokenizer.BPEtokenizer import BPEtokenizer
from cs336_basics.Transformer import TransformerModules
from cs336_basics.utils import utils

import torch
from torch import Tensor
import numpy as np
import os
import logging


def setup_logger(name: str, log_path: str) -> logging.Logger:
    logger = logging.getLogger(f"cs336.train.{name}")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    for handler in logger.handlers:
        handler.close()
    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger


def train(
    name: str,
    save_dir: str,
    load_path: str,
    init_ckpoint: str | None = None,
    vocab_size: int = 10000,
    context_length: int = 256,
    d_model: int = 512,
    d_ff: int = 1344,
    theta: float = 10000,
    alpha_max: float = 3e-4,
    alpha_min: float = 3e-5,
    T_w: int = 500,
    T_c: int = 5000,
    betas: float = (0.9, 0.999),
    eps: float = 1e-8,
    weight_decay: float = 0.01,
    num_layers: int = 4,
    num_heads: int = 16,
    batch_size: int = 32,
    max_iterations: int = 5000,
    iterations_per_log: int = 10,
    iterations_per_ckpoint: int = 500,
    max_l2_norm: float = 1.0,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None
):
    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, f"{name}.log")
    if os.path.exists(log_path):
        s = input(f"Path {log_path} already exists, continue? [Y/n]\n")
        while s.upper() != "Y" and s.upper() != "N":
            s = input(f"Path {log_path} already exists, continue? [Y/n]\n")
        if s.upper() == "N":
            return
    logger = setup_logger(name, log_path)
    logger.info(
        f"Start training {name}: save_dir={save_dir}, load_path={load_path}, "
        f"init_ckpoint={init_ckpoint}, vocab_size={vocab_size}, context_length={context_length}, "
        f"d_model={d_model}, d_ff={d_ff}, num_layers={num_layers}, num_heads={num_heads}, "
        f"theta={theta}, batch_size={batch_size}, max_iterations={max_iterations}, "
        f"alpha_max={alpha_max}, alpha_min={alpha_min}, T_w={T_w}, T_c={T_c}, "
        f"betas={betas}, eps={eps}, weight_decay={weight_decay}, "
        f"max_l2_norm={max_l2_norm}, device={device}, dtype={dtype}"
    )
    token_arr = utils.get_mmap(load_path)
    model = TransformerModules.TransformerLM(
        vocab_size=vocab_size,
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        theta=theta,
        context_length=context_length,
        device=device,
        dtype=dtype
    )
    optimizer = TransformerModules.AdamW(
        params=model.parameters(),
        lr=alpha_max,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay
    )
    start_iteration = 1
    if init_ckpoint is not None:
        start_iteration = utils.load_checkpoint(init_ckpoint, model, optimizer) + 1
        logger.info("Resumed checkpoint=%s at iteration=%d", init_ckpoint, start_iteration)
    
    model.train()
    
    for t in range(start_iteration, max_iterations + 1):
        x, y = utils.get_batch(token_arr, batch_size=batch_size, context_length=context_length, device_str=device)
        optimizer.zero_grad() # 清空上次梯度，每次梯度会累加
        logits = model(x)
        loss = TransformerModules.cross_entropy(logits, y).mean()
        loss.backward()
        for group in optimizer.param_groups:
            group["lr"] = TransformerModules.lr_cos_schedule(t, alpha_max, alpha_min, T_w, T_c)
        TransformerModules.clip_gradient(model.parameters(), max_l2_norm)
        optimizer.step()

        if t % iterations_per_log == 0 or t == start_iteration:
            logger.info(
                "iteration=%d/%d | loss=%.6f | lr=%.6g",
                t,
                max_iterations,
                loss.item(),
                optimizer.param_groups[0]["lr"],
            )
        
        if t % iterations_per_ckpoint == 0:
            checkpoint_path = os.path.join(save_dir, f"{name}_{t}.pt")
            utils.save_checkpoint(model, optimizer, t, checkpoint_path)
            logger.info("Saved checkpoint=%s", checkpoint_path)

    logger.info("Training complete at iteration=%d", max_iterations)
        

if __name__ == "__main__":
    load_path = "data/tokenized/ts_train_tokenized.npy"
    save_dir = "model/ts"

    train(
        name="tiny_stories",
        save_dir=save_dir,
        load_path = load_path,
    )
