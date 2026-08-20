from cs336_basics.BPETokenizer.BPEtokenizer import BPEtokenizer
from cs336_basics.Transformer import TransformerModules
from cs336_basics.utils import utils

import torch
from torch import Tensor
import numpy as np
import os
import logging


def setup_train_logger(name: str, log_path: str) -> logging.Logger:
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

def setup_eval_logger(name: str, log_path: str) -> logging.Logger:
    logger = logging.getLogger(f"cs336.eval.{name}")
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


def evaluate_model(
    model: torch.nn.Module,
    valid_arr: np.ndarray,
    num_samples: int,
    batch_size: int,
    context_length: int,
    device: torch.device | str | None,
    seed: int = 336,
) -> float:
    """Evaluate an in-memory model without changing training RNG or mode."""
    batch_rng_state = np.random.get_state()
    was_training = model.training
    try:
        np.random.seed(seed)
        model.eval()
        loss_sum = 0.0
        with torch.inference_mode():
            for _ in range(num_samples):
                x, y = utils.get_batch(valid_arr, batch_size, context_length, device)
                logits = model(x)
                loss_sum += TransformerModules.cross_entropy(logits, y).mean().item()
        return loss_sum / num_samples
    finally:
        np.random.set_state(batch_rng_state)
        model.train(was_training)

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
    dtype: torch.dtype | None = None,
    seed: int = 42,
    valid_path: str | None = "data/tokenized/ts_valid_tokenized.npy",
    eval_num_samples: int = 64,
    eval_batch_size: int = 32,
    eval_seed: int = 336,
):
    np.random.seed(seed)
    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, f"{name}.log")
    if os.path.exists(log_path):
        s = input(f"Path {log_path} already exists, continue? [Y/n]\n")
        while s.upper() != "Y" and s.upper() != "N":
            s = input(f"Path {log_path} already exists, continue? [Y/n]\n")
        if s.upper() == "N":
            return
    logger = setup_train_logger(name, log_path)
    logger.info(
        f"Start training {name}: save_dir={save_dir}, load_path={load_path}, "
        f"init_ckpoint={init_ckpoint}, vocab_size={vocab_size}, context_length={context_length}, "
        f"d_model={d_model}, d_ff={d_ff}, num_layers={num_layers}, num_heads={num_heads}, "
        f"theta={theta}, batch_size={batch_size}, max_iterations={max_iterations}, "
        f"alpha_max={alpha_max}, alpha_min={alpha_min}, T_w={T_w}, T_c={T_c}, "
        f"betas={betas}, eps={eps}, weight_decay={weight_decay}, "
        f"max_l2_norm={max_l2_norm}, device={device}, dtype={dtype}, seed={seed}, "
        f"valid_path={valid_path}, eval_num_samples={eval_num_samples}, "
        f"eval_batch_size={eval_batch_size}, eval_seed={eval_seed}"
    )
    token_arr = utils.get_mmap(load_path)
    valid_arr = utils.get_mmap(valid_path) if valid_path is not None else None
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
            if valid_arr is not None:
                eval_loss = evaluate_model(
                    model=model,
                    valid_arr=valid_arr,
                    num_samples=eval_num_samples,
                    batch_size=eval_batch_size,
                    context_length=context_length,
                    device=device,
                    seed=eval_seed,
                )
                logger.info(
                    "iteration=%d/%d | eval_loss=%.6f | eval_seed=%d",
                    t,
                    max_iterations,
                    eval_loss,
                    eval_seed,
                )

    logger.info("Training complete at iteration=%d", max_iterations)
        

def infer_bruteforce(
    input: str,
    tokenizer: BPEtokenizer,
    model: TransformerModules.TransformerLM,
    max_tokens: int = 1024,
    temperature: float = 0.1,
    top_p: float = 0.9,
    device: torch.device = None
):
    if temperature < 0.0:
        raise ValueError(f"Error temperature: {temperature}.")
    if top_p <= 0.0 or top_p > 1.0:
        raise ValueError(f"Error top_p: {top_p}.")
    input_tokens = tokenizer.encode(input)
    model.eval()
    with torch.inference_mode():
        while len(input_tokens) < max_tokens:
            x = torch.tensor(
                input_tokens[-model.context_length:],
                dtype=torch.long,
                device=device,
            ).unsqueeze(0)
            logits = model(x)[0, -1]
            
            if temperature == 0.0:
                next_token = torch.argmax(logits)
            else:
                probs = TransformerModules.softmax(logits / temperature)

                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

                # Keep the smallest set whose cumulative probability reaches top_p.
                keep = cumulative_probs - sorted_probs < top_p
                filtered_probs = sorted_probs * keep
                filtered_probs /= filtered_probs.sum()

                sampled_index = torch.multinomial(filtered_probs, num_samples=1)
                next_token = sorted_indices[sampled_index].squeeze(0)
            input_tokens.append(next_token.item())
            if next_token.item() == 256: break # special token

    return tokenizer.decode(input_tokens)

def eval(
    valid_path: str,
    model_path: str,
    num_samples: int = 64,
    vocab_size: int = 10000,
    context_length: int = 256,
    d_model: int = 512,
    d_ff: int = 1344,
    theta: float = 10000,
    num_layers: int = 4,
    num_heads: int = 16,
    batch_size: int = 32,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    log_path: str | None = None,
    seed: int = 336
):
    if log_path is None:
        model_path_without_suffix, _ = os.path.splitext(model_path)
        log_path = f"{model_path_without_suffix}_eval.log"
    log_dir = os.path.dirname(log_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    logger = setup_eval_logger(os.path.basename(model_path), log_path)
    logger.info(
        "Start evaluation: valid_path=%s, model_path=%s, num_samples=%d, "
        "batch_size=%d, context_length=%d, device=%s, dtype=%s",
        valid_path,
        model_path,
        num_samples,
        batch_size,
        context_length,
        device,
        dtype,
    )

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
    utils.load_checkpoint(model_path, model, map_location=device)
    arr = utils.get_mmap(valid_path)
    loss_mean = evaluate_model(
        model=model,
        valid_arr=arr,
        num_samples=num_samples,
        batch_size=batch_size,
        context_length=context_length,
        device=device,
        seed=seed,
    )
    logger.info("Evaluation complete | mean_loss=%.6f", loss_mean)
    return loss_mean

if __name__ == "__main__":
    # train ---------
    load_path = "data/tokenized/ts_train_tokenized.npy"
    save_dir = "model/ts"

    train(
        name="tiny_stories_64_20000_095",
        save_dir=save_dir,
        load_path=load_path,
        iterations_per_ckpoint=4000,
        batch_size=64,
        max_iterations=20000,
        T_c=20000,
        device="cuda:2",
        betas=(0.9, 0.95)
    )


    # generate ----------
    # device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    # print(f"Using device: {device}")

    # tokenizer = BPEtokenizer("cs336_basics/BPETokenizer/trained_data/ts_train_10000.pkl")
    # model = TransformerModules.TransformerLM(10000, 4, 512, 16, 1344, 10000, 256)
    # # optimizer = TransformerModules.AdamW(model.parameters())
    # utils.load_checkpoint(
    #     "model/ts/tiny_stories_official_20000_20000.pt",
    #     model,
    #     map_location=device,
    # )
    # model.to(device)
    # s = infer_bruteforce(
    #     "Once upon a time, there was a boy named Leon.",
    #     tokenizer,
    #     model,
    #     max_tokens=256,
    #     temperature=1.0,
    #     device=device,
    # )
    # print(s)

    # eval ----
    # eval(
    #     "data/tokenized/ts_valid_tokenized.npy",
    #     "model/ts/tiny_stories_official_10000_128_10000.pt",
    #     # device="mps"
    #     device="cuda:4"
    # )
