from cs336_basics.BPETokenizer.BPEtokenizer import BPEtokenizer
from cs336_basics.Transformer import TransformerModules
from cs336_basics.utils import utils

import torch
from torch import Tensor
import numpy as np
import os
import logging
import re
from pathlib import Path


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
    shared_embedding: bool = False,
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
    # calc resource
    # embedding -> attention blocks -> out embed -> rms norm -> softmax -> cross entropy -> loss
    params_embed = vocab_size * d_model * (1 if shared_embedding else 2)
    params_attn = num_layers * (4 * d_model * d_model + 3 * d_model * d_ff + 2 * d_model)
    params_rms = d_model
    total_params = params_embed + params_attn + params_rms
    print(f"total params: {total_params}, approx {total_params / 1e6:.2f} M")
    
    act_embed = batch_size * context_length * d_model
    act_attn = batch_size * num_layers * (context_length * d_model * 2 + context_length * d_model * 3 + num_heads * context_length * context_length * 2 + context_length * d_model + context_length * d_ff * 2 + context_length * d_model)
    act_aft = batch_size * context_length * (vocab_size * 4)
    total_act = act_embed + act_attn + act_aft
    print(f"activations total: {total_act}, approx {total_act / 1e6:.2f} M")
    
    total_bytes = 4 * (total_params * 4 + total_act)
    print(f"total bytes aprox: {total_bytes}, or {total_bytes / 1e9:.2f} GB")
    # if total_bytes > 20_000_000_000:
    if total_bytes > 10:
        s = input(f"Very Large Size, Continue? [Y/n]\n")
        while s.upper() != "Y" and s.upper() != "N":
            s = input(f"Very Large Size, Continue? [Y/n]\n")
        if s.upper() == "N":
            return
    
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
        shared_embedding=shared_embedding,
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
    try:
        from plot_loss import plot_losses

        plot_path = plot_losses(Path(log_path))
        logger.info("Saved loss plot=%s", plot_path)
    except Exception:
        logger.exception("Failed to generate loss plot from %s", log_path)
        

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
    seed: int = 336,
    plot_path: str | None = None,
    smooth_window: int = 10,
) -> dict[int, float]:
    model_path_obj = Path(model_path)
    if model_path_obj.suffix == ".pt":
        checkpoint_match = re.fullmatch(r"(.+)_(\d+)", model_path_obj.stem)
        if checkpoint_match is None:
            raise ValueError(f"Cannot infer run name from checkpoint: {model_path}")
        run_prefix = model_path_obj.with_name(checkpoint_match.group(1))
    elif model_path_obj.suffix == ".log":
        run_prefix = model_path_obj.with_suffix("")
    else:
        run_prefix = model_path_obj

    checkpoint_pattern = re.compile(rf"^{re.escape(run_prefix.name)}_(\d+)\.pt$")
    checkpoints: list[tuple[int, Path]] = []
    for checkpoint_path in run_prefix.parent.glob(f"{run_prefix.name}_*.pt"):
        match = checkpoint_pattern.fullmatch(checkpoint_path.name)
        if match:
            checkpoints.append((int(match.group(1)), checkpoint_path))
    checkpoints.sort()
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found for run prefix: {run_prefix}")

    if log_path is None:
        log_path = str(run_prefix.with_suffix(".log"))
    log_dir = os.path.dirname(log_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    logger = setup_eval_logger(run_prefix.name, log_path)
    logger.info(
        "Start run evaluation: valid_path=%s, run_prefix=%s, checkpoints=%d, "
        "num_samples=%d, batch_size=%d, context_length=%d, seed=%d, device=%s, dtype=%s",
        valid_path,
        run_prefix,
        len(checkpoints),
        num_samples,
        batch_size,
        context_length,
        seed,
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
    arr = utils.get_mmap(valid_path)
    caller_rng_state = np.random.get_state()
    results: dict[int, float] = {}
    try:
        for iteration, checkpoint_path in checkpoints:
            utils.load_checkpoint(checkpoint_path, model, map_location=device)
            loss_mean = evaluate_model(
                model=model,
                valid_arr=arr,
                num_samples=num_samples,
                batch_size=batch_size,
                context_length=context_length,
                device=device,
                seed=seed,
            )
            results[iteration] = loss_mean
            logger.info(
                "iteration=%d | eval_loss=%.6f | eval_seed=%d | checkpoint=%s",
                iteration,
                loss_mean,
                seed,
                checkpoint_path,
            )
    finally:
        np.random.set_state(caller_rng_state)

    from plot_loss import plot_losses

    output = plot_losses(
        log_path=Path(log_path),
        output=Path(plot_path) if plot_path is not None else None,
        smooth_window=smooth_window,
    )
    logger.info("Run evaluation complete | checkpoints=%d | plot=%s", len(results), output)
    return results

if __name__ == "__main__":
    # train ---------
    # load_path = "data/tokenized/ts_train_tokenized.npy"
    # save_dir = "model/ts"
    load_path = "data/tokenized/owt_train_tokenized.npy"
    save_dir = "model/owt"
    
    # train(
    #     name="ts_shared_40000_32",
    #     save_dir=save_dir,
    #     load_path=load_path,
    #     vocab_size=10000,
    #     iterations_per_ckpoint=4000,
    #     batch_size=32,
    #     max_iterations=40000,
    #     T_c=40000,
    #     shared_embedding=True,
    #     device="cuda:5",
    #     valid_path="data/tokenized/ts_valid_tokenized.npy"
    #     # betas=(0.9, 0.95)
    # )
    
    # train(
    #     name="ts_test_shared",
    #     save_dir=save_dir,
    #     load_path=load_path,
    #     vocab_size=10000,
    #     iterations_per_ckpoint=10,
    #     batch_size=64,
    #     max_iterations=20,
    #     T_c=20,
    #     shared_embedding=True,
    #     device="cuda:4",
    #     valid_path="data/tokenized/ts_valid_tokenized.npy"
    #     # betas=(0.9, 0.95)
    # )

    # train(
    #     name="owt_shared_layer_8_40000_32",
    #     save_dir=save_dir,
    #     load_path=load_path,
    #     vocab_size=32000,
    #     iterations_per_ckpoint=4000,
    #     batch_size=32,
    #     max_iterations=40000,
    #     T_c=40000,
    #     num_layers=8,
    #     shared_embedding=True,
    #     device="cuda:7",
    #     valid_path="data/tokenized/owt_valid_tokenized.npy"
    #     # betas=(0.9, 0.95)
    # )


    # generate ----------
    # device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = BPEtokenizer("cs336_basics/BPETokenizer/trained_data/ts_train_10000.pkl")
    # tokenizer = BPEtokenizer("cs336_basics/BPETokenizer/trained_data/owt_train_32000.pkl")
    model = TransformerModules.TransformerLM(10000, 4, 512, 16, 1344, 10000, 256, shared_embedding=True)
    # model = TransformerModules.TransformerLM(10000, 4, 512, 16, 1344, 10000, 256, shared_embedding=False)
    # model = TransformerModules.TransformerLM(32000, 4, 512, 16, 1344, 10000, 256)
    # optimizer = TransformerModules.AdamW(model.parameters())
    utils.load_checkpoint(
        "model/ts/ts_shared_20000_64_20000.pt",
        # "model/owt/owt_40000_32_40000.pt",
        model,
        map_location=device,
    )
    model.to(device)
    s = infer_bruteforce(
        "Once upon a time, there was a boy named Leon.",
        tokenizer,
        model,
        max_tokens=256,
        temperature=0.0,
        device=device,
    )
    print(s)
    s = infer_bruteforce(
        "The Chinese translation of \"good\" is:",
        tokenizer,
        model,
        max_tokens=256,
        temperature=0.0,
        device=device,
    )
    s = infer_bruteforce(
        "<|endoftext|>",
        tokenizer,
        model,
        max_tokens=256,
        temperature=0.0,
        device=device,
    )
    print(s)

    # eval ----
    # eval(
    #     "data/tokenized/ts_valid_tokenized.npy",
    #     "model/ts/tiny_stories_official_20000_20000.pt",
    #     # device="mps"
    #     device="cuda:4"
    # )

    # arr = utils.get_mmap("data/tokenized/owt_train_tokenized.npy")
    # print(len(arr))