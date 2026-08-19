import torch
import einops
from jaxtyping import Float, Int, Bool
from torch import Tensor
import math
from collections.abc import Callable, Iterable
from typing import Optional

class Init:
    @staticmethod
    def linear(
        d_in: int, 
        d_out: int, 
        device: torch.dtype | None = None, 
        dtype: torch.device | None = None
    ) -> torch.nn.Parameter:
        
        x = torch.nn.Parameter(
            torch.empty(
                d_in,
                d_out,
                device=device,
                dtype=dtype,
            )
        )
        std = math.sqrt(2 / (d_in + d_out))
        torch.nn.init.trunc_normal_(x, 0, std, -3 * std, 3 * std)
        return x
    
    @staticmethod
    def embedding(
        d_in: int, 
        d_out: int, 
        device: torch.dtype | None = None, 
        dtype: torch.device | None = None
    ) -> torch.nn.Parameter:
        
        x = torch.nn.Parameter(
            torch.empty(
                d_in,
                d_out,
                device=device,
                dtype=dtype,
            )
        )
        torch.nn.init.trunc_normal_(x, 0, 1, -3, 3)
        return x
    
    @staticmethod
    def rmsnorm(
        d_model: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        return torch.nn.Parameter(
            torch.ones(
                d_model,
                device=device,
                dtype=dtype
            )
        )

class Linear(torch.nn.Module):
    in_features: int
    out_features: int
    weights: Float[Tensor, "d_out d_in"]
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.weights = Init.linear(out_features, in_features, device=device, dtype=dtype)
        
    def forward(self, x: Tensor) -> Tensor:
        # return x @ W^T
        return einops.einsum(x, self.weights, "... d_in, d_out d_in -> ... d_out")

class Embedding(torch.nn.Module):
    
    embed_map: Float[Tensor, "vocab_size d_model"]
    num_embeddings: int
    embedding_dim: int
    
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.embed_map = Init.embedding(num_embeddings, embedding_dim, device=device, dtype=dtype)
        
    def forward(self, token_ids: Int[Tensor, "..."]) -> Float[Tensor, "... d_model"]:
        return self.embed_map[token_ids]

class RMSNorm(torch.nn.Module):
    
    # out = a_i / sqrt(sum(a_i^2) + eps) * g_i, where a_i is activations and g_i are learnable
    # RMSNorm happens within a token
    d_model: int
    eps: float
    weights: Float[Tensor, "d_model"]
    
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.device | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weights = Init.rmsnorm(d_model, device=device, dtype=dtype)
    
    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        in_dtype = x.dtype
        x = x.to(dtype=torch.float32)
        out = x / torch.sqrt(torch.mean(x.square(), dim=-1, keepdim=True) + self.eps) * self.weights
        return out.to(dtype=in_dtype)
        
class SwiGLUFFN(torch.nn.Module):
    # SwiGLUFFN(x) = SwiGLU(x, W1, W2, W3) = (SiLU(x@W1^T) * x@W3^T)@W2^T
    #              = (x@W1^T * sigmoid(x@W1^T) * x@W3^T)@W2^T
    # x: [d_model], W1, W3: [d_ff, d_model], W2: [d_model, d_ff]
    w1: Linear
    w2: Linear
    w3: Linear
    
    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        d_ff = ((d_model + 23) // 24) * 64 if d_ff is None else d_ff
        # self.w1 = Init.linear(self.d_ff, d_model, device=device, dtype=dtype)
        # self.w2 = Init.linear(d_model, self.d_ff, device=device, dtype=dtype)
        # self.w3 = Init.linear(self.d_ff, d_model, device=device, dtype=dtype)
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)
    
    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        
        # x_w1 = einops.einsum(
        #     x, self.w1, "... d_model, d_ff d_model -> ... d_ff"
        # )
        # mid = x_w1 * torch.sigmoid(x_w1) * einops.einsum(
        #     x, self.w3, "... d_model, d_ff d_model -> ... d_ff"
        # )
        # return einops.einsum(
        #     mid, self.w2, "... d_ff, d_model d_ff -> ... d_model"
        # )
        x_w1 = self.w1(x)
        mid = x_w1 * torch.sigmoid(x_w1) * self.w3(x)
        return self.w2(mid)

class RoPE(torch.nn.Module):
    
    theta: float
    d_k: int
    d_k_2: int
    max_seq_length: int
    r: Float[Tensor, "max_seq_length d_k_2 2 2"]
    
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_length: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        assert d_k % 2 == 0
        self.theta = theta
        self.d_k = d_k
        self.d_k_2 = d_k // 2
        self.max_seq_length = max_seq_length
        self.register_buffer("r", torch.empty(max_seq_length, self.d_k_2, 2, 2, device=device, dtype=dtype), persistent=False)
        self.register_buffer("j", torch.arange(self.d_k_2, device=device, dtype=dtype).view(1, self.d_k_2), persistent=False)
        
        i = torch.arange(self.max_seq_length, device=device, dtype=dtype).view(self.max_seq_length, 1)
        x = i / (self.theta ** (self.j * 2 / self.d_k))
        self.r[..., 0, 0] = torch.cos(x)
        self.r[..., 0, 1] = -torch.sin(x)
        self.r[..., 1, 0] = torch.sin(x)
        self.r[..., 1, 1] = torch.cos(x)
        # print(self.r)
        
    def forward(self, x: Float[Tensor, "... seq_len d_k"], token_positions: Int[Tensor, "... seq_len"] | None = None) -> Float[Tensor, "... seq_len d_k"]:
        assert x.shape[-1] == self.d_k
        seq_len = x.shape[-2]
        r_sliced: Int[Tensor, "... seq_len d_k_2 2 2"] = self.r[token_positions] if token_positions is not None else self.r[:seq_len]
        # x_grouped: Int[Tensor, "... seq_len d_k_2 2"] = x.reshape(*x.shape[:-1], -1, 2)
        y = einops.rearrange(x, "... seq_len (d_k_2 d_2) -> ... seq_len d_k_2 d_2", d_2=2)
        return einops.einsum(
            y, r_sliced, "... seq_len d_k_2 i, ... seq_len d_k_2 j i -> ... seq_len d_k_2 j"
        ).reshape(x.shape)

def softmax(x: Tensor, dim: int = -1):
    mid: Tensor = torch.exp(x - x.max(dim=dim, keepdim=True).values)
    return mid / mid.sum(dim=dim, keepdim=True)

def attention(
    q: Float[Tensor, "... len_q d_k"],
    k: Float[Tensor, "... len_k d_k"],
    v: Float[Tensor, "... len_k d_v"],
    mask: Bool[Tensor, "... len_q len_k"] | None = None
):
    pre_val = einops.einsum(q, k, "... len_q d_k, ... len_k d_k -> ... len_q len_k") / math.sqrt(q.shape[-1])
    # if mask is not None: pre_val = pre_val if mask else -torch.inf # 这么写会报错
    if mask is not None:
        pre_val = torch.where(mask, pre_val, -torch.inf)
    return einops.einsum(softmax(pre_val, dim=-1), v, "... len_q len_k, ... len_k d_v -> ... len_q d_v")

class MultiHeadAttention(torch.nn.Module):
    d_model: int
    d_k: int
    num_heads: int
    w_q: Float[Tensor, "d_model d_model"]
    w_k: Float[Tensor, "d_model d_model"]
    w_v: Float[Tensor, "d_model d_model"]
    w_o: Float[Tensor, "d_model d_model"]
    rope: RoPE
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        theta: float | None = None,
        max_seq_length: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.d_model = d_model
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        
        self.w_q = Init.linear(d_model, d_model, device=device, dtype=dtype)
        self.w_k = Init.linear(d_model, d_model, device=device, dtype=dtype)
        self.w_v = Init.linear(d_model, d_model, device=device, dtype=dtype)
        self.w_o = Init.linear(d_model, d_model, device=device, dtype=dtype)
        
        # rope
        if theta is None or max_seq_length is None: self.rope = None
        else: self.rope = RoPE(theta, self.d_k, max_seq_length, device=device, dtype=dtype)
    
    def forward(
        self, 
        x: Float[Tensor, "... seq_len d_model"], 
        token_positions: Int[Tensor, " ... sequence_length"] | None = None
    ) -> Float[Tensor, "... seq_len d_model"]:
        seq_len = x.shape[-2]
        # calc q, k, v
        q = einops.einsum(x, self.w_q, "... seq_len d_model, d_1 d_model -> ... seq_len d_1")
        k = einops.einsum(x, self.w_k, "... seq_len d_model, d_1 d_model -> ... seq_len d_1")
        v = einops.einsum(x, self.w_v, "... seq_len d_model, d_1 d_model -> ... seq_len d_1")
        
        # split heads
        # q.reshape(*q.shape[:-1], self.num_heads, self.d_k).transpose(-3, -2)
        # k.reshape(*k.shape[:-1], self.num_heads, self.d_k).transpose(-3, -2)
        # v.reshape(*v.shape[:-1], self.num_heads, self.d_k).transpose(-3, -2)
        
        # I'll try to rewrite everything in einops
        q = einops.rearrange(q, "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k", num_heads=self.num_heads)
        k = einops.rearrange(k, "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k", num_heads=self.num_heads)
        v = einops.rearrange(v, "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k", num_heads=self.num_heads)
        
        # apply rope
        if self.rope is not None:
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)
        
        # calc mask
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=q.device))
        
        # calc attentions
        atten = attention(q, k, v, mask)
        
        # concatenate
        return einops.einsum(
            einops.rearrange(atten, "... num_heads seq_len d_k -> ... seq_len (num_heads d_k)"),
            self.w_o, "... seq_len d_model, d_1 d_model -> ... seq_len d_1")

class TransformerBlock(torch.nn.Module):
    rms1: RMSNorm
    rms2: RMSNorm
    atten: MultiHeadAttention
    swiglu: SwiGLUFFN
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int | None = None,
        theta: float | None = None,
        max_seq_length: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.rms1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.rms2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.atten = MultiHeadAttention(d_model, num_heads, theta, max_seq_length, device=device, dtype=dtype)
        self.swiglu = SwiGLUFFN(d_model, d_ff, device=device, dtype=dtype)
    
    def forward(self, x: Float[Tensor, "... seq_len d_model"]) -> Float[Tensor, "... seq_len d_model"]:
        
        # layer 1
        y = x + self.atten(self.rms1(x))
        
        # layer 2
        return y + self.swiglu(self.rms2(y))
    
class TransformerLM(torch.nn.Module): # 最后不会softmax和cross entropy，计算出来logits
    tblocks: torch.nn.ModuleList
    in_embed: Embedding
    rms_final: RMSNorm
    out_embed: Linear
    num_layers: int
    
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int | None = None,
        theta: float | None = None,
        context_length: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.num_layers = num_layers
        self.tblocks = torch.nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, theta, context_length, device=device, dtype=dtype) 
            for _ in range(num_layers)
        ])
        self.in_embed = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.rms_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.out_embed = Linear(d_model, vocab_size, device=device, dtype=dtype)
    
    def forward(self, x: Int[Tensor, "... seq_len"]) -> Float[Tensor, "... seq_len vocab_size"]:
        # 得到的是每个vocab_size中word的概率logits。seq_len维的第i个就表示预测的第i+1个的概率分布（softmax之后）
        # embedding
        hidden = self.in_embed(x)
        
        # transformer
        for block in self.tblocks:
            hidden = block(hidden)
        
        # Return logits; the training loss applies softmax internally.
        return self.out_embed(self.rms_final(hidden))

def cross_entropy(
    logits: Float[Tensor, "... batch_size vocab_size"], 
    x: Int[Tensor, "... batch_size"]
) -> Float[Tensor, "..."]:
    
    # sum -log p_i = sum -log softmax logits_i = sum -[(logits_i - m) - logsumexp(logits)]
    l = logits - logits.max(dim=-1, keepdim=True).values
    return (torch.logsumexp(l, dim=-1, keepdim=False) - l.gather(dim=-1, index=x.unsqueeze(-1)).squeeze(-1)).mean(dim=-1)

def perplexity(
    logits: Float[Tensor, "... batch_size seq_len vocab_size"], 
    x: Int[Tensor, "... batch_size seq_len"]
) -> Float[Tensor, "... batch_size"]:
    return torch.exp(cross_entropy(logits, x))

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0: raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)
        
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        # print(self.param_groups)
        # print(self.state)
        for group in self.param_groups: # 每个group是一个字典[str, list[parameters]]
            lr = group["lr"] # learning rate，这一个group都用这个learning rate
            for p in group["params"]: # 对于params中的每一个parameter
                # print(p)
                if p.grad is None: continue # 不需要梯度下降
                state = self.state[p]
                t = state.get("t", 0)
                grad = p.grad.data
                p.data -= lr / math.sqrt(t + 1) * grad
                state["t"] = t + 1
        return loss
    
def train_sgd_example():
    weights = torch.nn.Parameter(5 * torch.randn(10, 10))
    opt = SGD([weights], lr=1e3)
    
    for t in range(100):
        opt.zero_grad()
        loss = (weights ** 2).mean()
        print(loss.cpu().item())
        loss.backward()
        opt.step()

class AdamW(torch.optim.Optimizer):
    def __init__(
        self, 
        params, 
        lr: float | Tensor = 0.001, 
        betas: tuple[float | Tensor, float | Tensor] = (0.9, 0.999), 
        eps: float = 1e-8, 
        weight_decay: float = 0.01):
        if lr < 0: raise ValueError(f"Invalid alpha {lr}.")
        if betas[0] >= 1 or betas[0] < 0 or betas[1] >= 1 or betas[1] < 0: raise ValueError(f"Invalid betas {betas}")
        if eps < 0: raise ValueError(f"Invalid eps {eps}")
        if weight_decay < 0: raise ValueError(f"Invalid weight_decay {weight_decay}")
        defaults = {"lr": lr, "eps": eps, "betas": betas, "weight_decay": weight_decay}
        super().__init__(params, defaults)
    
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            betas = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for param in group["params"]:
                if param.grad is None: continue
                state = self.state[param]
                grad = param.grad.data
                if "m" not in state: state["m"] = torch.zeros_like(grad)
                if "v" not in state: state["v"] = torch.zeros_like(grad)
                if "b_t" not in state: state["b_t"] = torch.ones(2, device=state["m"].device)
                
                state["m"] = state["m"] * betas[0] + grad * (1 - betas[0])
                state["v"] = state["v"] * betas[1] + grad * grad * (1 - betas[1])
                state["b_t"] *= torch.tensor(betas, device=state["b_t"].device)
                alpha = lr * math.sqrt(1 - state["b_t"][1]) / (1 - state["b_t"][0])
                param.data -= lr * weight_decay * param
                param.data -= alpha * state["m"] / (torch.sqrt(state["v"]) + eps)
                
        return loss

def lr_cos_schedule(t, alpha_max, alpha_min, T_w, T_c):
    if t < T_w: return t / T_w * alpha_max
    if t < T_c: return alpha_min + 0.5 * (alpha_max - alpha_min) * (1 + math.cos(math.pi * (t - T_w) / (T_c - T_w)))
    return alpha_min

def clip_gradient(param_list: Iterable[torch.nn.Parameter], max_l2_norm: float, eps: float = 1e-6):
    params = [p for p in param_list if p.grad is not None]
    norm = torch.sqrt(torch.stack([
        param.grad.pow(2).sum()
        for param in params
    ]).sum())
    if norm >= max_l2_norm:
        scale = max_l2_norm / (norm + eps)
        for param in params:
            param.grad.data *= scale

if __name__ == "__main__":
    pass
    # model = Linear(3, 4)
    # print(model.weights.data)
    # model = RMSNorm(10)
    # token = torch.ones(
    #     2, 3, 10
    # )
    # print(model(token))
    # model = RoPE(10000, 2, 4)
    # pass
    # x = torch.rand(2, 3)
    # print(x)
    # print(softmax(x))
    
    # print(torch.tril(torch.ones(3, 3)))
    # logits = torch.rand(3, 4, 5)
    # print(logits)
    # x = torch.randint(0, 5, (3, 4))
    # print(x)
    # print(cross_entropy(logits, x))
    # train_sgd_example()
    # print(torch.optim.AdamW())
    # optimizer = AdamW()
    # print(optimizer)
    # model = TransformerLM(vo)
    # print(model)
    
    
    
