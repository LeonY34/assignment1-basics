import torch
import einops
from jaxtyping import Float, Int, Bool
from torch import Tensor
import math

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
    
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        
        super().__init__()
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
    d_model: int
    d_ff: int
    w1: Float[Tensor, "d_ff d_model"]
    w2: Float[Tensor, "d_model d_ff"]
    w3: Float[Tensor, "d_ff d_model"]
    
    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = ((d_model + 23) // 24) * 64 if d_ff is None else d_ff
        self.w1 = Init.linear(self.d_ff, d_model, device=device, dtype=dtype)
        self.w2 = Init.linear(d_model, self.d_ff, device=device, dtype=dtype)
        self.w3 = Init.linear(self.d_ff, d_model, device=device, dtype=dtype)
    
    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        
        x_w1 = einops.einsum(
            x, self.w1, "... d_model, d_ff d_model -> ... d_ff"
        )
        mid = x_w1 * torch.sigmoid(x_w1) * einops.einsum(
            x, self.w3, "... d_model, d_ff d_model -> ... d_ff"
        )
        return einops.einsum(
            mid, self.w2, "... d_ff, d_model d_ff -> ... d_model"
        )

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
        device: torch.device | None = None
    ):
        super().__init__()
        assert d_k % 2 == 0
        self.theta = theta
        self.d_k = d_k
        self.d_k_2 = d_k // 2
        self.max_seq_length = max_seq_length
        self.register_buffer("r", torch.empty(max_seq_length, self.d_k_2, 2, 2, device=device), persistent=False)
        self.register_buffer("j", torch.arange(self.d_k_2, device=device).view(1, self.d_k_2), persistent=False)
        
        i = torch.arange(self.max_seq_length, device=device).view(self.max_seq_length, 1)
        x = i / (self.theta ** (self.j * 2 / self.d_k))
        self.r[..., 0, 0] = torch.cos(x)
        self.r[..., 0, 1] = -torch.sin(x)
        self.r[..., 1, 0] = torch.sin(x)
        self.r[..., 1, 1] = torch.cos(x)
        # print(self.r)
        
    def forward(self, x: Float[Tensor, "... seq_len d_k"], token_positions: Int[Tensor, "... seq_len"]) -> Float[Tensor, "... seq_len d_k"]:
        assert x.shape[-1] == self.d_k
        r_sliced: Int[Tensor, "... seq_len d_k_2 2 2"] = self.r[token_positions]
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
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
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
    
    def forward(self, x: Float[Tensor, "... seq_len d_model"]) -> Float[Tensor, "... seq_len d_model"]:
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
        
        # calc mask
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=q.device))
        
        # calc attentions
        atten = attention(q, k, v, mask)
        
        # concatenate
        return einops.einsum(
            einops.rearrange(atten, "... num_heads seq_len d_k -> ... seq_len (num_heads d_k)"),
            self.w_o, "... seq_len d_model, d_1 d_model -> ... seq_len d_1")
        

if __name__ == "__main__":
    
    # model = Linear(3, 4)
    # print(model.weights)
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
    
    print(torch.tril(torch.ones(3, 3)))
    
