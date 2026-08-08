import torch
import einops
from jaxtyping import Float
from jaxtyping import Int
from torch import Tensor
import math

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
        self.weights = torch.nn.Parameter(
            torch.empty(
                out_features,
                in_features,
                device=device,
                dtype=dtype
            )
        )
        std = math.sqrt(2 / (in_features + out_features))
        torch.nn.init.trunc_normal_(self.weights, 0, std, -3 * std, 3 * std)
        
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
        self.embed_map = torch.nn.Parameter(
            torch.empty(
                num_embeddings,
                embedding_dim,
                device=device,
                dtype=dtype
            )
        )
        torch.nn.init.trunc_normal_(self.embed_map, 0, 1, -3, 3)
        
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
        self.weights = torch.nn.Parameter(
            torch.ones(
                d_model,
                device=device,
                dtype=dtype
            )
        )
        self.weights
    
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
        self.w1 = torch.nn.Parameter(
            torch.empty(
                self.d_ff, d_model, device=device, dtype=dtype
            )
        )
        self.w2 = torch.nn.Parameter(
            torch.empty(
                d_model, self.d_ff, device=device, dtype=dtype
            )
        )
        self.w3 = torch.nn.Parameter(
            torch.empty(
                self.d_ff, d_model, device=device, dtype=dtype
            )
        )
        std = math.sqrt(2 / (self.d_ff + d_model))
        torch.nn.init.trunc_normal_(self.w1, 0, std, -3 * std, 3 * std)
        torch.nn.init.trunc_normal_(self.w2, 0, std, -3 * std, 3 * std)
        torch.nn.init.trunc_normal_(self.w3, 0, std, -3 * std, 3 * std)
    
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
    
    # def calc_buffer(self):
        
    #     # j = torch.arange(self.d_k_2).view(1, self.d_k_2)
        
        
    
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
        x_grouped: Int[Tensor, "... seq_len d_k_2 2"] = x.reshape(*x.shape[:-1], -1, 2)
        return einops.einsum(
            x_grouped, r_sliced, "... seq_len d_k_2 i, ... seq_len d_k_2 j i -> ... seq_len d_k_2 j"
        ).reshape(x.shape)

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
    pass
    
