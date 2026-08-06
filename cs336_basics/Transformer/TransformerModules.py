import torch
import einops
from jaxtyping import Float
from torch import Tensor
import math

class Linear(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        if dtype is None: dtype = torch.float32
        if device is None: device = "cpu"
        self.W = torch.nn.Parameter(
            torch.empty(
                out_features,
                in_features,
                device=device,
                dtype=dtype
            )
        )
        std = math.sqrt(2 / (in_features + out_features))
        torch.nn.init.trunc_normal_(self.W, 0, std, -3 * std, 3 * std)
        
    def forward(self, x: Tensor) -> Tensor:
        # return x @ W^T
        return einops.einsum(x, self.W, "... d_in, d_out d_in -> ... d_out")
    
if __name__ == "__main__":
    
    model = Linear(3, 4)
    print(model.W)
    print(model.W.data)
