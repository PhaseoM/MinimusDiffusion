import math
from abc import ABC, abstractmethod
from typing import Optional, List, Type, Tuple, Dict

import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange
from transformers import AutoTokenizer, CLIPTextModel

from models.modules.neural_ode import ConditionalVectorField


class MLP(nn.Module):
    def __init__(self, dims: List[int], activation: Type[nn.Module] = nn.SiLU, final_init: bool = False):
        super().__init__()
        mlp = []
        for i in range(len(dims) - 1):
            mlp.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                mlp.append(activation())
        self.model = torch.nn.Sequential(*mlp)

        if final_init:
            nn.init.zeros_(self.model[-1].weight)
            nn.init.zeros_(self.model[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            - x: (... d)
        Returns:
            - x: (... d)
        """
        return self.model(x)


class FourierEncoder(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0
        self.half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(1, self.half_dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            - t: (b)
        Return:
            - embeddings: (b d)
        """
        # compute frequencies f_i = 2 * pi * w_i * t
        t = t.view(-1, 1)  # b 1
        freqs = t * self.weights * 2 * math.pi  # b hd

        # compute sin(f_i) and cos(f_i)
        sin_embed = torch.sin(freqs)  # b hd
        cos_embed = torch.cos(freqs)  # b hd

        # Concatenate and return
        return torch.cat([sin_embed, cos_embed], dim=-1) * math.sqrt(2)  # b d


class PromptEncoder(nn.Module):
    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        text_encoder_path: str = "./pretrained/clip-vit-base-patch32",
    ):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(text_encoder_path, local_files_only=True)
        self.text_encoder = CLIPTextModel.from_pretrained(text_encoder_path, local_files_only=True)

        self.text_encoder.requires_grad_(False)
        self.text_encoder.eval()

        self.output_dim = self.text_encoder.config.hidden_size
        self.max_length = self.text_encoder.config.max_position_embeddings

    def train(self, mode: bool = True):
        super().train(mode)
        self.text_encoder.eval()
        return self

    def forward(self, prompts: List[str]) -> torch.Tensor:
        device = next(self.text_encoder.parameters()).device

        tokens = self.tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_attention_mask=True,
            return_tensors="pt",
        ).to(device)

        outputs = self.text_encoder(
            input_ids=tokens["input_ids"],
            attention_mask=tokens["attention_mask"],
            return_dict=True,
        ).pooler_output  # (B, output_dim)

        return outputs


class Patchifier(nn.Module):
    def __init__(self, img_size: int, patch_size: int, c_in: int, dim: int):
        super().__init__()
        assert img_size % patch_size == 0, "Image size must be divisible by patch size"

        self.model = nn.Sequential(
            nn.Conv2d(c_in, dim, kernel_size=patch_size, stride=patch_size),
            Rearrange("b d h w -> b (h w) d"),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            - x: (b c h w)
        Returns:
            - x: (b l d)
        """
        return self.model(x)


class Depatchifier(nn.Module):
    def __init__(self, img_size: int, patch_size: int, dim: int, final_dim: int, c_out: int):
        super().__init__()
        assert img_size % patch_size == 0, "Image size must be divisible by patch size"
        self.patch_size = patch_size
        h = w = img_size // patch_size

        self.model = nn.Sequential(
            # Norm + MLP
            nn.RMSNorm(dim, elementwise_affine=False),
            MLP([dim, 4 * dim, final_dim * patch_size**2]),
            # Depatchify
            Rearrange("b (h w) (f ph pw) -> b f (h ph) (w pw)", h=h, w=w, f=final_dim, ph=patch_size, pw=patch_size),
            # Final convolution
            nn.Conv2d(final_dim, c_out, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            - x: (b l d)
        Returns:
            - x: (b c h w)
        """
        return self.model(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, d_model: int):
        super().__init__()
        assert d_model % num_heads == 0

        self.qkv_project = nn.Linear(d_model, d_model * 3)
        self.split_heads = Rearrange("b l (h d) -> b h l d", h=num_heads)
        self.merge_heads = Rearrange("b h l d -> b l (h d)", h=num_heads)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            - x: (b l d)
        Returns:
            - x: (b l d)
        """
        # Compute queries, keys, and values (b l (h d))
        q, k, v = self.qkv_project(x).chunk(3, dim=-1)

        # split heads (b h l d)
        q, k, v = map(self.split_heads, (q, k, v))

        # Compute attention
        attn = nn.functional.scaled_dot_product_attention(q, k, v)

        # merge heads (b l (h d))
        x = self.merge_heads(attn)

        return self.out(x)


class DiTBlock(nn.Module):
    def __init__(self, num_heads: int, d_model: int):
        """
        Args:
            - num_heads: number of attention heads
            - d_model: dimension of hidden layers
        """
        super().__init__()

        # Normalization
        self.norm1 = nn.RMSNorm(d_model, elementwise_affine=False)
        self.norm2 = nn.RMSNorm(d_model, elementwise_affine=False)
        self.ada_ln = nn.Sequential(nn.RMSNorm(d_model, elementwise_affine=False), nn.Linear(d_model, d_model * 6))

        # Initialize conditioning to zero
        nn.init.zeros_(self.ada_ln[1].weight)
        nn.init.zeros_(self.ada_ln[1].bias)

        self.mha = MultiHeadAttention(num_heads, d_model)

        # Feedforward
        self.ff = MLP([d_model, 4 * d_model, d_model])

    def modulate(self, x: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        """
        Args:
            - x: b l d
            - scale: b l d
            - bias: b l d
        Returns:
            - x: b l d
        """
        return x * (1 + scale) + bias

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            - x: (b l d)
            - c: (b d)
        Returns:
            - x: (b l d)
        """
        # Compute conditioning gating, scaling, and bias
        c = rearrange(self.ada_ln(c), "b d -> b 1 d")
        attn_scale, attn_bias, attn_gate, ff_scale, ff_bias, ff_gate = c.chunk(6, dim=-1)

        # Attention + Feedforward
        x = x + attn_gate * self.mha(self.modulate(self.norm1(x), attn_scale, attn_bias))
        x = x + ff_gate * self.ff(self.modulate(self.norm2(x), ff_scale, ff_bias))
        return x


class DiffusionTransformer(nn.Module):
    def __init__(
        self,
        num_heads: int,
        num_layers: int,
        num_tokens: int,
        d_model: int,
    ):
        """
        Args:
            - num_heads: number of attention heads
            - num_layers: number of layers
            - num_tokens: sequence length (for sake of positional embeddings)
            - d_model: dimension of hidden layers
        """
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(num_layers):
            self.layers.append(DiTBlock(num_heads, d_model))

        # Positional encodings
        self.pos_encodings = nn.Parameter(torch.randn(num_tokens, d_model))

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            - x: (b l d)
            - c: (b d)
        Return:
            - x: (b l d)
        """
        x = x + self.pos_encodings.unsqueeze(0)
        for layer in self.layers:
            x = layer(x, c)
        return x


class DiffusionTransformerWithFlowModel(ConditionalVectorField):
    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 8,
        num_layers: int = 12,
        c: int = 1,
        d_model: int = 256,
        num_heads: int = 4,
        final_dim: int = 10,
        text_encoder_name: str = "openai/clip-vit-base-patch32",
        text_encoder_path: str = "./pretrained/clip-vit-base-patch32",
    ):
        super().__init__()
        # build time_embedder, prompt_embedder and prompt_proj
        self.time_embedder = FourierEncoder(d_model)
        self.prompt_embedder = PromptEncoder(model_name=text_encoder_name, text_encoder_path=text_encoder_path)
        self.prompt_proj = nn.Sequential(
            nn.LayerNorm(self.prompt_embedder.output_dim),
            MLP([self.prompt_embedder.output_dim, d_model, d_model]),
        )

        # build patchifier
        self.patchifier = Patchifier(img_size=img_size, patch_size=patch_size, c_in=c, dim=d_model)

        # build DiT
        num_tokens = (img_size // patch_size) ** 2
        self.dit = DiffusionTransformer(num_heads, num_layers, num_tokens, d_model)

        # build depatchifier
        self.depatchifier = Depatchifier(
            img_size=img_size, patch_size=patch_size, dim=d_model, final_dim=final_dim, c_out=c
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: Optional[torch.Tensor | list[str]]) -> torch.Tensor:
        """
        Args:
            - x: b c h w
            - t: b 1 1 1
            - c: b 1 1 1
        Return:
            - u_t^theta(x|y): b c h w
        """
        # Embed time and prompt
        time_embed = self.time_embedder(t)  # b d
        prompt_features = self.prompt_embedder(y)  # b d
        prompt_embed = self.prompt_proj(prompt_features)

        x = self.patchifier(x)  # b l d
        x = self.dit(x, time_embed + prompt_embed)  # b l d
        x = self.depatchifier(x)

        return x
