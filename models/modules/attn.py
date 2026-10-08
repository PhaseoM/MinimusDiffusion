import torch
from torch import nn
import torch.nn.functional as nnF


# class ScaledDotAttention(nn.Module):
#     def __init__(self, masked=False):
#         super().__init__()
#         self.masked = masked


#     def forward(self, Q_, K_, V_):
#         seq_len = Q_.shape[-2]
#         scale = Q_.shape[-1] ** 0.5
#         att_score = torch.matmul(Q_, K_.transpose(-2, -1)) / scale
#         if self.masked is True:
#             causal_mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=att_score.device).triu(diagonal=1)
#             att_score = att_score.masked_fill(causal_mask, -torch.inf)
#         att_score = nnF.softmax(att_score, dim=-1)
#         att_score = torch.matmul(att_score, V_)
#         return att_score


class ScaledDotAttention(nn.Module):
    def __init__(self, masked=False):
        super().__init__()
        self.masked = masked

    def forward(self, Q_, K_, V_):
        return nnF.scaled_dot_product_attention(Q_, K_, V_, is_causal=self.masked)


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, d_model, att_score, bias=False):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model can't be evenly divisible by num_heads")
        self.attention = att_score
        self.num_heads = num_heads
        self.w_q = nn.Linear(in_features=d_model, out_features=d_model, bias=bias)
        self.w_k = nn.Linear(in_features=d_model, out_features=d_model, bias=bias)
        self.w_v = nn.Linear(in_features=d_model, out_features=d_model, bias=bias)
        self.w_o = nn.Linear(in_features=d_model, out_features=d_model, bias=bias)

    def forward(self, X):
        query = self._transpose(self.w_q(X))
        key = self._transpose(self.w_k(X))
        value = self._transpose(self.w_v(X))
        multi_att = self.attention(query, key, value)

        multi_att = self.w_o(self._transpose_inverse(multi_att))
        return multi_att

    def _transpose(self, X):
        X = X.reshape(X.shape[0], X.shape[1], self.num_heads, -1)
        X = X.permute(0, 2, 1, 3)
        return X

    def _transpose_inverse(self, X):
        X = X.permute(0, 2, 1, 3)
        X = X.reshape(X.shape[0], X.shape[1], -1)
        return X


class AttentionBlock(nn.Module):
    def __init__(
        self,
        num_heads=8,
        d_model=512,
        mlp_hiddens=2048,
        dropout=0,
        att_score=None,
    ):
        super().__init__()
        if att_score is None:
            att_score = ScaledDotAttention()
        self.att_model = nn.Sequential(
            nn.LayerNorm(normalized_shape=d_model),
            MultiHeadAttention(
                num_heads=num_heads,
                d_model=d_model,
                att_score=att_score,
            ),
        )
        self.mlp_model = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(in_features=d_model, out_features=mlp_hiddens),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_features=mlp_hiddens, out_features=d_model),
            nn.Dropout(dropout),
        )

    def forward(self, X):
        X = X + self.att_model(X)
        X = X + self.mlp_model(X)
        return X


class AttentionStack(nn.Module):
    def __init__(
        self,
        num_layers=12,
        num_heads=8,
        d_model=512,
        mlp_hiddens=2048,
        dropout=0,
        att_score=None,
    ):
        super().__init__()
        self.model = nn.Sequential(*self._attn_stack(num_layers, num_heads, d_model, mlp_hiddens, dropout, att_score))

    def _attn_stack(self, num_layers, num_heads, d_model, mlp_hiddens, dropout, att_score):
        attn_stack = []
        for _ in range(num_layers):
            attn_stack.append(
                AttentionBlock(
                    num_heads=num_heads,
                    d_model=d_model,
                    mlp_hiddens=mlp_hiddens,
                    dropout=dropout,
                    att_score=att_score,
                )
            )
        return attn_stack

    def forward(self, X):
        return self.model(X)


class SequentialTranspose(nn.Module):
    def __init__(self, dim0, dim1):
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, X):
        return X.transpose(self.dim0, self.dim1)


class ImgPatchEmbedding(nn.Module):
    def __init__(
        self,
        img_size,
        patch_size,
        d_model,
        dropout,
    ):
        super().__init__()

        def _make_tuple(x):
            if not isinstance(x, (list, tuple)):
                return (x, x)
            return x

        img_size, patch_size = _make_tuple(img_size), _make_tuple(patch_size)
        self.num_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1]) + 1
        self.cls_token = nn.Parameter(torch.empty(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02, a=-0.04, b=0.04)
        self.patch_embed = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=d_model,
                kernel_size=patch_size,
                stride=patch_size,
            ),
            nn.Flatten(2),
            SequentialTranspose(1, 2),
        )
        self.pos_embed = nn.Parameter(torch.empty(1, self.num_patches, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02, a=-0.04, b=0.04)
        self.dropout_embed = nn.Dropout(dropout)

    def forward(self, X):
        cls_token_ = self.cls_token.expand(X.shape[0], -1, -1)
        patch_embed_ = self.patch_embed(X)
        patch_embed_cls = torch.cat((cls_token_, patch_embed_), dim=1)
        pos_embed_ = self.pos_embed.expand(X.shape[0], -1, -1)
        output = self.dropout_embed(patch_embed_cls + pos_embed_)
        return output


class VisionTransformer(nn.Module):
    def __init__(
        self,
        size_n=12,
        d_model=768,
        num_heads=12,
        num_classes=10,
        mlp_hiddens=3072,
        img_size=32,
        patch_size=4,
        dropout=0,
    ):
        super().__init__()
        self.embedding = ImgPatchEmbedding(img_size, patch_size, d_model, dropout)
        self.encoder = AttentionStack(size_n, num_heads, d_model, mlp_hiddens, dropout)
        self.mlp_cls = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(in_features=d_model, out_features=num_classes),
        )

    def forward(self, X):
        X = self.embedding(X)
        cls_token = self.encoder(X)[:, 0]
        output = self.mlp_cls(cls_token)
        return output
