import torch
import torch.nn as nn
from torch.nn import functional as F
from pytorch_wavelets import DWTForward, DWTInverse

class PatchEmbedding(nn.Module):
    def __init__(self, patch_size=16, in_channels=3, embed_dim=768):
        super(PatchEmbedding, self).__init__()
        self.patch_embeddings = nn.Conv2d(in_channels=in_channels,
                                          out_channels=embed_dim,
                                          kernel_size=patch_size,
                                          stride=patch_size)

    def forward(self, x):
        patches = self.patch_embeddings(x)
        patches = patches.flatten(2)
        patches = patches.transpose(1, 2)
        return patches


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=1024):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        return attn_output


class MLPBlock(nn.Module):
    def __init__(self, embed_dim, mlp_dim=3072):
        super(MLPBlock, self).__init__()
        self.fc1 = nn.Linear(embed_dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, embed_dim)

    def forward(self, x):
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim):
        super(TransformerEncoderLayer, self).__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLPBlock(embed_dim, mlp_dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class FDB(nn.Module):
    def __init__(self, patch_size=8, in_channels=3, embed_dim=768, depth=12, num_heads=12, mlp_dim=3072, o_channels=64):
        super(FDB, self).__init__()
        self.conv384_3 = nn.Sequential(
            nn.Conv2d(o_channels*6, o_channels*4, kernel_size=1),
            nn.BatchNorm2d(o_channels*4),
            nn.ReLU(),
            nn.Conv2d(o_channels*4, 3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU()
        )
        self.dwt = DWTForward(J=2, wave='db1', mode='zero')
        self.idwt = DWTInverse(wave='db1', mode='zero')
        self.upsample = nn.Upsample(size=(256, 256), mode='bilinear', align_corners=False)
        self.patch_embed = PatchEmbedding(patch_size, in_channels, embed_dim)
        self.pos_encoding = PositionalEncoding(embed_dim)
        self.encoder_layers = nn.ModuleList(
            [TransformerEncoderLayer(embed_dim, num_heads, mlp_dim) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)

        self.adaptive_pool0 = nn.AdaptiveAvgPool1d(o_channels*8)

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=o_channels*8, out_channels=o_channels*16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(o_channels*16),
            nn.ReLU(),
            nn.Conv2d(in_channels=o_channels*16, out_channels=o_channels*16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(o_channels*16),
            nn.ReLU(),
            nn.Conv2d(in_channels=o_channels*16, out_channels=o_channels*8, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(o_channels*8),
            nn.ReLU()
        )

        self.upconv1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=o_channels*8, out_channels=o_channels*8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(o_channels*8),
            nn.ReLU(),
            nn.Conv2d(in_channels=o_channels*8, out_channels=o_channels*4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(o_channels*4),
            nn.ReLU()
        )

        self.upconv2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=o_channels*4, out_channels=o_channels*4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(o_channels*4),
            nn.ReLU(),
            nn.Conv2d(in_channels=o_channels*4, out_channels=o_channels*2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(o_channels*2),
            nn.ReLU()
        )

        self.upconv3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=o_channels*2, out_channels=o_channels*2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(o_channels*2),
            nn.ReLU(),
            nn.Conv2d(in_channels=o_channels*2, out_channels=o_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(o_channels),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv384_3(x)
        Yl, Yh = self.dwt(x)
        x = self.idwt((Yl, Yh))
        x = self.upsample(x)
        x = self.patch_embed(x)
        x = self.pos_encoding(x)
        for layer in self.encoder_layers:
            x = layer(x)
        x = self.norm(x)
        x = self.adaptive_pool0(x)
        x = x.reshape(x.size(0), -1, 32, 32)
        x4 = self.conv1(x)
        x3 = self.upconv1(x4)
        x2 = self.upconv2(x3)
        x1 = self.upconv3(x2)

        return (x1, x2, x3, x4)
