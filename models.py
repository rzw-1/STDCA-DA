import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from einops import rearrange
from torch.autograd import Function

class GradientReversalLayer(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, *grad_output):
        return grad_output[0].neg() * ctx.alpha, None

def grad_reverse(x, alpha=1.0):
    return GradientReversalLayer.apply(x, alpha)

class DomainClassifier(nn.Module):
    def __init__(self, embed_size, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_size, hidden),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, 2)
        )

    def forward(self, x, alpha=1.0):
        x = grad_reverse(x, alpha)
        return self.net(x)

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads, dropout):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
                self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.queries = nn.Linear(embed_size, embed_size)
        self.keys = nn.Linear(embed_size, embed_size)
        self.values = nn.Linear(embed_size, embed_size)

        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

        self.norm = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x,  mask=None):
        residual = x
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.heads)

        energy = torch.einsum("bhqd,bhkd->bhqk", [queries, keys]) * (self.head_dim ** -0.5)

        if mask is not None:
            energy.masked_fill_(mask == 0, -1e9)
        attention = torch.softmax(energy, dim=-1)
        attention = self.dropout(attention)

        out = torch.einsum("bhal,bhlv->bhav", [attention, values])

        out = rearrange(out, "b h n d -> b n (h d)")

        out = self.fc_out(out)
        out = self.dropout(out)
        out = self.norm(residual + out)

        return out

class SpatioTemporalCrossAttention(nn.Module):
    def __init__(self, temp_embed_size, spat_embed_size, heads, dropout):
        super(SpatioTemporalCrossAttention, self).__init__()
        self.heads = heads
        self.temp_embed_size = temp_embed_size
        self.spat_embed_size = spat_embed_size
        self.head_dim = temp_embed_size // heads

        assert (
                self.head_dim * heads == temp_embed_size
        ), "Temporal embedding size needs to be divisible by heads"
        assert (
                temp_embed_size == spat_embed_size
        ), "Temporal and spatial embedding sizes must be equal for cross attention"

        self.temp_to_spat_queries = nn.Linear(spat_embed_size, spat_embed_size)
        self.temp_to_spat_keys = nn.Linear(spat_embed_size, spat_embed_size)
        self.temp_to_spat_values = nn.Linear(spat_embed_size, spat_embed_size)
        self.temp_to_spat_fc_out = nn.Linear(heads * self.head_dim, spat_embed_size)

        self.spat_to_temp_values = nn.Linear(temp_embed_size, temp_embed_size)
        self.spat_to_temp_keys = nn.Linear(temp_embed_size, temp_embed_size)
        self.spat_to_temp_queries = nn.Linear(temp_embed_size, temp_embed_size)
        self.spat_to_temp_fc_out = nn.Linear(heads * self.head_dim, temp_embed_size)

        self.norm = nn.LayerNorm(temp_embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, temp_features, spat_features, mask=None):
        s2t_residual = temp_features
        t2s_residual = spat_features

        t2s_queries = rearrange(self.temp_to_spat_queries(spat_features), "b n (h d) -> b h n d", h=self.heads)

        t2s_keys = rearrange(self.temp_to_spat_keys(temp_features), "b n (h d) -> b h n d", h=self.heads)
   
        t2s_values = rearrange(self.temp_to_spat_values(temp_features), "b n (h d) -> b h n d", h=self.heads)

        t2s_energy = torch.einsum("bhqd,bhkd->bhqk", [t2s_queries, t2s_keys]) * (self.head_dim ** -0.5)

        if mask is not None:
            t2s_energy.masked_fill_(mask == 0, -1e9)
        t2s_attention = torch.softmax(t2s_energy, dim=-1)
        t2s_attention = self.dropout(t2s_attention)

        t2s_out = torch.einsum("bhal,bhlv->bhav", [t2s_attention, t2s_values])

        t2s_out = rearrange(t2s_out, "b h n d -> b n (h d)")

        t2s_out = self.temp_to_spat_fc_out(t2s_out)
        t2s_out = self.dropout(t2s_out)
        t2s_out = self.norm(t2s_residual + t2s_out)

        s2t_queries = rearrange(self.spat_to_temp_queries(temp_features), "b n (h d) -> b h n d", h=self.heads)
        s2t_keys = rearrange(self.spat_to_temp_keys(spat_features), "b n (h d) -> b h n d", h=self.heads)
        s2t_values = rearrange(self.spat_to_temp_values(spat_features), "b n (h d) -> b h n d", h=self.heads)

        s2t_energy = torch.einsum("bhqd,bhkd->bhqk", [s2t_queries, s2t_keys]) * (self.head_dim ** -0.5)

        if mask is not None:
            s2t_energy.masked_fill_(mask == 0, -1e9)
        s2t_attention = torch.softmax(s2t_energy, dim=-1)
        s2t_attention = self.dropout(s2t_attention)

        s2t_out = torch.einsum("bhal,bhlv->bhav", [s2t_attention, s2t_values])

        s2t_out = rearrange(s2t_out, "b h n d -> b n (h d)")

        s2t_out = self.spat_to_temp_fc_out(s2t_out)
        s2t_out = self.dropout(s2t_out)
        s2t_out = self.norm(s2t_residual + s2t_out)

        return s2t_out, t2s_out

class SpatioTemporalFeatureExtractor(nn.Module):
    def __init__(self, num_channels, time_length, embed_size):
        super().__init__()

        self.temporal_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(1, 15), padding=(0, 7)),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.AvgPool2d((2, 2)),

            nn.Conv2d(32, 64, kernel_size=(1, 15), padding=(0, 7)),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.AvgPool2d((2, 2)),

            nn.Conv2d(64, 128, kernel_size=(1, 15), padding=(0, 7)),
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(0.3)
        )

        self.spatial_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.AvgPool2d((2, 2)),

            nn.Conv2d(32, 64, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.AvgPool2d((2, 2)),

            nn.Conv2d(64, 128, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(0.3)
        )

        self._init_output_dim(num_channels, time_length)

        self.projection = nn.Sequential(
            nn.Conv2d(128, embed_size, (1, 1)),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )

        self.self_attention = SelfAttention(
            embed_size=embed_size, 
            heads=8,
            dropout=0.3
        )

        self.cross_attention = SpatioTemporalCrossAttention(
            temp_embed_size=embed_size,
            spat_embed_size=embed_size,
            heads=8,
            dropout=0.3
        )

    def _init_output_dim(self, num_channels, time_length):
        with torch.no_grad():
            x = torch.randn(1, 1, num_channels, time_length)
            x_temp = self.temporal_conv(x)
            x_spat = self.spatial_conv(x)
            self.output_dim = x_temp.shape[1] + x_spat.shape[1]

    def forward(self, x):
        x = x.unsqueeze(1)
        x_temp = self.projection(self.temporal_conv(x))
        x_spat = self.projection(self.spatial_conv(x))

        x_temp = self.self_attention(x_temp)

        x_spat = self.self_attention(x_spat)

        temp_cross, spat_cross = self.cross_attention(x_temp, x_spat)

        temp_cross = rearrange(temp_cross, "b (h w) c -> b c h w", h=4, w=64)
        spat_cross = rearrange(spat_cross, "b (h w) c -> b c h w", h=4, w=64)

        return torch.cat([temp_cross, spat_cross], dim=1)

class STDCA_Net(nn.Module):
    def __init__(self, num_channels, time_length, num_classes, embed_size):
        super().__init__()

        self.feature_extractor = SpatioTemporalFeatureExtractor(num_channels, time_length, embed_size)

        self.fusion = nn.Sequential(
            nn.Conv2d(self.feature_extractor.output_dim, 128, kernel_size=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.Dropout(0.3)
        )

        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, embed_size))

        self.classifier = nn.Sequential(
            nn.Linear(embed_size, 64),
            nn.ELU(),
            nn.Dropout(0.5),                         
            nn.Linear(64, num_classes)
        )

        self.aux_classifier = nn.Sequential(
            nn.Linear(embed_size, 64),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

        self.domain_classifier = DomainClassifier(embed_size, 64)

    def forward(self, x, alpha=1.0):
        features = self.feature_extractor(x)

        fused = self.fusion(features)

        pooled = self.adaptive_pool(fused)
        pooled = pooled.squeeze(2).transpose(1, 2)

        feat = pooled.mean(dim=1)

        class_pred = self.classifier(feat)
        aux_pred = self.aux_classifier(feat)

        domain_pred = self.domain_classifier(feat, alpha)

        return class_pred, aux_pred, domain_pred
