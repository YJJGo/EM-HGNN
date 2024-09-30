import math
import torch
import torch.nn as nn
import itertools
import numpy as np
import torch.nn.functional as F
from torch_sparse import SparseTensor


class LinearPerMetapath(nn.Module):
    def __init__(self, num_metapaths, in_dim, out_dim):
        super(LinearPerMetapath, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_metapaths = num_metapaths
        self.W = nn.Parameter(torch.randn(self.num_metapaths, self.in_dim, self.out_dim))
        self.bias = nn.Parameter(torch.zeros(self.num_metapaths, self.out_dim))
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        fan_in, fan_out = self.W.size()[-2:]
        std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
        a = math.sqrt(3.0) * std
        nn.init._no_grad_uniform_(self.W, -a, a)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        return torch.einsum('bcm,cmn->bcn', x, self.W) + self.bias.unsqueeze(0)


class EMHGNN(nn.Module):
    def __init__(self, dataset, in_dims, feat_keys, embed_dim, hidden, num_classes, tgt_type, input_drop,
                 dropout, out_layers_num, residual=False, attention_module=False):
        super(EMHGNN, self).__init__()
        self.dataset = dataset
        self.in_dims = in_dims
        self.feat_keys = feat_keys
        self.embed_dim = embed_dim
        self.hidden = hidden
        self.num_classes = num_classes
        self.num_channels = num_channels = len(self.feat_keys)
        self.tgt_type = tgt_type
        self.input_drop = nn.Dropout(input_drop)
        self.residual = residual
        self.attention_module = attention_module
        self.flatten = nn.Flatten()

        self.feat_fc = nn.ParameterDict({})
        for k, v in in_dims.items():
            self.feat_fc[str(k)] = nn.Parameter(torch.Tensor(v, embed_dim))

        if attention_module:
            if dataset != 'AMiner':
                self.length_attention = nn.Parameter(torch.ones(num_channels) / num_channels)
                self.type_attention = nn.Parameter(torch.ones(num_channels) / num_channels)
            else:
                self.length_attention = nn.Parameter(torch.ones(num_channels) / num_channels)
                self.type_attention = nn.Parameter(torch.ones(num_channels) / num_channels)
        self.linear_fc = nn.Sequential(
            LinearPerMetapath(num_channels, embed_dim, hidden),
            nn.LayerNorm([num_channels, hidden]),
            nn.PReLU(),
            nn.Dropout(dropout),
            LinearPerMetapath(num_channels, hidden, hidden),
            nn.LayerNorm([num_channels, hidden]),
            nn.PReLU(),
            nn.Dropout(dropout)
        )

        self.concat_fc = nn.Linear(num_channels * hidden, hidden)
        if self.residual:
            self.res_fc = nn.Linear(embed_dim, hidden)
        if self.dataset not in ['IMDB', 'AMiner']:
            self.out_fc = nn.Sequential(
                *([nn.PReLU(),
                   nn.Dropout(dropout)]
                  + list(itertools.chain.from_iterable([
                    [nn.Linear(hidden, hidden),
                     nn.BatchNorm1d(hidden, affine=False),
                     nn.PReLU(),
                     nn.Dropout(dropout)] for _ in range(out_layers_num - 1)]))
                  + [nn.Linear(hidden, num_classes),
                     nn.BatchNorm1d(num_classes, affine=False, track_running_stats=False)]
                  )
            )
        else:
            self.out_fc = nn.ModuleList(
                [nn.Sequential(
                    nn.PReLU(),
                    nn.Dropout(dropout))]
                + [nn.Sequential(
                    nn.Linear(hidden, hidden),
                    nn.BatchNorm1d(hidden, affine=False),
                    nn.PReLU(),
                    nn.Dropout(dropout)) for _ in range(out_layers_num - 1)]
                + [nn.Sequential(
                    nn.Linear(hidden, num_classes),
                    nn.LayerNorm(num_classes, elementwise_affine=False))]
            )
        self.reset_parameters()

    def reset_parameters(self):
        for k, v in self._modules.items():
            if isinstance(v, nn.ParameterDict):
                for _k, _v in v.items():
                    _v.data.uniform_(-0.5, 0.5)
            elif isinstance(v, nn.ModuleList):
                for block in v:
                    if isinstance(block, nn.Sequential):
                        for layer in block:
                            if hasattr(layer, 'reset_parameters'):
                                layer.reset_parameters()
                    elif hasattr(block, 'reset_parameters'):
                        block.reset_parameters()
            elif isinstance(v, nn.Sequential):
                for layer in v:
                    if hasattr(layer, 'reset_parameters'):
                        layer.reset_parameters()
            elif hasattr(v, 'reset_parameters'):
                v.reset_parameters()

    def forward(self, feats_dict):
        feats = {k: self.input_drop(x @ self.feat_fc[k]) for k, x in feats_dict.items()}

        if self.attention_module:
            weights = []
            for i, k in enumerate(self.feat_keys):
                path_length = len(k)
                path_type = 1 if k[0] == k[-1] else 0
                if self.dataset != 'AMiner':
                    weight = self.length_attention[i] * path_length + self.type_attention[i] * path_type
                else:
                    weight = 1 + self.length_attention[i] * path_length / 4 + self.type_attention[i] * path_type
                weight = torch.clamp(weight, min=0)
                weights.append(weight)
            for i, (k, x) in enumerate(feats.items()):
                feats[k] = x * weights[i]

        out = [feats[k] for k, x in feats.items()]
        out = torch.stack(out, dim=1)
        out = self.linear_fc(out)
        out = self.concat_fc(self.flatten(out.transpose(1, 2)))
        if self.residual:
            out = out + self.res_fc(feats[self.tgt_type])
        if self.dataset not in ['IMDB', 'AMiner']:
            out = self.out_fc(out)
        else:
            out = self.out_fc[0](out)
            for i in range(1, len(self.out_fc) - 1):
                out = self.out_fc[i](out) + out
            out = self.out_fc[-1](out)

        return out

    def get_weights(self):
        with torch.no_grad():
            weights = []
            for i, k in enumerate(self.feat_keys):
                path_length = len(k)
                path_type = int(k[0] == k[-1])
                weight = self.length_attention[i] * path_length + self.type_attention[i] * path_type
                weight = torch.clamp(weight, min=0)
                weights.append(weight)

        return weights
