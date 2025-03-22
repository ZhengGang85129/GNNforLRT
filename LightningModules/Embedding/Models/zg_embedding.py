# System imports
import sys
import os

# 3rd party imports
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from ..embedding_base import EmbeddingBase
from torch.nn import Linear
import torch.nn as nn
from torch_cluster import radius_graph
import torch
from torch_geometric.data import DataLoader

# Local imports
from ..utils import graph_intersection


class ZGEmbedding(EmbeddingBase):
    def __init__(self, hparams):
        super().__init__(hparams)
        """
        Initialise the Lightning Module that can scan over different embedding training regimes
        """

        # Construct the MLP architecture
        layers = []
        in_channels = hparams["in_channels"] 
        print(hparams["emb_hidden"])
        for emb_hidden in hparams["emb_hidden"]: 
            layers.append(nn.Linear(in_channels, emb_hidden))
            layers.append(nn.LayerNorm(emb_hidden))
            layers.append(nn.Tanh())
            in_channels = emb_hidden 
        self.layers = nn.ModuleList(layers)
        self.emb_layer = nn.Linear(hparams["emb_hidden"][-1], hparams["emb_dim"])
        #self.norm = nn.LayerNorm(hparams["emb_hidden"][-1])
        self.save_hyperparameters()

    def forward(self, x):
        #         hits = self.normalize(hits)
        for l in self.layers:
            x = l(x)
        x = self.emb_layer(x)
        return x
