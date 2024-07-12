from torch import nn

from base.prim.Activator import GELU
from base.prim.Linear import Linear
from base.util.Log import Logger

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layers = nn.Sequential(
            Linear(config.embed_dim, config.embed_dim_ff_dim),
            GELU(),
            nn.Dropout(config.drop_rate),
            Linear(config.embed_dim_ff_dim, config.embed_dim),
            nn.Dropout(config.drop_rate)
        )
        self.log = Logger.get_instance()
    def forward(self, x):
        self.log.debug("FF Linear 1st:", self.config.embed_dim, self.config.embed_dim_ff_dim)
        self.log.debug("FF Linear 2nd:", self.config.embed_dim_ff_dim, self.config.embed_dim)

        return self.layers(x)