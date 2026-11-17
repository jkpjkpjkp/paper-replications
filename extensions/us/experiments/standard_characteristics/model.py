"""Original LiMT with only the active input projection resized for CZ."""
from torch import nn
from limt.model import LiMT


class Model(LiMT):
    def __init__(self, feature_count, **settings):
        super().__init__(**settings)
        if feature_count != 158:
            self.net.feature_layer[0] = nn.Linear(feature_count, self.net.feature_layer[0].out_features)
