"""Sequence adapters for existing LSTM and vendored public Qlib backbones.

The GRUModel/TCNModel class bodies retain installed Qlib 0.9.7 definitions.
They are public reference implementations, not the authors' private original
experimental code. Provenance and MIT notices live in vendor/qlib_backbones/.
"""
from __future__ import annotations

from torch import nn

from core import Forecaster
from vendor.qlib_backbones import GRUModel, TCNModel


class QlibSequenceAdapter(nn.Module):
    """Convert date tensors N,T,F to Qlib's feature-major flattened N,F*T."""
    def __init__(self, backbone, features, patches):
        super().__init__()
        self.backbone = backbone
        self.features = features
        self.patches = patches

    def forward(self, x):
        if x.ndim != 3 or x.shape[-1] != self.features or x.shape[1] < 1:
            raise ValueError(f'Expected N x T x {self.features} with T >= 1')
        flat = x.transpose(1, 2).contiguous().flatten(1)
        # Qlib uses squeeze(), which otherwise removes a singleton batch axis.
        return self.backbone(flat).reshape(-1)


def _functional_tcn(backbone):
    """Keep the same modules/parameters with one registered path per conv.

    Qlib registers each weighted convolution twice: TemporalBlock.conv1/conv2
    and TemporalBlock.net[0]/net[4] reference the same objects. In Torch 2.0,
    functional_call can restore duplicated module paths in the wrong order,
    leaving graph-bearing look-ahead tensors installed as model parameters.
    Keep net's execution path registered and retain conv1/conv2 as ordinary
    object aliases. No convolution, weight-normalization hook, g/v parameter,
    initial value or forward computation changes. Upstream class bodies remain
    unmodified, including init_weights' ability to access conv1 and conv2.
    """
    for block in backbone.tcn.network:
        for name in ['conv1', 'conv2']:
            convolution = block._modules.pop(name)
            object.__setattr__(block, name, convolution)
    return backbone


def effective_model_config(config):
    """Resolve the constructor choices recorded with each trained backbone."""
    model = config.get('model', {})
    name = str(model.get('name', 'LSTM')).upper()
    features = int(model.get('input_features', config.get('input_features', 66)))
    patches = int(model.get('patches', config.get('patches', 26)))
    hidden = int(model.get('hidden_size', 64))
    layers = int(model.get('layers', 1))
    dropout = float(model.get('dropout', 0.0))
    if min(features, patches, hidden, layers) < 1 or not 0 <= dropout < 1:
        raise ValueError('Features, patches, width and layers must be positive; dropout in [0,1)')
    effective = {'name': name, 'input_features': features, 'patches': patches,
                 'hidden_size': hidden, 'layers': layers, 'dropout': dropout}
    if name == 'TCN':
        levels = int(model.get('tcn_layers', 3))
        kernel = int(model.get('tcn_kernel_size', 3))
        if levels < 1 or kernel < 2:
            raise ValueError('Vendored causal TCN requires tcn_layers >= 1 and tcn_kernel_size >= 2')
        effective.update(tcn_layers=levels, tcn_kernel_size=kernel,
                         receptive_field_patches=1 + 2 * (kernel - 1) * (2**levels - 1))
        effective['implementation'] = 'Qlib 0.9.7 TCNModel + TemporalConvNet'
        effective['functional_call_adapter'] = 'Keep original legacy g/v weight normalization; register each convolution once under net and keep conv1/conv2 as ordinary object aliases to avoid Torch2.0 duplicate-module restoration bugs'
        effective['choice_note'] = 'Three two-convolution dilated blocks, kernel3, cover 29 patches; independent choice because paper omits exact sizes'
    elif name == 'GRU':
        effective['implementation'] = 'Qlib 0.9.7 GRUModel'
    elif name == 'LSTM':
        effective['implementation'] = 'Existing core.Forecaster'
    else:
        raise ValueError(f'Unknown model name {name!r}; supported: LSTM, GRU, TCN')
    effective['input_adapter'] = ('identity N,T,F' if name == 'LSTM' else
                                  'N,T,F -> transpose(1,2) -> contiguous feature-major N,F*T')
    return effective


def make_forecaster(config):
    """Build LSTM/GRU/TCN with N,T,F -> N and recorded effective settings.

    model.input_features or config.input_features sets F (default 66).
    config.patches documents the intended T (default 26); the models accept
    other positive sequence lengths. TCN uses its separate tcn_layers=3 and
    tcn_kernel_size=3 defaults, giving a receptive field of 29 patches.
    Every returned module exposes a serializable backbone_config dictionary.
    """
    settings = effective_model_config(config)
    name = settings['name']
    features, hidden = settings['input_features'], settings['hidden_size']
    layers, dropout = settings['layers'], settings['dropout']
    if name == 'LSTM':
        result = Forecaster(features=features, hidden_size=hidden, layers=layers, dropout=dropout)
    else:
        if name == 'GRU':
            backbone = GRUModel(d_feat=features, hidden_size=hidden, num_layers=layers, dropout=dropout)
        else:
            backbone = _functional_tcn(TCNModel(num_input=features, output_size=1,
                                num_channels=[hidden] * settings['tcn_layers'],
                                kernel_size=settings['tcn_kernel_size'], dropout=dropout))
        result = QlibSequenceAdapter(backbone, features, settings['patches'])
    result.backbone_config = settings
    return result
