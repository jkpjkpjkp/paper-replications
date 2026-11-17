import torch
import torch.nn.functional as F
import dgl.function as fn
import sympy
import scipy
import dgl.nn.pytorch.conv as dglnn
import dgl
from torch import nn
from scipy.special import comb
import math
import copy
import numpy as np


class PolyConv(nn.Module):
    def __init__(self, theta):
        super(PolyConv, self).__init__()
        self._theta = theta
        self._k = len(self._theta)

    def forward(self, graph, feat):
        def unnLaplacian(feat, D_invsqrt, graph):
            """Operation Feat * D^-1/2 A D^-1/2"""
            graph.ndata["h"] = feat * D_invsqrt
            graph.update_all(fn.copy_u("h", "m"), fn.sum("m", "h"))
            return feat - graph.ndata.pop("h") * D_invsqrt

        with graph.local_scope():
            D_invsqrt = (
                torch.pow(graph.in_degrees().float().clamp(min=1), -0.5)
                .unsqueeze(-1)
                .to(feat.device)
            )
            h = self._theta[0] * feat
            for k in range(1, self._k):
                feat = unnLaplacian(feat, D_invsqrt, graph)
                h += self._theta[k] * feat
        return h


class BernConv(nn.Module):
    def __init__(self, orders=2):
        super().__init__()
        self.K = orders
        self.weight = nn.Parameter(torch.ones(orders + 1))

    def forward(self, graph, feat):
        def unnLaplacian1(feat, D_invsqrt, graph):
            """\hat{L} X"""
            graph.ndata["h"] = feat * D_invsqrt
            graph.update_all(fn.copy_u("h", "m"), fn.sum("m", "h"))
            return feat - graph.ndata.pop("h") * D_invsqrt

        def unnLaplacian2(feat, D_invsqrt, graph):
            """(2I - \hat{L}) X"""
            graph.ndata["h"] = feat * D_invsqrt
            graph.update_all(fn.copy_u("h", "m"), fn.sum("m", "h"))
            return feat + graph.ndata.pop("h") * D_invsqrt

        with graph.local_scope():
            tmp = [feat]
            weight = nn.functional.relu(self.weight)
            D_invsqrt = (
                torch.pow(graph.in_degrees().float().clamp(min=1), -0.5)
                .unsqueeze(-1)
                .to(feat.device)
            )
            for i in range(self.K):
                feat = unnLaplacian2(feat, D_invsqrt, graph)
                tmp.append(feat)

            out_feat = (comb(self.K, 0) / (2**self.K)) * weight[0] * tmp[self.K]
            for i in range(self.K):
                x = tmp[self.K - i - 1]
                for j in range(i + 1):
                    x = unnLaplacian1(feat, D_invsqrt, graph)
                out_feat = (
                    out_feat + (comb(self.K, i + 1) / (2**self.K)) * weight[i + 1] * x
                )
        return out_feat


class BernNet(nn.Module):
    def __init__(
        self,
        in_feats,
        h_feats=32,
        num_classes=2,
        orders=2,
        mlp_layers=1,
        dropout_rate=0,
        activation="ReLU",
        **kwargs,
    ):
        super().__init__()
        self.bernconv = BernConv(orders=orders)
        self.linear1 = nn.Linear(in_feats, h_feats)
        self.linear2 = nn.Linear(h_feats, h_feats)
        self.mlp = MLP(
            h_feats, h_feats, num_classes, mlp_layers, dropout_rate=dropout_rate
        )
        self.act = getattr(nn, activation)()

    def forward(self, graph):
        in_feat = graph.ndata["feature"]
        h = self.linear1(in_feat)
        h = self.act(h)
        h = self.linear2(h)
        h = self.act(h)
        h = self.bernconv(graph, h)
        h = self.act(h)
        h = self.mlp(h, False)
        return h


class AMNet(nn.Module):
    def __init__(
        self,
        in_feats,
        h_feats=32,
        num_classes=2,
        orders=2,
        num_layers=3,
        mlp_layers=1,
        dropout_rate=0,
        activation="ReLU",
        **kwargs,
    ):
        super().__init__()
        self.linear1 = nn.Linear(in_feats, h_feats)
        self.linear2 = nn.Linear(h_feats, h_feats)
        self.mlp_layers = nn.Sequential()
        if mlp_layers > 0:
            for i in range(mlp_layers - 1):
                self.mlp_layers.append(nn.Linear(h_feats, h_feats))
            self.mlp_layers.append(nn.Linear(h_feats, num_classes))
        self.act = getattr(nn, activation)()
        self.attn = nn.Tanh()
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None
        self.num_layers = num_layers
        self.layers = nn.Sequential()
        for i in range(num_layers):
            self.layers.append(BernConv(orders))
        self.linear_transform_in = nn.Sequential(self.linear1, self.act, self.linear2)
        self.W_f = nn.Sequential(self.linear2, self.attn)
        self.W_x = nn.Sequential(self.linear2, self.attn)
        self.linear_cls_out = nn.Sequential(self.mlp_layers)

    def forward(self, graph):
        in_feat = graph.ndata["feature"]
        h = self.linear_transform_in(in_feat)
        h_list = []
        for i, layer in enumerate(self.layers):
            h_ = layer(graph, h)
            h_list.append(h_)
        h_filters = torch.stack(h_list, dim=1)
        h_filters_proj = self.W_f(h_filters)
        x_proj = self.W_x(h).unsqueeze(-1)
        score_logit = torch.bmm(h_filters_proj, x_proj)
        soft_score = F.softmax(score_logit, dim=1)
        score = soft_score
        res = h_filters[:, 0, :] * score[:, 0]
        for i in range(1, self.num_layers):
            res += h_filters[:, i, :] * score[:, i]
        y_hat = self.linear_cls_out(res)
        return y_hat


def calculate_theta(d):
    thetas = []
    x = sympy.symbols("x")
    for i in range(d + 1):
        f = sympy.poly(
            (x / 2) ** i
            * (1 - x / 2) ** (d - i)
            / (scipy.special.beta(i + 1, d + 1 - i))
        )
        coeff = f.all_coeffs()
        inv_coeff = []
        for i in range(d + 1):
            inv_coeff.append(float(coeff[d - i]))
        thetas.append(inv_coeff)
    return thetas


class BWGNN(nn.Module):
    def __init__(
        self,
        in_feats,
        h_feats=32,
        num_classes=2,
        num_layers=2,
        mlp_layers=2,
        dropout_rate=0,
        activation="ReLU",
        **kwargs,
    ):
        super(BWGNN, self).__init__()
        self.thetas = calculate_theta(d=num_layers)
        self.conv = []
        for i in range(len(self.thetas)):
            self.conv.append(PolyConv(self.thetas[i]))
        self.linear = nn.Linear(in_feats, h_feats)
        self.linear2 = nn.Linear(h_feats, h_feats)
        self.mlp = MLP(
            h_feats * len(self.conv), h_feats, num_classes, mlp_layers, dropout_rate
        )
        self.act = getattr(nn, activation)()
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, graph):
        in_feat = graph.ndata["feature"]
        h = self.linear(in_feat)
        h = self.act(h)
        h = self.linear2(h)
        h = self.act(h)
        h_final = torch.zeros([len(in_feat), 0], device=h.device)

        for conv in self.conv:
            h0 = conv(graph, h)
            h_final = torch.cat([h_final, h0], -1)
        h_final = self.dropout(h_final)
        h = self.mlp(h_final, False)
        return h


class GCN(nn.Module):
    def __init__(
        self,
        in_feats,
        h_feats=32,
        num_classes=2,
        num_layers=2,
        mlp_layers=1,
        dropout_rate=0,
        activation="ReLU",
        **kwargs,
    ):
        super().__init__()
        self.h_feats = h_feats
        self.layers = nn.ModuleList()
        self.act = getattr(nn, activation)()
        self.layers.append(dglnn.GraphConv(in_feats, h_feats, activation=self.act))
        for i in range(num_layers - 1):
            self.layers.append(dglnn.GraphConv(h_feats, h_feats, activation=self.act))
        self.mlp = MLP(h_feats, h_feats, num_classes, mlp_layers, dropout_rate)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, graph):
        h = graph.ndata["feature"]
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(graph, h)
        h = self.mlp(h, False)
        return h


class RGCN(nn.Module):
    def __init__(
        self,
        in_feats,
        h_feats=32,
        num_classes=2,
        num_layers=2,
        mlp_layers=1,
        dropout_rate=0,
        activation="ReLU",
        etypes=None,
        **kwargs,
    ):
        super().__init__()
        self.h_feats = h_feats
        self.layers = nn.ModuleList()
        self.act = getattr(nn, activation)()
        self.layers.append(
            dgl.nn.HeteroGraphConv(
                {
                    (st, et, dt): dglnn.GraphConv(
                        in_feats, h_feats, activation=self.act
                    )
                    for (st, et, dt) in etypes
                },
                aggregate="mean",
            )
        )
        self.type = etypes[0][0]
        for i in range(num_layers - 1):
            self.layers.append(
                dgl.nn.HeteroGraphConv(
                    {
                        (st, et, dt): dglnn.GraphConv(
                            h_feats, h_feats, activation=self.act
                        )
                        for (st, et, dt) in etypes
                    },
                    aggregate="mean",
                )
            )
        self.mlp = MLP(h_feats, h_feats, num_classes, mlp_layers, dropout_rate)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, graph):
        h = {self.type: graph.ndata["feature"]}
        for i, layer in enumerate(self.layers):
            # print(layer)
            if i != 0:
                h[self.type] = self.dropout(h[self.type])
            # print(graph, h)
            h = layer(graph, h)
        h = self.mlp(h[self.type], False)
        return h


class HGT(nn.Module):
    def __init__(
        self,
        in_feats,
        h_feats=32,
        num_heads=1,
        num_classes=2,
        num_layers=2,
        mlp_layers=1,
        dropout_rate=0.2,
        activation="ReLU",
        etypes=None,
        **kwargs,
    ):
        super().__init__()
        self.h_feats = h_feats
        self.layers = nn.ModuleList()
        self.act = getattr(nn, activation)()
        self.input_linear = nn.Linear(in_feats, h_feats)
        self.type = etypes[0][0]
        for i in range(num_layers):
            self.layers.append(
                dgl.nn.HGTConv(
                    h_feats,
                    h_feats // num_heads,
                    num_heads,
                    1,
                    len(etypes),
                    dropout=dropout_rate,
                )
            )
        self.mlp = MLP(h_feats, h_feats, num_classes, mlp_layers, dropout_rate)
        self.graph = None

    def forward(self, graph):
        if self.graph is None:
            self.graph = dgl.to_homogeneous(graph, ndata=["feature"])
        h = self.graph.ndata["feature"]
        graph = self.graph
        h = self.input_linear(h)
        for i, layer in enumerate(self.layers):
            h = layer(graph, h, graph.ndata[dgl.NTYPE], graph.edata[dgl.ETYPE])
            h = self.act(h)
        h = self.mlp(h, False)
        # print( graph.ndata[dgl.NTYPE], graph.edata[dgl.ETYPE])
        return h


class GIN(nn.Module):
    def __init__(
        self,
        in_feats,
        h_feats=32,
        num_classes=2,
        num_layers=2,
        agg="mean",
        dropout_rate=0,
        activation="ReLU",
        **kwargs,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        self.act = getattr(nn, activation)()
        self.layers.append(
            dglnn.GINConv(
                nn.Linear(in_feats, h_feats), activation=self.act, aggregator_type=agg
            )
        )
        for i in range(1, num_layers - 1):
            self.layers.append(
                dglnn.GINConv(
                    nn.Linear(h_feats, h_feats),
                    activation=self.act,
                    aggregator_type=agg,
                )
            )
        self.layers.append(
            dglnn.GINConv(
                nn.Linear(h_feats, num_classes), activation=None, aggregator_type=agg
            )
        )
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, graph):
        h = graph.ndata["feature"]
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(graph, h)
        return h


class GIN_noparam(nn.Module):
    def __init__(self, num_layers=2, agg="mean", init_eps=-1, **kwargs):
        super().__init__()
        self.gnn = dglnn.GINConv(
            None, activation=None, init_eps=init_eps, aggregator_type=agg
        )
        self.num_layers = num_layers

    def forward(self, graph):
        h = graph.ndata["feature"]
        h_final = h.detach().clone()
        for i in range(self.num_layers):
            h = self.gnn(graph, h)
            h_final = torch.cat([h_final, h], -1)
        print(h_final)
        return h_final


class ChebNet(nn.Module):
    def __init__(
        self,
        in_feats,
        h_feats=32,
        num_classes=2,
        num_layers=2,
        dropout_rate=0,
        activation="ReLU",
        **kwargs,
    ):
        super().__init__()
        self.input_linear = nn.Linear(in_feats, h_feats)
        self.act = getattr(nn, activation)()
        self.chebconv = dglnn.ChebConv(
            h_feats, h_feats, num_layers, activation=self.act
        )
        self.output_linear = nn.Linear(h_feats, num_classes)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, graph):
        h = graph.ndata["feature"]
        h = self.input_linear(h)
        h = self.act(h)
        h = self.dropout(h)
        h = self.chebconv(graph, h, lambda_max=[2])
        h = self.dropout(h)
        h = self.output_linear(h)
        return h


class GraphSAGE(nn.Module):
    def __init__(
        self,
        in_feats,
        h_feats=32,
        num_classes=2,
        num_layers=2,
        agg="pool",
        dropout_rate=0,
        activation="ReLU",
        **kwargs,
    ):
        super(GraphSAGE, self).__init__()
        self.layers = nn.ModuleList()
        self.act = getattr(nn, activation)()
        self.layers.append(dglnn.SAGEConv(in_feats, h_feats, agg, activation=self.act))
        for i in range(num_layers - 1):
            self.layers.append(
                dglnn.SAGEConv(h_feats, h_feats, agg, activation=self.act)
            )
        self.output_linear = nn.Linear(h_feats, num_classes)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, graph):
        h = graph.ndata["feature"]
        for layer in self.layers:
            h = self.dropout(h)
            h = layer(graph, h)
        h = self.output_linear(h)
        return h


class MLP(nn.Module):
    def __init__(
        self,
        in_feats,
        h_feats=32,
        num_classes=2,
        num_layers=2,
        dropout_rate=0,
        activation="ReLU",
        **kwargs,
    ):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.act = getattr(nn, activation)()
        if num_layers == 0:
            return
        if num_layers == 1:
            self.layers.append(nn.Linear(in_feats, num_classes))
        else:
            self.layers.append(nn.Linear(in_feats, h_feats))
            for i in range(1, num_layers - 1):
                self.layers.append(nn.Linear(h_feats, h_feats))
            self.layers.append(nn.Linear(h_feats, num_classes))
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, h, is_graph=True):
        if is_graph:
            h = h.ndata["feature"]
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(h)
            if i != len(self.layers) - 1:
                h = self.act(h)
        return h


class SGC(nn.Module):
    def __init__(
        self,
        in_feats,
        h_feats=32,
        num_classes=2,
        k=2,
        mlp_layers=1,
        dropout_rate=0,
        **kwargs,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        self.mlp = None
        if mlp_layers == 1:
            self.sgc = dglnn.SGConv(in_feats, num_classes, k=k)
        else:
            self.sgc = dglnn.SGConv(in_feats, h_feats, k=k)
            self.mlp = MLP(
                h_feats, num_classes, num_classes, mlp_layers - 1, dropout_rate
            )
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, graph):
        h = graph.ndata["feature"]
        h = self.dropout(h)
        h = self.sgc(graph, h)
        if self.mlp is not None:
            h = self.mlp(h, False)
        return h


class DGI_GIN(nn.Module):
    def __init__(
        self, in_feats, h_feats=32, num_layers=2, agg="mean", dropout_rate=0, **kwargs
    ):
        super().__init__()
        self.h_feats = h_feats
        self.layers = nn.ModuleList()
        self.layers.append(
            dglnn.GINConv(
                nn.Linear(in_feats, h_feats), activation=F.relu, aggregator_type=agg
            )
        )
        for i in range(num_layers - 1):
            self.layers.append(
                dglnn.GINConv(
                    nn.Linear(h_feats, h_feats), activation=F.relu, aggregator_type=agg
                )
            )
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None

    def forward(self, graph, h=None):
        if h is None:
            h = graph.ndata["feature"]
        for i, layer in enumerate(self.layers):
            if i != 0 and self.dropout:
                h = self.dropout(h)
            h = layer(graph, h)
        return h


class DCI_Encoder(nn.Module):
    def __init__(
        self,
        in_feats,
        h_feats=32,
        num_classes=2,
        num_layers=2,
        mlp_layers=2,
        dropout_rate=0,
        agg="mean",
        **kwargs,
    ):
        super().__init__()
        self.conv = DGI_GIN(in_feats, h_feats, num_layers, agg, dropout_rate)
        self.mlp = MLP(h_feats, h_feats, num_classes, mlp_layers, dropout_rate)

    def forward(self, graph, corrupt=False, use_mlp=False):
        h = graph.ndata["feature"]
        if corrupt:
            perm = torch.randperm(graph.num_nodes())
            h = h[perm]
        h = self.conv(graph, h)
        if use_mlp:
            h = self.mlp(h, is_graph=False)
        return h


class Discriminator(nn.Module):
    def __init__(self, h_feats, **kwargs):
        super(Discriminator, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(h_feats, h_feats))
        self.reset_parameters()

    def uniform(self, size, tensor):
        bound = 1.0 / math.sqrt(size)
        if tensor is not None:
            tensor.data.uniform_(-bound, bound)

    def reset_parameters(self):
        size = self.weight.size(0)
        self.uniform(size, self.weight)

    def forward(self, features, summary):
        features = torch.matmul(features, torch.matmul(self.weight, summary))
        return features


class DCI(nn.Module):
    def __init__(
        self,
        in_feats,
        h_feats=32,
        num_classes=2,
        num_layers=2,
        mlp_layers=1,
        dropout_rate=0,
        **kwargs,
    ):
        super().__init__()
        self.encoder = DCI_Encoder(
            in_feats, h_feats, num_classes, num_layers, mlp_layers, dropout_rate
        )
        self.discriminator = Discriminator(h_feats)
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, graph, cluster_info, cluster_num):
        positive = self.encoder(graph, corrupt=False)
        negative = self.encoder(graph, corrupt=True)
        loss = 0
        for i in range(cluster_num):
            node_idx = cluster_info[i]

            positive_block = torch.unsqueeze(positive[node_idx], 0)
            negative_block = torch.unsqueeze(negative[node_idx], 0)
            summary = torch.sigmoid(positive.mean(dim=0))

            negative_block = self.discriminator(negative_block, summary)
            positive_block = self.discriminator(positive_block, summary)

            l1 = self.loss(positive_block, torch.ones_like(positive_block))
            l2 = self.loss(negative_block, torch.zeros_like(negative_block))

            loss_tmp = l1 + l2
            loss += loss_tmp

        return loss / cluster_num

    def get_emb(self, graph):
        h = self.encoder(graph, corrupt=False)
        return h


class PrincipalAggregate(nn.Module):
    def __init__(
        self,
        aggregators=["mean", "max", "min", "std"],
        h_feats=32,
        act="ReLU",
        dropout=0.0,
    ):
        super().__init__()
        self.aggregators = aggregators
        self.agg_funcs = [getattr(self, f"agg_{agg}") for agg in aggregators]
        self.linear = nn.Linear(len(self.agg_funcs) * h_feats, h_feats)
        self.act = getattr(nn, act)()

    def forward(self, mfg, feat):
        h = [agg(mfg, feat) for agg in self.agg_funcs]
        return self.act(self.linear(torch.cat(h, dim=1)))

    def agg_mean(self, mfg, X):
        mfg.srcdata["h"] = X
        mfg.update_all(fn.copy_u("h", "m"), fn.mean("m", "h"))
        return mfg.dstdata["h"]

    def agg_min(self, mfg, X):
        mfg.srcdata["h"] = X
        mfg.update_all(fn.copy_u("h", "m"), fn.min("m", "h"))
        return mfg.dstdata["h"]

    def agg_max(self, mfg, X):
        mfg.srcdata["h"] = X
        mfg.update_all(fn.copy_u("h", "m"), fn.max("m", "h"))
        return mfg.dstdata["h"]

    def agg_std(self, mfg, X):
        diff = self.agg_mean(mfg, X**2) - self.agg_mean(mfg, X) ** 2
        return torch.sqrt(F.relu(diff) + 1e-5)

    def agg_sum(self, mfg, X):
        mfg.srcdata["h"] = X
        mfg.update_all(fn.copy_u("h", "m"), fn.sum("m", "h"))
        return mfg.dstdata["h"]

    def __repr__(self):
        return f"""PrincipalAggregate(
            aggregators={self.aggregators}
            linear={self.linear}
        )"""


class PNA(nn.Module):
    def __init__(
        self,
        in_feats,
        h_feats=32,
        num_classes=2,
        num_layers=2,
        activation="ReLU",
        dropout_rate=0,
        **kwargs,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        # self.layers.append(nn.Linear(in_feats, h_feats))
        self.input_linear = nn.Linear(in_feats, h_feats)

        self.act = getattr(nn, activation)()
        self.output_linear = nn.Linear(h_feats, num_classes)
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None

        for i in range(0, num_layers):
            self.layers.append(
                PrincipalAggregate(
                    h_feats=h_feats, act=activation, dropout=dropout_rate
                )
            )

    def forward(self, graph):
        h = graph.ndata["feature"]
        h = self.input_linear(h)
        for i, layer in enumerate(self.layers):
            if i != 0 and self.dropout:
                h = self.dropout(h)
            h = layer(graph, h)
        h = self.output_linear(h)
        return h


class CAREConv(nn.Module):
    def __init__(
        self,
        in_feats,
        h_feats,
        num_classes=2,
        activation=None,
        step_size=0.02,
        **kwargs,
    ):
        super().__init__()
        self.activation = activation
        self.step_size = step_size
        self.in_feats = in_feats
        self.h_feats = h_feats
        self.num_classes = num_classes
        self.dist = {}
        self.linear = nn.Linear(self.in_feats, self.h_feats)
        self.MLP = nn.Linear(self.in_feats, self.num_classes)
        self.p = {}
        self.last_avg_dist = {}
        self.f = {}
        self.cvg = {}

    def _calc_distance(self, edges):
        # formula 2
        d = torch.norm(
            torch.tanh(self.MLP(edges.src["h"])) - torch.tanh(self.MLP(edges.dst["h"])),
            1,
            1,
        )
        return {"d": d}

    def _top_p_sampling(self, graph, p):
        # Compute the number of neighbors to keep for each node
        in_degrees = graph.in_degrees()
        num_neigh = torch.ceil(in_degrees.float() * p).int()

        # Fetch all edges and their distances
        all_edges = graph.edges(form="eid")
        dist = graph.edata["d"]

        # Create a prefix sum array for in-degrees to use for indexing
        prefix_sum = torch.cat([torch.tensor([0]).cuda(), in_degrees.cumsum(0)[:-1]])

        # Get the edges for each node using advanced indexing
        selected_edges = []
        for i, node_deg in enumerate(num_neigh):
            start_idx = prefix_sum[i]
            end_idx = start_idx + node_deg
            sorted_indices = torch.argsort(dist[start_idx:end_idx])[:node_deg]
            selected_edges.append(all_edges[start_idx:end_idx][sorted_indices])
        return torch.cat(selected_edges)

    def forward(self, graph, epoch=0):
        feat = graph.ndata["feature"]
        edges = graph.canonical_etypes
        if epoch == 0:
            for etype in edges:
                self.p[etype] = 0.5
                self.last_avg_dist[etype] = 0
                self.f[etype] = []
                self.cvg[etype] = False

        with graph.local_scope():
            graph.ndata["h"] = feat

            hr = {}
            for i, etype in enumerate(edges):
                graph.apply_edges(self._calc_distance, etype=etype)
                self.dist[etype] = graph.edges[etype].data["d"]
                sampled_edges = self._top_p_sampling(graph[etype], self.p[etype])

                # formula 8
                graph.send_and_recv(
                    sampled_edges,
                    fn.copy_u("h", "m"),
                    fn.mean("m", "h_%s" % etype[1]),
                    etype=etype,
                )
                hr[etype] = graph.ndata["h_%s" % etype[1]]
                if self.activation is not None:
                    hr[etype] = self.activation(hr[etype])

            # formula 9 using mean as inter-relation aggregator
            p_tensor = (
                torch.Tensor(list(self.p.values())).view(-1, 1, 1).to(graph.device)
            )
            h_homo = torch.sum(torch.stack(list(hr.values())) * p_tensor, dim=0)
            h_homo += feat
            if self.activation is not None:
                h_homo = self.activation(h_homo)

            return self.linear(h_homo)


class CAREGNN(nn.Module):
    def __init__(
        self,
        in_feats,
        num_classes=2,
        h_feats=64,
        edges=None,
        num_layers=1,
        activation=None,
        step_size=0.02,
        **kwargs,
    ):
        super().__init__()
        self.in_feats = in_feats
        self.h_feats = h_feats
        self.num_classes = num_classes
        self.activation = None if activation is None else getattr(nn, activation)()
        self.step_size = step_size
        self.num_layers = num_layers
        self.output_linear = nn.Linear(h_feats, num_classes)
        self.layers = nn.ModuleList()
        self.layers.append(  # Input layer
            CAREConv(
                self.in_feats,
                self.num_classes,
                self.num_classes,
                activation=self.activation,
                step_size=self.step_size,
            )
        )
        for i in range(self.num_layers - 1):  # Hidden layers with n - 2 layers
            self.layers.append(
                CAREConv(
                    self.h_feats,
                    self.h_feats,
                    self.num_classes,
                    activation=self.activation,
                    step_size=self.step_size,
                )
            )
            # self.layers.append(   # Output layer
            # CAREConv(self.h_feats, self.num_classes, self.num_classes, activation=self.activation, step_size=self.step_size,))

    def forward(self, graph, epoch=0):
        for layer in self.layers:
            feat = layer(graph, epoch)
        return feat

    def RLModule(self, graph, epoch, idx):
        for layer in self.layers:
            for etype in graph.canonical_etypes:
                if not layer.cvg[etype]:
                    # formula 5
                    eid = graph.in_edges(idx, form="eid", etype=etype)
                    avg_dist = torch.mean(layer.dist[etype][eid])

                    # formula 6
                    if layer.last_avg_dist[etype] < avg_dist:
                        if layer.p[etype] - self.step_size > 0:
                            layer.p[etype] -= self.step_size
                        layer.f[etype].append(-1)
                    else:
                        if layer.p[etype] + self.step_size <= 1:
                            layer.p[etype] += self.step_size
                        layer.f[etype].append(+1)
                    layer.last_avg_dist[etype] = avg_dist

                    # formula 7
                    if epoch >= 9 and abs(sum(layer.f[etype][-10:])) <= 2:
                        layer.cvg[etype] = True


class H2FDetector_layer(nn.Module):
    def __init__(
        self, in_feats, h_feats, head, relation_aware, etype, dropout_rate, if_sum=False
    ):
        super().__init__()
        self.etype = etype
        self.head = head
        self.hd = h_feats
        self.if_sum = if_sum
        self.relation_aware = relation_aware
        self.w_liner = nn.Linear(in_feats, h_feats * head)
        self.atten = nn.Linear(2 * self.hd, 1)
        self.relu = nn.ReLU()
        self.leakyrelu = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, graph, h):
        with graph.local_scope():
            graph.ndata["feat"] = h
            graph.apply_edges(self.sign_edges, etype=self.etype)
            h = self.w_liner(h)
            graph.ndata["h"] = h
            graph.update_all(
                message_func=self.message, reduce_func=self.reduce, etype=self.etype
            )
            out = graph.ndata["out"]
            return out

    def message(self, edges):
        src = edges.src
        src_features = edges.data["sign"].view(-1, 1) * src["h"]
        src_features = src_features.view(-1, self.head, self.hd)
        z = torch.cat(
            [src_features, edges.dst["h"].view(-1, self.head, self.hd)], dim=-1
        )
        alpha = self.atten(z)
        alpha = self.leakyrelu(alpha)
        return {"atten": alpha, "sf": src_features}

    def reduce(self, nodes):
        alpha = nodes.mailbox["atten"]
        sf = nodes.mailbox["sf"]
        alpha = self.softmax(alpha)
        out = torch.sum(alpha * sf, dim=1)
        if not self.if_sum:
            out = out.view(-1, self.head * self.hd)
        else:
            out = out.sum(dim=-2)
        return {"out": out}

    def sign_edges(self, edges):
        src = edges.src["feat"]
        dst = edges.dst["feat"]
        score = self.relation_aware(src, dst)
        return {"sign": torch.sign(score)}


class RelationAware(nn.Module):
    def __init__(self, in_feats, h_feats, dropout_rate):
        super().__init__()
        self.d_liner = nn.Linear(in_feats, h_feats)
        self.f_liner = nn.Linear(3 * h_feats, 1)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None

    def forward(self, src, dst):
        src = self.d_liner(src)
        dst = self.d_liner(dst)
        diff = src - dst
        e_feats = torch.cat([src, dst, diff], dim=1)
        if self.dropout is not None:
            e_feats = self.dropout(e_feats)
        score = self.f_liner(e_feats).squeeze()
        score = self.tanh(score)
        return score


def hinge_loss(labels, scores):
    margin = 1
    ls = labels * scores

    loss = F.relu(margin - ls)
    loss = loss.mean()
    return loss


class MultiRelationH2FDetectorLayer(nn.Module):
    def __init__(self, in_feats, h_feats, head, dataset, dropout_rate, if_sum=False):
        super().__init__()
        self.relation = copy.deepcopy(dataset.etypes)
        self.n_relation = len(self.relation)
        if not if_sum:
            self.liner = nn.Linear(self.n_relation * h_feats * head, h_feats * head)
        else:
            self.liner = nn.Linear(self.n_relation * h_feats, h_feats)
        self.relation_aware = RelationAware(in_feats, h_feats * head, dropout_rate)
        self.minelayers = nn.ModuleDict()
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        for e in self.relation:
            self.minelayers[e] = H2FDetector_layer(
                in_feats, h_feats, head, self.relation_aware, e, dropout_rate, if_sum
            )

    def forward(self, graph, h):
        hs = []
        for e in self.relation:
            he = self.minelayers[e](graph, h)
            hs.append(he)
        h = torch.cat(hs, dim=1)
        if self.dropout is not None:
            h = self.dropout(h)

        h = self.liner(h)
        return h

    def loss(self, graph, h):
        with graph.local_scope():
            graph.ndata["feat"] = h
            agg_h = self.forward(graph, h)

            graph.apply_edges(self.score_edges, etype="homo")
            edges_score = graph.edges["homo"].data["score"]
            edge_train_mask = graph.edges["homo"].data["train_mask"].bool()
            edge_train_label = graph.edges["homo"].data["label"][edge_train_mask]
            edge_train_pos = edge_train_label == 1
            edge_train_neg = edge_train_label == -1
            edge_train_pos_index = (
                edge_train_pos.nonzero().flatten().detach().cpu().numpy()
            )
            edge_train_neg_index = (
                edge_train_neg.nonzero().flatten().detach().cpu().numpy()
            )
            edge_train_pos_index = np.random.choice(
                edge_train_pos_index, size=len(edge_train_neg_index)
            )
            index = np.concatenate([edge_train_pos_index, edge_train_neg_index])
            index.sort()
            edge_train_score = edges_score[edge_train_mask]
            # hinge loss
            edge_diff_loss = hinge_loss(
                edge_train_label[index], edge_train_score[index]
            )

            train_mask = graph.ndata["train_mask"].bool()
            train_h = agg_h[train_mask]
            train_label = graph.ndata["label"][train_mask]
            train_pos = train_label == 1
            train_neg = train_label == 0
            train_pos_index = train_pos.nonzero().flatten().detach().cpu().numpy()
            train_neg_index = train_neg.nonzero().flatten().detach().cpu().numpy()
            train_neg_index = np.random.choice(
                train_neg_index, size=len(train_pos_index)
            )
            node_index = np.concatenate([train_neg_index, train_pos_index])
            node_index.sort()
            pos_prototype = torch.mean(train_h[train_pos], dim=0).view(1, -1)
            neg_prototype = torch.mean(train_h[train_neg], dim=0).view(1, -1)
            train_h_loss = train_h[node_index]
            pos_prototypes = pos_prototype.expand(train_h_loss.shape)
            neg_prototypes = neg_prototype.expand(train_h_loss.shape)
            diff_pos = -F.pairwise_distance(train_h_loss, pos_prototypes)
            diff_neg = -F.pairwise_distance(train_h_loss, neg_prototypes)
            diff_pos = diff_pos.view(-1, 1)
            diff_neg = diff_neg.view(-1, 1)
            diff = torch.cat([diff_neg, diff_pos], dim=1)
            diff_loss = F.cross_entropy(diff, train_label[node_index])

            return agg_h, edge_diff_loss, diff_loss

    def score_edges(self, edges):
        src = edges.src["feat"]
        dst = edges.dst["feat"]
        score = self.relation_aware(src, dst)
        return {"score": score}


class H2FD(nn.Module):
    def __init__(
        self,
        in_feats,
        graph,
        n_layer=1,
        intra_dim=16,
        n_class=2,
        gamma1=1.2,
        gamma2=2,
        head=2,
        dropout_rate=0.1,
        **kwargs,
    ):
        super().__init__()
        self.in_feats = in_feats
        self.n_layer = n_layer
        self.intra_dim = intra_dim
        self.n_class = n_class
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.head = head
        self.dropout_rate = dropout_rate
        self.mine_layers = nn.ModuleList()
        if n_layer == 1:
            self.mine_layers.append(
                MultiRelationH2FDetectorLayer(
                    self.in_feats, self.n_class, head, graph, dropout_rate, if_sum=True
                )
            )
        else:
            self.mine_layers.append(
                MultiRelationH2FDetectorLayer(
                    self.in_feats, self.intra_dim, head, graph, dropout_rate
                )
            )
            for _ in range(1, self.n_layer - 1):
                self.mine_layers.append(
                    MultiRelationH2FDetectorLayer(
                        self.intra_dim * head, self.intra_dim, head, graph, dropout_rate
                    )
                )
            self.mine_layers.append(
                MultiRelationH2FDetectorLayer(
                    self.intra_dim * head,
                    self.n_class,
                    head,
                    graph,
                    dropout_rate,
                    if_sum=True,
                )
            )
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.relu = nn.ReLU()

    def forward(self, graph):
        feats = graph.ndata["feature"].float()
        train_mask = graph.ndata["train_mask"].bool()
        train_label = graph.ndata["label"][train_mask]
        train_pos = train_label == 1
        train_neg = train_label == 0

        pos_index = train_pos.nonzero().flatten().detach().cpu().numpy()
        neg_index = train_neg.nonzero().flatten().detach().cpu().numpy()
        neg_index = np.random.choice(neg_index, size=len(pos_index), replace=False)
        index = np.concatenate([pos_index, neg_index])
        index.sort()
        h, edge_loss, prototype_loss = self.mine_layers[0].loss(graph, feats)
        if self.n_layer > 1:
            h = self.relu(h)
            h = self.dropout(h)
            for i in range(1, len(self.mine_layers) - 1):
                h, e_loss, p_loss = self.mine_layers[i].loss(graph, h)
                h = self.relu(h)
                h = self.dropout(h)
                edge_loss += e_loss
                prototype_loss += p_loss
            h, e_loss, p_loss = self.mine_layers[-1].loss(graph, h)
            edge_loss += e_loss
            prototype_loss += p_loss
        model_loss = F.cross_entropy(h[train_mask][index], train_label[index])
        loss = model_loss + self.gamma1 * edge_loss + self.gamma2 * prototype_loss
        return loss, h


# =============================================================================
# FraudGNAM: INherent Spectral Interaction Graph neTwork
# =============================================================================


def _get_beta_coeffs(order):
    """Compute Beta Wavelet polynomial coefficients."""
    thetas = []
    for i in range(order + 1):
        coeffs = [0.0] * (order + 1)
        for k in range(order - i + 1):
            sign = (-1) ** k
            binom_outer = comb(order, i, exact=True)
            binom_inner = comb(order - i, k, exact=True)
            coeff = sign * binom_outer * binom_inner
            power = i + k
            coeffs[power] += coeff
        thetas.append(coeffs)
    return thetas


class _FeatureNet(nn.Module):
    """Individual feature network for NAM."""

    def __init__(self, hidden_dims=[64, 32], dropout=0.1):
        super().__init__()
        layers = []
        in_dim = 1
        for h_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(in_dim, h_dim),
                    nn.LayerNorm(h_dim),
                    nn.LeakyReLU(0.1),
                    nn.Dropout(dropout),
                ]
            )
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class _FeatureNetGrouped(nn.Module):
    """Global feature network processing all features together."""

    def __init__(self, num_features, group_size=10, hidden_dim=32, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x)


class _PairwiseNet(nn.Module):
    """Pairwise interaction network."""

    def __init__(self, hidden_dim=32, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x_i, x_j):
        return self.net(torch.cat([x_i, x_j, x_i * x_j], dim=-1))


class _BandNet(nn.Module):
    """Spectral band processing network."""

    def __init__(self, num_features, hidden_dim=64, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        return self.net(x)


class _EFINet(nn.Module):
    """Explicit Feature Interaction."""

    def __init__(self, num_features, hidden_dim=64, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, band_features, original_features):
        return self.net(band_features * original_features)


class _SpectralFilter(nn.Module):
    """Apply polynomial spectral filter with learnable coefficients."""

    def __init__(self, coeffs):
        super().__init__()
        # Learnable coefficients initialized from Beta wavelet basis
        self.coeffs = nn.Parameter(torch.tensor(coeffs, dtype=torch.float32))

    def forward(self, L_powers):
        h = torch.zeros_like(L_powers[0])
        for k in range(len(self.coeffs)):
            if k < len(L_powers):
                h = h + self.coeffs[k] * L_powers[k]
        return h


class _HighPassFilter(nn.Module):
    """Iterative high-pass filter: feat = feat - D^{-1/2}AD^{-1/2} feat.

    Applies the normalized Laplacian K times to extract high-frequency
    (heterophilic) graph signals. Equivalent to SEC-GFD's HighConv.
    """

    def __init__(self, iters=1):
        super().__init__()
        self.iters = iters

    def forward(self, graph, feat):
        with graph.local_scope():
            device = feat.device
            D_invsqrt = (
                graph.in_degrees()
                .float()
                .clamp(min=1)
                .to(device)
                .pow(-0.5)
                .unsqueeze(-1)
            )
            h = feat
            for _ in range(self.iters):
                graph.ndata["h"] = h * D_invsqrt
                graph.update_all(fn.copy_u("h", "m"), fn.sum("m", "h"))
                h = h - graph.ndata.pop("h") * D_invsqrt
            return h


class _BandNetEmbed(nn.Module):
    """Spectral band processing network outputting h_feats-dim embeddings."""

    def __init__(self, num_features, hidden_dim=64, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class _EFINetEmbed(nn.Module):
    """EFI network outputting h_feats-dim embeddings."""

    def __init__(self, num_features, hidden_dim=64, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
        )

    def forward(self, band_features, original_features):
        return self.net(band_features * original_features)


class FraudGNAM(nn.Module):
    """
    FraudGNAM: INherent Spectral Interaction Graph neTwork

    DGL-compatible version for GADBench integration.
    Inherently interpretable GNN combining:
    - NAM (individual feature contributions)
    - GA²M (pairwise feature interactions)
    - Beta Wavelet Spectral Filtering (multi-scale graph decomposition)
    - EFI (Explicit Feature Interaction: band ⊙ x)
    """

    def __init__(
        self,
        in_feats,
        h_feats=64,
        num_classes=2,
        order=2,
        feature_hidden=None,
        num_pairwise=15,
        use_pairwise=True,
        use_efi=True,
        drop_rate=0.1,
        use_grouped_features=False,
        feature_group_size=10,
        self_loop_weight=1.0,
        num_highpass=0,
        use_nce=False,
        use_concat_integration=False,
        **kwargs,
    ):
        super().__init__()
        self.in_feats = in_feats
        self.num_classes = num_classes
        self.h_feats = h_feats
        self.order = order
        self.num_bands = order + 1
        self.num_pairwise = min(num_pairwise, in_feats * (in_feats - 1) // 2)
        self.use_pairwise = use_pairwise
        self.use_efi = use_efi
        self.self_loop_weight = self_loop_weight
        self.use_grouped_features = use_grouped_features
        self.num_highpass = num_highpass
        self.use_nce = use_nce
        self.use_concat_integration = use_concat_integration

        if feature_hidden is None:
            feature_hidden = [64, 32]

        # Feature networks: use grouped for high-dim features to reduce params
        if use_grouped_features or in_feats > 100:
            self.feature_net_grouped = _FeatureNetGrouped(
                in_feats, feature_group_size, h_feats, drop_rate
            )
            self.feature_nets = None
        else:
            self.feature_net_grouped = None
            self.feature_nets = nn.ModuleList(
                [
                    _FeatureNet(hidden_dims=feature_hidden, dropout=drop_rate)
                    for _ in range(in_feats)
                ]
            )

        # Pairwise networks
        if use_pairwise and self.num_pairwise > 0:
            self.pairwise_nets = nn.ModuleList(
                [
                    _PairwiseNet(hidden_dim=h_feats // 2, dropout=drop_rate)
                    for _ in range(self.num_pairwise)
                ]
            )
            self.pair_indices = nn.Parameter(
                torch.zeros(self.num_pairwise, 2, dtype=torch.long), requires_grad=False
            )
            self._init_pair_indices()
        else:
            self.pairwise_nets = None

        # Spectral filters
        coeffs_list = _get_beta_coeffs(order)
        self.spectral_filters = nn.ModuleList(
            [_SpectralFilter(coeffs) for coeffs in coeffs_list]
        )

        # Band & EFI networks
        self.band_nets = nn.ModuleList(
            [_BandNet(in_feats, h_feats, drop_rate) for _ in range(self.num_bands)]
        )
        if use_efi:
            self.efi_nets = nn.ModuleList(
                [_EFINet(in_feats, h_feats, drop_rate) for _ in range(self.num_bands)]
            )
        else:
            self.efi_nets = None

        # High-pass spectral branch (for heterophilic edges)
        if num_highpass > 0:
            self.highpass_filters = nn.ModuleList(
                [_HighPassFilter(iters=k + 1) for k in range(num_highpass)]
            )
            self.highpass_nets = nn.ModuleList(
                [_BandNet(in_feats, h_feats, drop_rate) for _ in range(num_highpass)]
            )
            self.highpass_efi_nets = (
                nn.ModuleList(
                    [_EFINet(in_feats, h_feats, drop_rate) for _ in range(num_highpass)]
                )
                if use_efi
                else None
            )
            self.highpass_scale = nn.Parameter(torch.tensor(0.5))
            self.highpass_weights = nn.Parameter(torch.ones(num_highpass))
        else:
            self.highpass_filters = None
            self.highpass_nets = None
            self.highpass_efi_nets = None

        # NCE contrastive branch (2-layer GCN, outputs in_feats for cosine sim with original features)
        if use_nce:
            self.nce_conv1 = dglnn.GraphConv(
                in_feats, h_feats, allow_zero_in_degree=True
            )
            self.nce_conv2 = dglnn.GraphConv(
                h_feats, in_feats, allow_zero_in_degree=True
            )
        else:
            self.nce_conv1 = None
            self.nce_conv2 = None

        # Concat+MLP integration (replaces scalar-sum when enabled)
        if use_concat_integration:
            # Count components: feature(1) + bands(num_bands) + efi(num_bands if use_efi) + highpass(num_highpass) + highpass_efi(num_highpass if use_efi)
            n_components = 1 + self.num_bands  # feature + bands
            if use_efi:
                n_components += self.num_bands  # efi
            if num_highpass > 0:
                n_components += num_highpass  # highpass bands
                if use_efi:
                    n_components += num_highpass  # highpass efi
            if use_pairwise and self.num_pairwise > 0:
                n_components += 1  # pairwise (pooled to 1 embedding)

            # Embed-dim variants for concat path
            self.band_nets_embed = nn.ModuleList(
                [
                    _BandNetEmbed(in_feats, h_feats, drop_rate)
                    for _ in range(self.num_bands)
                ]
            )
            if use_efi:
                self.efi_nets_embed = nn.ModuleList(
                    [
                        _EFINetEmbed(in_feats, h_feats, drop_rate)
                        for _ in range(self.num_bands)
                    ]
                )
            else:
                self.efi_nets_embed = None
            if num_highpass > 0:
                self.highpass_nets_embed = nn.ModuleList(
                    [
                        _BandNetEmbed(in_feats, h_feats, drop_rate)
                        for _ in range(num_highpass)
                    ]
                )
                if use_efi:
                    self.highpass_efi_nets_embed = nn.ModuleList(
                        [
                            _EFINetEmbed(in_feats, h_feats, drop_rate)
                            for _ in range(num_highpass)
                        ]
                    )
                else:
                    self.highpass_efi_nets_embed = None
            else:
                self.highpass_nets_embed = None
                self.highpass_efi_nets_embed = None

            # Feature embedding: reuse grouped/individual net but project to h_feats
            self.feature_embed_proj = nn.Linear(1, h_feats)
            # Pairwise embedding projection
            if use_pairwise and self.num_pairwise > 0:
                self.pairwise_embed_proj = nn.Linear(1, h_feats)

            self.integration_mlp = nn.Sequential(
                nn.Linear(n_components * h_feats, h_feats),
                nn.ReLU(),
                nn.Dropout(drop_rate),
                nn.Linear(h_feats, 1),
            )
        else:
            self.band_nets_embed = None
            self.efi_nets_embed = None
            self.highpass_nets_embed = None
            self.highpass_efi_nets_embed = None
            self.integration_mlp = None

        # Learnable scales (used in scalar-sum path)
        self.output_bias = nn.Parameter(torch.zeros(1))
        self.feature_scale = nn.Parameter(torch.tensor(1.0))
        self.band_scale = nn.Parameter(torch.tensor(1.0))
        self.efi_scale = nn.Parameter(torch.tensor(0.5))
        self.pairwise_scale = nn.Parameter(torch.tensor(0.5))
        self.band_weights = nn.Parameter(torch.ones(self.num_bands))
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def _init_pair_indices(self):
        idx = 0
        for i in range(self.in_feats):
            for j in range(i + 1, self.in_feats):
                if idx < self.num_pairwise:
                    self.pair_indices.data[idx, 0] = i
                    self.pair_indices.data[idx, 1] = j
                    idx += 1

    def set_important_pairs(self, features, labels):
        """
        Select top-M feature pairs by absolute Pearson correlation with labels.
        Call once before training with training data.
        Replaces the sequential default from _init_pair_indices().
        """
        if not self.use_pairwise or self.pairwise_nets is None:
            return

        x_np = features.detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy().astype(float)

        pair_scores = []
        for i in range(self.in_feats):
            for j in range(i + 1, self.in_feats):
                x_prod = x_np[:, i] * x_np[:, j]
                corr = np.abs(np.corrcoef(x_prod, labels_np)[0, 1])
                if np.isnan(corr):
                    corr = 0.0
                pair_scores.append((i, j, corr))

        pair_scores.sort(key=lambda t: t[2], reverse=True)

        for idx, (i, j, _) in enumerate(pair_scores[: self.num_pairwise]):
            self.pair_indices.data[idx, 0] = i
            self.pair_indices.data[idx, 1] = j

        top_pairs = pair_scores[: self.num_pairwise]
        print(
            f"  Pairwise: selected top-{len(top_pairs)} pairs by label correlation "
            f"(max r={top_pairs[0][2]:.3f}, min r={top_pairs[-1][2]:.3f})"
        )

    def _compute_laplacian_powers_dgl(self, graph, x):
        """Compute (L/2)^k * x using stable BernConv-style Laplacian."""
        with graph.local_scope():
            device = x.device
            degs = graph.in_degrees().float().clamp(min=1).to(device)
            D_invsqrt = degs.pow(-0.5).unsqueeze(-1)

            def unnLaplacian(feat):
                """Normalized Laplacian: L_hat * feat = feat - D^{-1/2} A D^{-1/2} feat"""
                graph.ndata["h"] = feat * D_invsqrt
                graph.update_all(fn.copy_u("h", "m"), fn.sum("m", "h"))
                return feat - graph.ndata.pop("h") * D_invsqrt

            # Compute L/2 powers: (L_hat/2)^k * x
            powers = [x]
            L_k = x
            for _ in range(self.order):
                L_k = unnLaplacian(L_k) / 2.0
                powers.append(L_k)
            return powers

    def forward(self, graph):
        x = graph.ndata["feature"]
        num_nodes = x.size(0)
        device = x.device

        # 1. Feature contributions
        if self.feature_net_grouped is not None:
            total_feature = self.feature_net_grouped(x)
        else:
            total_feature = torch.zeros(num_nodes, 1, device=device)
            for i, f_net in enumerate(self.feature_nets):
                total_feature = total_feature + f_net(x[:, i : i + 1])

        # 2. Pairwise contributions
        total_pairwise = torch.zeros(num_nodes, 1, device=device)
        if self.use_pairwise and self.pairwise_nets is not None:
            for idx, p_net in enumerate(self.pairwise_nets):
                i, j = self.pair_indices[idx]
                total_pairwise = total_pairwise + p_net(
                    x[:, i : i + 1], x[:, j : j + 1]
                )

        # 3. Spectral band + EFI contributions (cache band_x for reuse)
        L_powers = self._compute_laplacian_powers_dgl(graph, x)
        band_x_cache = []
        total_band = torch.zeros(num_nodes, 1, device=device)
        total_efi = torch.zeros(num_nodes, 1, device=device)

        band_weights = F.softmax(self.band_weights, dim=0) * self.num_bands
        for j, spec_filter in enumerate(self.spectral_filters):
            band_x = spec_filter(L_powers)
            band_x_cache.append(band_x)
            total_band = total_band + self.band_nets[j](band_x) * band_weights[j]
            if self.use_efi and self.efi_nets is not None:
                total_efi = total_efi + self.efi_nets[j](band_x, x) * band_weights[j]

        # 4. High-pass spectral contributions (cache hp_x for reuse)
        hp_x_cache = []
        total_highpass = torch.zeros(num_nodes, 1, device=device)
        total_highpass_efi = torch.zeros(num_nodes, 1, device=device)
        if self.highpass_filters is not None:
            hp_weights = F.softmax(self.highpass_weights, dim=0) * self.num_highpass
            for k, hp_filter in enumerate(self.highpass_filters):
                hp_x = hp_filter(graph, x)
                hp_x_cache.append(hp_x)
                total_highpass = (
                    total_highpass + self.highpass_nets[k](hp_x) * hp_weights[k]
                )
                if self.use_efi and self.highpass_efi_nets is not None:
                    total_highpass_efi = (
                        total_highpass_efi
                        + self.highpass_efi_nets[k](hp_x, x) * hp_weights[k]
                    )

        # 5. Combine
        if self.use_concat_integration and self.integration_mlp is not None:
            # Concat+MLP path: collect h_feats-dim embeddings (reuse cached band_x/hp_x)
            embeds = []

            # Feature embedding
            embeds.append(self.feature_embed_proj(total_feature))  # (N, h_feats)

            # Band embeddings (reuse cached band_x)
            for j in range(self.num_bands):
                embeds.append(self.band_nets_embed[j](band_x_cache[j]))
                if self.use_efi and self.efi_nets_embed is not None:
                    embeds.append(self.efi_nets_embed[j](band_x_cache[j], x))

            # High-pass embeddings (reuse cached hp_x)
            if (
                self.highpass_filters is not None
                and self.highpass_nets_embed is not None
            ):
                for k in range(self.num_highpass):
                    embeds.append(self.highpass_nets_embed[k](hp_x_cache[k]))
                    if self.use_efi and self.highpass_efi_nets_embed is not None:
                        embeds.append(self.highpass_efi_nets_embed[k](hp_x_cache[k], x))

            # Pairwise embedding
            if self.use_pairwise and self.pairwise_nets is not None:
                embeds.append(self.pairwise_embed_proj(total_pairwise))

            concat_emb = torch.cat(embeds, dim=-1)  # (N, n_components * h_feats)
            fraud_logit = self.integration_mlp(concat_emb).squeeze(-1)
        else:
            # Scalar-sum path (original)
            fraud_logit = (
                self.output_bias
                + self.feature_scale * total_feature.squeeze(-1)
                + self.pairwise_scale * total_pairwise.squeeze(-1)
                + self.band_scale * total_band.squeeze(-1)
                + self.efi_scale * total_efi.squeeze(-1)
            )
            if self.highpass_filters is not None:
                fraud_logit = fraud_logit + self.highpass_scale * (
                    total_highpass.squeeze(-1) + total_highpass_efi.squeeze(-1)
                )

        fraud_logit = fraud_logit / self.temperature.clamp(min=0.1)
        logits = torch.stack([-fraud_logit / 2, fraud_logit / 2], dim=1)

        # NCE embeddings (training only)
        if self.use_nce and self.nce_conv1 is not None and self.training:
            nce_h = F.relu(self.nce_conv1(graph, x))
            nce_emb = self.nce_conv2(graph, nce_h)
            return logits, nce_emb

        return logits
