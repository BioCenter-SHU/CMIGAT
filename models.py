""" Componets of the model
"""
import itertools
import random

import torch.nn as nn
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv

def xavier_init(m):
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
           m.bias.data.fill_(0.0)

class GAT(torch.nn.Module):
    def __init__(self, in_feats, hgcn_dim,dropout):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_feats, hgcn_dim[0], heads=1,concat=False)
        # self.conv2 = GATConv(hgcn_dim[0], hgcn_dim[1], heads=1,concat=False)
        # self.conv3 = GATConv(hgcn_dim[1], hgcn_dim[2], heads=1,concat=False)
        self.decoder = nn.Sequential(
            # nn.Linear(hgcn_dim[2], hgcn_dim[1]),
            # nn.LeakyReLU(0.25),
            # nn.Linear(hgcn_dim[1], hgcn_dim[0]),
            # nn.LeakyReLU(0.25),
            nn.Linear(hgcn_dim[0], in_feats),
        )
        self.decoder.apply(xavier_init)
        self.dropout = dropout

    def forward(self, data,adj, return_reconstruction=False):
        x= data
        edge_index =adj
        x = self.conv1(x, edge_index)
        x = F.leaky_relu(x, 0.25)
        x = F.dropout(x, self.dropout, training=self.training)
        # x = self.conv2(x, edge_index)
        # x = F.leaky_relu(x, 0.25)
        # x = F.dropout(x, self.dropout, training=self.training)
        # x = self.conv3(x, edge_index)
        # x = F.leaky_relu(x, 0.25)
        # x = F.dropout(x, self.dropout, training=self.training)

        if return_reconstruction:
            recon = self.decoder(x)
            return x, recon
        return x




class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        nn.init.xavier_normal_(self.weight.data)
        if self.bias is not None:
            self.bias.data.fill_(0.0)

    def forward(self, x, adj):
        support = torch.mm(x, self.weight)
        # ensure dtype match between sparse adj and dense support
        if adj.dtype != support.dtype:
            adj = adj.to(dtype=support.dtype)
        output = torch.sparse.mm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class LinearLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.clf = nn.Sequential(nn.Linear(in_dim, out_dim))
        self.clf.apply(xavier_init)

    def forward(self, x):
        x = self.clf(x)
        return x


class TCP(nn.Module):
    def __init__(
            self,
            in_dim,
            hidden_dim,
            num_class,
            dropout,
            enable_confidence=True,
            fusion_backend="attention",
            shared_config=None,
            fusion_config=None,
            shared_loss_weights=None,
            use_specific_branch=True,
            use_specific_features=True,
    ):
        super().__init__()
        self.views = len(in_dim)
        self.classes = num_class
        self.dropout = dropout
        self.enable_confidence = enable_confidence
        self.fusion_backend = (fusion_backend or "attention").lower()
        self.fusion_config = fusion_config.copy() if fusion_config else {}
        # 控制特异性分支是否参与训练 / 融合
        self.use_specific_branch = use_specific_branch
        self.use_specific_features = use_specific_features

        base_hidden = hidden_dim[0]
        if enable_confidence:
            self.TCPConfidenceLayer = nn.ModuleList([LinearLayer(base_hidden, 1) for _ in range(self.views)])
        else:
            self.TCPConfidenceLayer = None
        self.TCPClassifierLayer = nn.ModuleList([LinearLayer(base_hidden, num_class) for _ in range(self.views)])

        # Shared representation module -------------------------------------------------
        self.shared_module = None
        self.shared_loss_weights = {"sem": 0.0, "mmd": 0.0}
        if shared_loss_weights:
            for key in self.shared_loss_weights:
                if key in shared_loss_weights:
                    self.shared_loss_weights[key] = shared_loss_weights[key]

        if shared_config:
            shared_cfg = shared_config.copy()
            if "input_dims" not in shared_cfg:
                shared_cfg["input_dims"] = in_dim
            shared_cfg.setdefault("latent_dim", base_hidden)
            shared_cfg.setdefault("target_dim", max(shared_cfg["input_dims"]))
            self.shared_module = SharedLatentEncoder(**shared_cfg)
        self.shared_node_count = self.shared_module.num_views if self.shared_module is not None else 0
        self.shared_latent_dim = self.shared_module.latent_dim if self.shared_module is not None else 0

        # Fusion backends -------------------------------------------------------------
        default_modes = ["CMIG"]
        self.available_graph_modes = self.fusion_config.get("graph_mode_options", default_modes)
        self.feature_integrator = None
        self.paired_attentionLayer = None
        effective_specific = self.views if self.use_specific_features else 0
        backend = self.fusion_backend
        if backend == "gat":
            fusion_dim = self.fusion_config.get("fusion_dim", base_hidden)
            num_heads = self.fusion_config.get("num_heads", 1)
            num_layers = self.fusion_config.get("num_layers", 1)
            fusion_dropout = self.fusion_config.get("dropout", dropout)
            readout = self.fusion_config.get("readout", "mean")
            graph_mode = "CMIG"
            shared_dim = self.shared_module.latent_dim if self.shared_module is not None else base_hidden
            self.feature_integrator = MultiFeatureGATFusion(
                specific_dim=base_hidden,
                shared_dim=shared_dim,
                fusion_dim=fusion_dim,
                num_layers=num_layers,
                num_heads=num_heads,
                dropout=fusion_dropout,
                graph_mode=graph_mode,
                readout=readout,
                available_graph_modes=self.available_graph_modes,
                random_k_neighbors=self.fusion_config.get("random_k_neighbors"),
                random_k_seed=self.fusion_config.get("random_k_seed", 42),
            )
            # fusion_input_dim depends on readout strategy; flatten keeps all node embeddings.
            fusion_input_dim = self.feature_integrator.output_dim(
                num_specific=effective_specific,
                num_shared=self.shared_node_count,
            )
        elif backend == "fc_concat":
            concat_shared_dim = self.shared_latent_dim if self.shared_latent_dim > 0 else 0
            fusion_input_dim = effective_specific * base_hidden + self.shared_node_count * concat_shared_dim
        elif backend == "paired_attention":
            if self.shared_module is None or self.shared_node_count == 0:
                raise ValueError("paired_attention backend requires shared representations")
            pair_dim = base_hidden + self.shared_latent_dim
            # If paired_attention is still needed later, you will need a self attention mechanism. 
            # We remove it completely now as requested to focus purely on gat CMIG mode.
            raise ValueError("paired_attention fusion backend requires SelfAttention which is removed")
        else:
            raise ValueError(f"Unsupported fusion backend: {fusion_backend}")

        fusion_layers = []
        prev_dim = fusion_input_dim
        for dim in hidden_dim[1:]:
            fusion_layers.append(LinearLayer(prev_dim, dim))
            fusion_layers.append(nn.ReLU())
            fusion_layers.append(nn.Dropout(p=dropout))
            prev_dim = dim
        fusion_layers.append(LinearLayer(prev_dim, num_class))
        self.MMClasifier = nn.Sequential(*fusion_layers)

    def forward(self, feature, label=None, infer=False, raw_inputs=None, return_details=False, **kwargs):
        criterion = torch.nn.CrossEntropyLoss(reduction='none')
        per_view_logits, per_view_confidence = {}, {}
        processed_features = []
        if self.use_specific_branch or self.use_specific_features:
            for view in range(self.views):
                view_feat = F.relu(feature[view])
                view_feat = F.dropout(view_feat, self.dropout, training=self.training)
                if self.use_specific_branch:
                    logits = self.TCPClassifierLayer[view](view_feat)
                    per_view_logits[view] = logits
                    if self.enable_confidence:
                        conf = self.TCPConfidenceLayer[view](view_feat)
                        per_view_confidence[view] = conf
                        # 只有在特征也参与融合时才进行置信度缩放
                        if self.use_specific_features:
                            view_feat = view_feat * conf
                if self.use_specific_features:
                    processed_features.append(view_feat)

        shared_latents = []
        shared_losses = {}
        if self.shared_module is not None:
            if raw_inputs is None:
                raise ValueError("raw_inputs must be provided when shared_module is enabled")
            shared_output = self.shared_module(raw_inputs)
            shared_latents = shared_output["z_list"]
            shared_losses = shared_output["losses"]
        fusion_inputs = processed_features + shared_latents
        if len(fusion_inputs) == 0:
            raise ValueError("No features available for fusion. Enable specific_features or provide shared_module outputs.")

        backend = self.fusion_backend
        if backend == "gat":
            graph_override = kwargs.get("graph_mode_override")
            fusion_vec, node_outputs = self.feature_integrator(
                processed_features,
                shared_latents,
                graph_mode_override=graph_override,
            )
            MMfeature = fusion_vec
        elif backend == "fc_concat":
            if not fusion_inputs:
                raise ValueError("No features available for fusion")
            MMfeature = torch.cat(fusion_inputs, dim=1)
            node_outputs = None
        elif backend == "paired_attention":
            raise ValueError("paired_attention fusion backend requires SelfAttention which is removed")
        else:
            raise ValueError(f"Unsupported fusion backend: {backend}")

        MMlogit = self.MMClasifier(MMfeature)
        if infer:
            return MMlogit

        total_loss = torch.mean(criterion(MMlogit, label))
        loss_breakdown = {"fusion_cls": total_loss.detach().cpu().item()}

        if self.use_specific_branch:
            for view in range(self.views):
                per_view_loss = torch.mean(criterion(per_view_logits[view], label))
                if self.enable_confidence:
                    pred = F.softmax(per_view_logits[view], dim=1)
                    p_target = torch.gather(input=pred, dim=1, index=label.unsqueeze(dim=1)).view(-1)
                    confidence_loss = torch.mean(
                        F.mse_loss(per_view_confidence[view].view(-1), p_target) + criterion(per_view_logits[view], label))
                    total_loss = total_loss + confidence_loss
                    loss_breakdown[f"confidence_{view+1}"] = confidence_loss.detach().cpu().item()
                else:
                    total_loss = total_loss + per_view_loss
                loss_breakdown[f"view_cls_{view+1}"] = per_view_loss.detach().cpu().item()

        for key, value in shared_losses.items():
            weight = self.shared_loss_weights.get(key, 0.0)
            loss_breakdown[f"L_{key}"] = value.detach().cpu().item()
            if weight > 0.0:
                total_loss = total_loss + weight * value
        if return_details:
            aux = {"loss_breakdown": loss_breakdown}
            if node_outputs is not None:
                aux["fusion_nodes_mean"] = node_outputs.detach().mean().cpu().item()
            return total_loss, MMlogit, aux

        return total_loss, MMlogit

    def infer(self, data_list, raw_inputs=None, **kwargs):
        return self.forward(data_list, infer=True, raw_inputs=raw_inputs, **kwargs)

    def shared_parameters(self):
        params = []
        if self.shared_module is not None:
            params += list(self.shared_module.parameters())
        if self.fusion_backend == "gat" and self.feature_integrator is not None:
            params += list(self.feature_integrator.parameters())
        params += list(self.MMClasifier.parameters())
        return params

    def available_graph_strategies(self):
        if self.fusion_backend != "gat" or self.feature_integrator is None:
            return []
        return self.feature_integrator.available_graph_modes

class Classifier_1(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.clf = nn.Sequential(nn.Linear(in_dim, out_dim))
        self.clf.apply(xavier_init)

    def forward(self, x):
        x = self.clf(x)
        return x

class FeatureAligner(nn.Module):
    """Align heterogeneous omics feature sizes before shared encoding."""

    def __init__(self, input_dims, mode="linear", target_dim=None):
        super().__init__()
        self.input_dims = input_dims
        self.mode = mode
        if target_dim is None:
            target_dim = max(input_dims)
        self.target_dim = target_dim
        if mode not in {"linear", "none"}:
            raise ValueError(f"Unsupported alignment mode: {mode}")
        if mode == "linear":
            self.projections = nn.ModuleList([nn.Linear(dim, target_dim) for dim in input_dims])
            self.projections.apply(xavier_init)
        else:
            self.projections = None
        self.output_dim = target_dim

    def forward(self, inputs):
        aligned = []
        for idx, tensor in enumerate(inputs):
            if self.mode == "linear":
                aligned.append(self.projections[idx](tensor))
            else:  # none
                if tensor.size(1) != self.target_dim:
                    raise ValueError("All inputs must already share the target dimension when mode='none'")
                aligned.append(tensor)
        return aligned


class SharedLatentEncoder(nn.Module):
    """Shared encoder that maps multi-omics inputs into a unified latent space."""

    def __init__(
            self,
            input_dims,
            latent_dim,
            encoder_hidden=None,
            align_mode="linear",
            target_dim=None,
            dropout=0.1,
            mmd_kernel_bandwidth=2.0,
            use_sem_loss=True,
            use_mmd_loss=True,
    ):
        super().__init__()
        self.input_dims = input_dims
        self.num_views = len(input_dims)
        self.latent_dim = latent_dim
        self.align_mode = align_mode
        self.aligner = FeatureAligner(input_dims, mode=align_mode, target_dim=target_dim)
        hidden_dim = encoder_hidden or max(latent_dim * 2, 64)
        self.encoder = nn.Sequential(
            nn.Linear(self.aligner.output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
        )
        self.encoder.apply(xavier_init)
        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)
        self.mu_head.apply(xavier_init)
        self.logvar_head.apply(xavier_init)

        self.use_sem_loss = use_sem_loss
        self.use_mmd_loss = use_mmd_loss
        self.mmd_bandwidth = mmd_kernel_bandwidth

    def forward(self, inputs, detach_inputs=False):
        data = [x.detach() if detach_inputs else x for x in inputs]
        aligned = self.aligner(data)
        mu_list, logvar_list, z_list = [], [], []
        for tensor in aligned:
            hidden = self.encoder(tensor)
            mu = self.mu_head(hidden)
            logvar = self.logvar_head(hidden)
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std) if self.training else torch.zeros_like(std)
            z = mu + eps * std
            mu_list.append(mu)
            logvar_list.append(logvar)
            z_list.append(z)

        losses = {}
        if self.use_sem_loss and len(mu_list) > 1:
            losses["sem"] = self._semantic_alignment(mu_list, logvar_list)
        if self.use_mmd_loss and len(z_list) > 1:
            losses["mmd"] = self._pairwise_mmd(z_list)

        return {
            "aligned": aligned,
            "mu": mu_list,
            "logvar": logvar_list,
            "z_list": z_list,
            "losses": losses,
        }

    def _semantic_alignment(self, mu_list, logvar_list):
        pair_losses = []
        var_list = [torch.exp(logvar) for logvar in logvar_list]
        for i, j in itertools.combinations(range(len(mu_list)), 2):
            pair_losses.append((mu_list[i] - mu_list[j]).pow(2).mean())
            pair_losses.append((var_list[i] - var_list[j]).pow(2).mean())
        if not pair_losses:
            return torch.tensor(0.0, device=mu_list[0].device)
        return torch.stack(pair_losses).mean()

    def _pairwise_mmd(self, z_list):
        pair_losses = []
        for i, j in itertools.combinations(range(len(z_list)), 2):
            pair_losses.append(self._mmd_distance(z_list[i], z_list[j]))
        if not pair_losses:
            return torch.tensor(0.0, device=z_list[0].device)
        return torch.stack(pair_losses).mean()

    def _mmd_distance(self, x, y):
        if x.size(0) < 2 or y.size(0) < 2:
            return torch.tensor(0.0, device=x.device)
        gamma = 1.0 / (2.0 * (self.mmd_bandwidth ** 2))
        xx = torch.exp(-gamma * self._pairwise_sq_dists(x, x))
        yy = torch.exp(-gamma * self._pairwise_sq_dists(y, y))
        xy = torch.exp(-gamma * self._pairwise_sq_dists(x, y))
        m = x.size(0)
        n = y.size(0)
        xx_term = (xx.sum() - xx.diag().sum()) / (m * (m - 1))
        yy_term = (yy.sum() - yy.diag().sum()) / (n * (n - 1))
        xy_term = xy.mean()
        return xx_term + yy_term - 2 * xy_term

    @staticmethod
    def _pairwise_sq_dists(x, y):
        x_norm = (x ** 2).sum(dim=1).unsqueeze(1)
        y_norm = (y ** 2).sum(dim=1).unsqueeze(0)
        return x_norm + y_norm - 2.0 * torch.mm(x, y.t())

class MultiFeatureGATFusion(nn.Module):
    """Fuse modality-specific and shared features using a graph attention network."""

    def __init__(
            self,
            specific_dim,
            shared_dim=None,
            fusion_dim=None,
            num_layers=1,
            num_heads=4,
            dropout=0.1,
            graph_mode="CMIG",
            readout="mean",
            available_graph_modes=None,
            random_k_neighbors=None,
            random_k_seed=42,
    ):
        super().__init__()
        self.specific_dim = specific_dim
        self.shared_dim = shared_dim if shared_dim is not None else specific_dim
        self.fusion_dim = fusion_dim if fusion_dim is not None else specific_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.readout = readout
        self.available_graph_modes = ["CMIG"]
        self.graph_mode = "CMIG"
        self.random_k_neighbors = random_k_neighbors
        self.random_k_seed = random_k_seed

        self.specific_proj = nn.Linear(self.specific_dim, self.fusion_dim) if self.specific_dim != self.fusion_dim else None
        self.shared_proj = nn.Linear(self.shared_dim, self.fusion_dim) if self.shared_dim != self.fusion_dim else None
        if self.specific_proj is not None:
            self.specific_proj.apply(xavier_init)
        if self.shared_proj is not None:
            self.shared_proj.apply(xavier_init)

        self.gat_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.gat_layers.append(
                GATConv(self.fusion_dim, self.fusion_dim, heads=self.num_heads, concat=False, dropout=dropout)
            )

    def forward(self, specific_features, shared_features, graph_mode_override=None):
        mode = graph_mode_override if graph_mode_override in (self.available_graph_modes or []) else self.graph_mode
        processed_specific = []
        for feat in specific_features:
            if self.specific_proj is not None:
                processed_specific.append(self.specific_proj(feat))
            else:
                processed_specific.append(feat)
        processed_shared = []
        for feat in shared_features:
            if self.shared_proj is not None:
                processed_shared.append(self.shared_proj(feat))
            else:
                processed_shared.append(feat)

        aggregate_modes = {"shared_mean_fc", "shared_mean_star"}
        if mode in aggregate_modes:
            if not processed_shared:
                raise ValueError(f"{mode} requires shared features, but none were provided.")
            stacked_shared = torch.stack(processed_shared, dim=0)
            mean_shared = stacked_shared.mean(dim=0)
            processed_shared = [mean_shared]

        all_nodes = processed_specific + processed_shared
        if not all_nodes:
            raise ValueError("Fusion requires at least one feature tensor")

        node_tensor = torch.stack(all_nodes, dim=1)  # [batch, nodes, dim]
        batch_size, num_nodes, feat_dim = node_tensor.size()
        node_features = node_tensor.view(batch_size * num_nodes, feat_dim)

        spec_count = len(processed_specific)
        shared_count = len(processed_shared)
        base_edge_index = self._base_edge_index(spec_count, shared_count, mode, node_features.device)
        edge_index = self._repeat_edge_index(base_edge_index, batch_size, num_nodes, node_features.device)

        x = node_features
        for layer in self.gat_layers:
            x = layer(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = x.view(batch_size, num_nodes, feat_dim)
        fused = self._readout(x, spec_count, shared_count)
        return fused, x

    def _base_edge_index(self, num_specific, num_shared, mode, device):
        total = num_specific + num_shared
        if total == 1:
            return torch.zeros((2, 1), dtype=torch.long, device=device)
        edges = []
        shared_offset = num_specific

        def connect_bidirectional(u, v):
            edges.append((u, v))
            edges.append((v, u))

        # Always use CMIG mode
        if num_specific > 1:
            for i in range(num_specific):
                connect_bidirectional(i, (i + 1) % num_specific)
        if num_shared > 0:
            pairs = min(num_specific, num_shared)
            for idx in range(pairs):
                connect_bidirectional(idx, shared_offset + idx)
            if pairs == 0 and num_specific > 0:
                connect_bidirectional(0, shared_offset)
            # Attach remaining nodes if counts mismatch
            for extra in range(pairs, num_specific):
                target_shared = shared_offset + (pairs - 1 if pairs > 0 else 0)
                connect_bidirectional(extra, target_shared)
            if num_specific > 0:
                for extra in range(pairs, num_shared):
                    target_spec = (pairs - 1) if pairs > 0 else 0
                    connect_bidirectional(shared_offset + extra, target_spec)

        if not edges:
            for i in range(total):
                for j in range(total):
                    if i == j:
                        continue
                    edges.append((i, j))
        edge_index = torch.tensor(edges, dtype=torch.long, device=device).t()
        return edge_index

    @staticmethod
    def _repeat_edge_index(edge_index, batch_size, num_nodes, device):
        if batch_size == 1:
            return edge_index
        offset = torch.arange(batch_size, device=device, dtype=torch.long).repeat_interleave(edge_index.size(1)) * num_nodes
        return edge_index.repeat(1, batch_size) + offset.unsqueeze(0)

    def _readout(self, node_tensor, num_specific, num_shared):
        if self.readout == "sum":
            return node_tensor.sum(dim=1)
        if self.readout == "specific" and num_specific > 0:
            return node_tensor[:, :num_specific, :].mean(dim=1)
        if self.readout == "shared" and num_shared > 0:
            return node_tensor[:, num_specific:, :].mean(dim=1)
        if self.readout == "cls":
            return node_tensor[:, 0, :]
        if self.readout == "flatten":
            # Preserve all node embeddings (e.g., 6 x 200) so downstream MLP can learn weights.
            batch, nodes, dim = node_tensor.shape
            return node_tensor.reshape(batch, nodes * dim)
        return node_tensor.mean(dim=1)

    def output_dim(self, num_specific, num_shared):
        """Return fused feature dimension based on the current readout."""
        aggregate_modes = {"shared_mean_fc", "shared_mean_star"}
        # Some graph modes collapse shared nodes into a single mean node.
        effective_shared = 1 if (self.graph_mode in aggregate_modes and num_shared > 0) else num_shared
        node_count = num_specific + effective_shared
        if self.readout == "flatten":
            return node_count * self.fusion_dim
        return self.fusion_dim


def init_model_dict(
        num_view,
        num_class,
        dim_list,
        dim_he_list,
        dim_hc,
        model_dopout=0.5,
        enable_confidence=True,
        fusion_backend="attention",
        shared_module_config=None,
        fusion_config=None,
        shared_loss_weights=None,
        use_specific_branch=True,
        use_specific_features=True,
    ):
    model_dict = {}
    for i in range(num_view):
        model_dict["E{:}".format(i + 1)] = GAT(dim_list[i], dim_he_list, model_dopout)
        model_dict["C{:}".format(i+1)] = Classifier_1(dim_he_list[-1], num_class)
    fusion_hidden_dims = [dim_he_list[0]] + [dim for dim in dim_he_list[1:]]
    shared_cfg = None
    if shared_module_config:
        shared_cfg = shared_module_config.copy()
        shared_cfg.setdefault("input_dims", dim_list)
        shared_cfg.setdefault("latent_dim", dim_he_list[0])
        shared_cfg.setdefault("target_dim", max(dim_list))
    fusion_cfg = fusion_config.copy() if fusion_config else None
    model_dict["Fus"] = TCP(
        dim_list,
        fusion_hidden_dims,
        num_class,
        model_dopout,
        enable_confidence=enable_confidence,
        fusion_backend=fusion_backend,
        shared_config=shared_cfg,
        fusion_config=fusion_cfg,
        shared_loss_weights=shared_loss_weights,
        use_specific_branch=use_specific_branch,
        use_specific_features=use_specific_features,
    )
    return model_dict


def init_optim(num_view, model_dict, lr_e=1e-4):
    optim_dict = {}
    for i in range(num_view):
        optim_dict["C{:}".format(i+1)] = torch.optim.Adam(
                list(model_dict["E{:}".format(i+1)].parameters())+list(model_dict["C{:}".format(i+1)].parameters()),
                lr=lr_e)
    optim_dict["M"] = torch.optim.Adam(list(model_dict["Fus"].parameters()),lr=lr_e)

    return optim_dict
