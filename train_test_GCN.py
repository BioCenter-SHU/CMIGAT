"""Training and testing pipeline using GCN encoders + GAT fusion."""
import os
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import TCP, Classifier_1, GraphConvolution, init_optim as init_optim_base
from utils import (
    one_hot_tensor,
    cal_sample_weight,
    gen_adj_mat_tensor,
    gen_test_adj_mat_tensor,
    cal_adj_mat_parameter,
    Normal_value_of_sample,
    Samp_pro_tensor,
)  # noqa: F401

# CUDA setup
if torch.cuda.is_available():
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    cuda = True
else:
    device = torch.device("cpu")
    cuda = False


# ------------------ Metric helpers ------------------ #
def _init_best_metrics(num_class):
    if num_class == 2:
        keys = ["ACC", "F1", "AUC"]
    else:
        keys = ["ACC", "F1_weighted", "F1_macro"]
    return {k: {"value": float("-inf"), "epoch": None} for k in keys}


def _compute_metrics(labels, probs, num_class):
    preds = probs.argmax(1)
    metrics = {"ACC": accuracy_score(labels, preds)}
    if num_class == 2:
        metrics["F1"] = f1_score(labels, preds)
        metrics["AUC"] = roc_auc_score(labels, probs[:, 1])
    else:
        metrics["F1_weighted"] = f1_score(labels, preds, average='weighted')
        metrics["F1_macro"] = f1_score(labels, preds, average='macro')
    return metrics


def _print_metrics(metrics, num_class):
    print("Test ACC: {:.3f}".format(metrics["ACC"]))
    if num_class == 2:
        print("Test F1: {:.3f}".format(metrics["F1"]))
        print("Test AUC: {:.3f}".format(metrics["AUC"]))
    else:
        print("Test F1 weighted: {:.3f}".format(metrics["F1_weighted"]))
        print("Test F1 macro: {:.3f}".format(metrics["F1_macro"]))


def _update_best(best_metrics, metrics, epoch_label):
    for key, value in metrics.items():
        if value > best_metrics[key]["value"]:
            best_metrics[key]["value"] = value
            best_metrics[key]["epoch"] = epoch_label


def _is_better_overall(metrics, current_best, num_class):
    """ACC 优先，其次 F1/F1_weighted，再次 AUC/F1_macro，比较小数点后三位。"""
    def r3(v): return round(v, 3)
    order = ["ACC", "F1", "AUC"] if num_class == 2 else ["ACC", "F1_weighted", "F1_macro"]
    if current_best is None:
        return True
    for key in order:
        new_v = r3(metrics[key]); old_v = r3(current_best["metrics"][key])
        if new_v > old_v:
            return True
        if new_v < old_v:
            return False
    return False


def _update_best_overall(best_overall, metrics, epoch_label, num_class):
    if _is_better_overall(metrics, best_overall, num_class):
        return {"metrics": metrics, "epoch": epoch_label}
    return best_overall


def _print_best_summary(best_metrics):
    print("\nBest results:")
    for key, item in best_metrics.items():
        if item["value"] == float("-inf"):
            continue
        epoch_info = item["epoch"]
        if epoch_info is None:
            suffix = ""
        elif epoch_info == "test":
            suffix = " (test run)"
        else:
            suffix = f" (epoch {epoch_info})"
        print(f"{key}: {item['value']:.3f}{suffix}")


def _print_best_overall(best_overall, num_class):
    if not best_overall:
        return
    metrics = best_overall["metrics"]
    epoch_info = best_overall["epoch"]
    suffix = " (test run)" if epoch_info == "test" else f" (epoch {epoch_info})"
    if num_class == 2:
        print(f"\nBest overall by ACC→F1→AUC: ACC {metrics['ACC']:.3f}, F1 {metrics['F1']:.3f}, AUC {metrics['AUC']:.3f}{suffix}")
    else:
        print(f"\nBest overall by ACC→F1_weighted→F1_macro: "
              f"ACC {metrics['ACC']:.3f}, F1_weighted {metrics['F1_weighted']:.3f}, "
              f"F1_macro {metrics['F1_macro']:.3f}{suffix}")


# ------------------ Data helpers ------------------ #
def prepare_trte_data(data_folder, view_list):
    num_view = len(view_list)
    labels_tr = np.loadtxt(os.path.join(data_folder, "labels_tr.csv"), delimiter=',').astype(int)
    labels_te = np.loadtxt(os.path.join(data_folder, "labels_te.csv"), delimiter=',').astype(int)
    data_tr_list, data_te_list = [], []
    for i in view_list:
        data_tr_list.append(np.loadtxt(os.path.join(data_folder, f"{i}_tr.csv"), delimiter=','))
        data_te_list.append(np.loadtxt(os.path.join(data_folder, f"{i}_te.csv"), delimiter=','))
    num_tr = data_tr_list[0].shape[0]; num_te = data_te_list[0].shape[0]
    data_mat_list = [np.concatenate((data_tr_list[i], data_te_list[i]), axis=0) for i in range(num_view)]
    data_tensor_list = [torch.FloatTensor(mat).cuda() if cuda else torch.FloatTensor(mat) for mat in data_mat_list]
    idx_dict = {"tr": list(range(num_tr)), "te": list(range(num_tr, num_tr + num_te))}
    data_train_list = [tensor[idx_dict["tr"]].clone() for tensor in data_tensor_list]
    data_all_list = [torch.cat((tensor[idx_dict["tr"]].clone(), tensor[idx_dict["te"]].clone()), 0)
                     for tensor in data_tensor_list]
    labels = np.concatenate((labels_tr, labels_te))
    return data_train_list, data_all_list, idx_dict, labels


def gen_trte_adj_mat(data_tr_list, data_trte_list, trte_idx, adj_parameter):
    adj_metric = "cosine"
    adj_train_list, adj_test_list = [], []
    for i in range(len(data_tr_list)):
        adj_parameter_adaptive = cal_adj_mat_parameter(adj_parameter, data_tr_list[i], adj_metric)
        adj_train_list.append(gen_adj_mat_tensor(data_tr_list[i], adj_parameter_adaptive, adj_metric))
        adj_test_list.append(gen_test_adj_mat_tensor(data_trte_list[i], trte_idx, adj_parameter_adaptive, adj_metric))
    return adj_train_list, adj_test_list


def _edge_index_to_sparse(edge_index, num_nodes, dtype, device):
    """Convert edge_index (2, E) to a torch.sparse_coo_tensor with unit weights."""
    values = torch.ones(edge_index.size(1), device=device, dtype=dtype)
    adj = torch.sparse_coo_tensor(edge_index, values, (num_nodes, num_nodes))
    return adj.coalesce()


# ------------------ Model: GCN encoder with optional reconstruction ------------------ #
class GCNEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dims, dropout=0.3, activation=F.leaky_relu, use_decoder=True):
        super().__init__()
        self.layers = nn.ModuleList()
        dims = [in_dim] + hidden_dims
        for i in range(len(dims) - 1):
            self.layers.append(GraphConvolution(dims[i], dims[i + 1]))
        self.dropout = dropout
        self.activation = activation
        self.use_decoder = use_decoder
        if use_decoder:
            self.decoder = nn.Linear(dims[-1], in_dim)
            nn.init.xavier_normal_(self.decoder.weight)
            if self.decoder.bias is not None:
                self.decoder.bias.data.fill_(0.0)
        else:
            self.decoder = None

    def forward(self, x, adj, return_reconstruction=False):
        out = x
        for layer in self.layers:
            out = layer(out, adj)
            if self.activation:
                out = self.activation(out)
            out = F.dropout(out, self.dropout, training=self.training)
        if return_reconstruction and self.decoder is not None:
            recon = self.decoder(out)
            return out, recon
        return out


# ------------------ Epoch loops ------------------ #
def train_epoch(
        data_list,
        adj_list,
        label,
        one_hot_label,
        sample_weight,
        model_dict,
        optim_dict,
        train_TCP=True,
        recon_weight=1.0,
        train_specific_heads=True,
    ):
    loss_dict = {}
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    for m in model_dict:
        model_dict[m].train()
    num_view = len(data_list)
    use_reconstruction = recon_weight > 0.0
    if train_specific_heads:
        for i in range(num_view):
            optim_dict[f"C{i+1}"].zero_grad()
            recon_loss = None
            if use_reconstruction:
                feature, recon = model_dict[f"E{i+1}"](data_list[i], adj_list[i], return_reconstruction=True)
                recon_loss = F.mse_loss(recon, data_list[i])
            else:
                feature = model_dict[f"E{i+1}"](data_list[i], adj_list[i])
            ci = model_dict[f"C{i+1}"](feature)
            ci_loss = torch.mean(torch.mul(criterion(ci, label), sample_weight))
            total_loss = ci_loss + (recon_weight * recon_loss if recon_loss is not None else 0.0)
            total_loss.backward()
            optim_dict[f"C{i+1}"].step()
            loss_dict[f"C{i+1}"] = ci_loss.detach().cpu().numpy().item()
            if recon_loss is not None:
                loss_dict[f"Recon{i+1}"] = recon_loss.detach().cpu().numpy().item()
                loss_dict[f"TotalC{i+1}"] = total_loss.detach().cpu().numpy().item()
    if train_TCP:
        optim_dict["M"].zero_grad()
        out_feat = []
        for i in range(num_view):
            feature = model_dict[f"E{i+1}"](data_list[i], adj_list[i])
            out_feat.append(feature)
        fusion_out = model_dict["Fus"](out_feat, label, raw_inputs=data_list, return_details=True)
        if isinstance(fusion_out, tuple) and len(fusion_out) == 3:
            c_loss, MMlogit, aux = fusion_out
            if aux and "loss_breakdown" in aux:
                for key, value in aux["loss_breakdown"].items():
                    loss_dict[f"Fus_{key}"] = value
        else:
            c_loss, MMlogit = fusion_out
        c_loss.backward()
        optim_dict["M"].step()
        loss_dict["M"] = c_loss.detach().cpu().numpy().item()
    return loss_dict


def test_epoch(data_list, adj_list, te_idx, model_dict):
    for m in model_dict:
        model_dict[m].eval()
    num_view = len(data_list)
    ci_list = []
    for i in range(num_view):
        code = model_dict[f"E{i + 1}"](data_list[i], adj_list[i])
        ci_list.append(code)
    clogit = model_dict["Fus"].infer(ci_list, raw_inputs=data_list)
    clogit = clogit[te_idx, :]
    prob = F.softmax(clogit, dim=1).data.cpu().numpy()
    return prob


# ------------------ Model init ------------------ #
def init_model_dict_gcn(
        num_view,
        num_class,
        dim_list,
        dim_he_list,
        dim_hc,
        model_dropout=0.3,
        enable_confidence=True,
        fusion_backend="gat",
        shared_module_config=None,
        fusion_config=None,
        shared_loss_weights=None,
        use_specific_branch=True,
        use_specific_features=True,
    ):
    model_dict = {}
    for i in range(num_view):
        model_dict[f"E{i + 1}"] = GCNEncoder(dim_list[i], dim_he_list, dropout=model_dropout)
        model_dict[f"C{i + 1}"] = Classifier_1(dim_he_list[-1], num_class)
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
        model_dropout,
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
    return init_optim_base(num_view, model_dict, lr_e)


# ------------------ Main entry ------------------ #
def train_test(data_folder, view_list, num_class,
               lr_e_pretrain, lr_e, testonly,
               num_epoch_pretrain, num_epoch,
               use_confidence=True, recon_weight=0.5,
               fusion_backend="gat",
               shared_module_cfg={"align_mode": "linear", "dropout": 0.1, "use_sem_loss": True, "use_mmd_loss": True, "mmd_kernel_bandwidth": 1.0},
               fusion_cfg={"graph_mode": "CMIG", "graph_mode_options": ["CMIG"], "num_heads": 1, "num_layers": 1, "readout": "flatten", "random_k_neighbors": 3, "random_k_seed": 97},
               shared_loss_weights={"sem": 0.1, "mmd": 0.4},
               shared_pretrain_epochs=0,
               shared_pretrain_lr=5e-3,
               use_specific_branch=True,
               use_specific_features=True,
               return_overall_snapshot=True):
    test_inverval = 1
    
    num_view = len(view_list)
    dim_hvcdn = pow(num_class, num_view)

    if data_folder == 'BRCA':
        adj_parameter = 2
        dim_he_list = [200]
    data_tr_list, data_trte_list, trte_idx, labels_trte = prepare_trte_data(data_folder, view_list)
    labels_tr_tensor = torch.LongTensor(labels_trte[trte_idx["tr"]])
    onehot_labels_tr_tensor = one_hot_tensor(labels_tr_tensor, num_class)
    sample_weight_tr = torch.FloatTensor(cal_sample_weight(labels_trte[trte_idx["tr"]], num_class))
    if cuda:
        labels_tr_tensor = labels_tr_tensor.cuda()
        onehot_labels_tr_tensor = onehot_labels_tr_tensor.cuda()
        sample_weight_tr = sample_weight_tr.cuda()
    # 原 gen_trte_adj_mat 返回 edge_index；GCN 需要稀疏邻接矩阵，故先生成 edge_index，再转 sparse。
    adj_tr_edge, adj_te_edge = gen_trte_adj_mat(data_tr_list, data_trte_list, trte_idx, adj_parameter)
    adj_tr_list, adj_te_list = [], []
    for edge_index in adj_tr_edge:
        num_nodes = data_tr_list[0].shape[0]
        adj_tr_list.append(_edge_index_to_sparse(edge_index, num_nodes, dtype=data_tr_list[0].dtype, device=data_tr_list[0].device))
    for edge_index in adj_te_edge:
        num_nodes = data_trte_list[0].shape[0]
        adj_te_list.append(_edge_index_to_sparse(edge_index, num_nodes, dtype=data_trte_list[0].dtype, device=data_trte_list[0].device))
    dim_list = [x.shape[1] for x in data_tr_list]
    model_dict = init_model_dict_gcn(
        num_view,
        num_class,
        dim_list,
        dim_he_list,
        dim_hvcdn,
        enable_confidence=use_confidence,
        fusion_backend=fusion_backend,
        shared_module_config=shared_module_cfg,
        fusion_config=fusion_cfg,
        shared_loss_weights=shared_loss_weights,
        use_specific_branch=use_specific_branch,
        use_specific_features=use_specific_features,
    )
    print(f"Confidence learning enabled: {use_confidence}")
    print(f"Fusion backend: {fusion_backend}")
    best_metrics = _init_best_metrics(num_class)
    best_overall = None
    if testonly:
        ckpt_dir = os.path.join("checkpoints", data_folder, "state_dict")
        if os.path.exists(ckpt_dir):
            print(f"Loading checkpoint from {ckpt_dir}")
            for m in model_dict:
                model_file = os.path.join(ckpt_dir, f"{m}.pth")
                if os.path.exists(model_file):
                    model_dict[m].load_state_dict(torch.load(model_file, map_location=device))
                else:
                    print(f"Warning: {model_file} not found")
            print("Checkpoint loaded successfully")
        else:
            print(f"Checkpoint directory not found at {ckpt_dir}")

        for m in model_dict:
            if cuda:
                model_dict[m].cuda()
        te_prob = test_epoch(data_trte_list, adj_te_list, trte_idx["te"], model_dict)
        metrics = _compute_metrics(labels_trte[trte_idx["te"]], te_prob, num_class)
        _print_metrics(metrics, num_class)
        _update_best(best_metrics, metrics, "test")
        best_overall = _update_best_overall(best_overall, metrics, "test", num_class)
    else:
        for m in model_dict:
            if cuda:
                model_dict[m].cuda()
        print("\nPretrain GCNs...")
        optim_dict = init_optim(num_view, model_dict, lr_e_pretrain)
        for epoch in range(num_epoch_pretrain):
            train_epoch(
                data_tr_list, adj_tr_list, labels_tr_tensor,
                onehot_labels_tr_tensor, sample_weight_tr,
                model_dict, optim_dict, train_TCP=False, recon_weight=recon_weight,
                train_specific_heads=use_specific_branch)
        if shared_pretrain_epochs > 0:
            print("\nPretraining shared fusion module...")
            shared_lr = shared_pretrain_lr if shared_pretrain_lr is not None else lr_e_pretrain
            shared_optim = init_optim(num_view, model_dict, shared_lr)
            for epoch in range(shared_pretrain_epochs):
                train_epoch(
                    data_tr_list, adj_tr_list, labels_tr_tensor,
                    onehot_labels_tr_tensor, sample_weight_tr,
                    model_dict, shared_optim, train_TCP=True,
                    recon_weight=recon_weight, train_specific_heads=False,
                )
        print("\nTraining...")
        optim_dict = init_optim(num_view, model_dict, lr_e)
        for epoch in range(num_epoch + 1):
            train_epoch(
                data_tr_list, adj_tr_list, labels_tr_tensor,
                onehot_labels_tr_tensor, sample_weight_tr,
                model_dict, optim_dict, recon_weight=recon_weight,
                train_specific_heads=use_specific_branch)
            if epoch % test_inverval == 0:
                te_prob = test_epoch(data_trte_list, adj_te_list, trte_idx["te"], model_dict)
                print("\nTest: Epoch {:d}".format(epoch))
                metrics = _compute_metrics(labels_trte[trte_idx["te"]], te_prob, num_class)
                _print_metrics(metrics, num_class)
                _update_best(best_metrics, metrics, epoch)
                best_overall = _update_best_overall(best_overall, metrics, epoch, num_class)
                if num_class != 2:
                    print()
    _print_best_summary(best_metrics)
    _print_best_overall(best_overall, num_class)

    if return_overall_snapshot:
        if best_overall:
            overall_snapshot = _init_best_metrics(num_class)
            for key in overall_snapshot:
                overall_snapshot[key]["value"] = best_overall["metrics"][key]
                overall_snapshot[key]["epoch"] = best_overall["epoch"]
        else:
            overall_snapshot = best_metrics
        return overall_snapshot
    return best_metrics


if __name__ == "__main__":
    # simple manual test placeholder
    print("This module is intended to be imported and called via train_test().")
