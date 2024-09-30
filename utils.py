import gc
import os
import random
import scipy.sparse as sp
import torch
import torch.nn as nn
import numpy as np
import dgl
import dgl.function as fn
import torch.nn.functional as F
from sklearn.metrics import f1_score
import torch_sparse
from torch_sparse import SparseTensor


def set_random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cosine_similarity(x):
    x_normalized = F.normalize(x, dim=1)
    cosine_sim = torch.matmul(x_normalized, x_normalized.t())
    cosine_sim = SparseTensor.from_dense(cosine_sim)

    return cosine_sim


def jaccard_similarity(dl):
    row_list = []
    col_list = []
    for i, (k, v) in enumerate(dl.links['data'].items()):
        v = v.tocoo()
        row_list.append(torch.LongTensor(v.row))
        col_list.append(torch.LongTensor(v.col))
    num_nodes = dl.nodes['total']
    row, col, sparse_sizes = torch.cat(tuple(row_list)), torch.cat(tuple(col_list)), (num_nodes, num_nodes)
    adj = SparseTensor(row=row, col=col, value=torch.ones(len(row)), sparse_sizes=sparse_sizes)
    degrees = adj.sum(dim=1)
    print(f'num_nodes: {num_nodes}')

    intersection = adj @ adj.t()
    rows, cols, values = intersection.coo()
    union = degrees[rows] + degrees[cols] - values
    values = values / union
    jaccard_sim = SparseTensor(row=rows, col=cols, value=values, sparse_sizes=(num_nodes, num_nodes))

    return jaccard_sim


def sparse_tensor_row_norm(tensor):
    deg = torch_sparse.sum(tensor, dim=1)
    deg_inv_sqrt = deg.pow_(-1.0)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
    tensor = torch_sparse.mul(tensor, deg_inv_sqrt.view(-1, 1))

    return tensor


def prune(sparse_tensor, threshold):
    if threshold == 0 or sparse_tensor.nnz() == 0:
        return sparse_tensor
    row, col, values = sparse_tensor.coo()
    threshold_value = torch.kthvalue(values, int(values.numel() * threshold)).values
    mask = values > threshold_value
    pruned_sparse_tensor = SparseTensor(row=row[mask], col=col[mask], value=values[mask])

    return pruned_sparse_tensor


def get_jaccard_sim(dl):
    jaccard_sim = jaccard_similarity(dl)

    return jaccard_sim


def get_feature_sim(feature):
    feature_sim = cosine_similarity(feature)

    return feature_sim


def get_full_paths(edge_types, tgt_type, num_hops):

    def dfs(current, path, hops):
        if hops == 0:
            return [path]

        paths = []
        for edge in edge_types:
            if edge.startswith(current):
                next_node = edge[1]
                new_path = path + edge[1]
                paths.extend(dfs(next_node, new_path, hops - 1))

        return paths

    all_paths = []
    for n in range(1, num_hops + 1):
        hop_paths = dfs(tgt_type, tgt_type, n)
        all_paths.extend([path for path in hop_paths if len(path) == n + 1])

    return all_paths


def get_path_adj(adjs, hop_adjs, path, calc):
    if len(path) > 2:
        if path[:-1] in hop_adjs.keys():
            return calc.matmul(hop_adjs[path[:-1]], adjs[path[-2:]])
        else:
            return calc.matmul(get_path_adj(adjs, hop_adjs, path[:-1], calc), adjs[path[-2:]])
    else:
        return adjs[path]


def calculate_overlap_ratio(sparse1, sparse2, calc, method='sum'):
    overlap = calc.mul(sparse1, sparse2)
    overlap_count = overlap.nnz()
    count1 = sparse1.nnz()
    count2 = sparse2.nnz()
    if method == 'sum':
        count = count1 + count2 - overlap_count
    elif method == 'max':
        count = max(count1, count2)
    elif method == 'min':
        count = min(count1, count2)
    else:
        raise ValueError
    ratio = overlap_count / count

    return ratio


def merge_sparse(sparse1, sparse2, calc, reduce='sum'):
    if reduce == 'sum':
        merged = calc.add(sparse1, sparse2)
    else:
        row1, col1, value1 = sparse1.coo()
        row2, col2, value2 = sparse2.coo()
        row = torch.cat([row1, row2])
        col = torch.cat([col1, col2])
        value = torch.cat([value1, value2])
        merged = SparseTensor(row=row, col=col, value=value, sparse_sizes=sparse1.sizes())
        merged = merged.coalesce(reduce=reduce)

    return merged


def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn*s
        pp += nn

    return pp


def train(model, feats, labels_cuda, loss_fcn, optimizer, train_loader, scalar=None):
    model.train()
    device = labels_cuda.device
    total_loss = 0
    iter_num = 0

    for batch in train_loader:
        if isinstance(feats, list):
            batch_feats = [x[batch].to(device) for x in feats]
        elif isinstance(feats, dict):
            batch_feats = {k: x[batch].to(device) for k, x in feats.items()}
        else:
            assert 0
        batch_y = labels_cuda[batch]

        optimizer.zero_grad()
        if scalar is not None:
            with torch.cuda.amp.autocast():
                output_att = model(batch_feats)
                train_loss = loss_fcn(output_att, batch_y)
            scalar.scale(train_loss).backward()
            scalar.step(optimizer)
            scalar.update()
        else:
            output_att = model(batch_feats)
            train_loss = loss_fcn(output_att, batch_y)
            train_loss.backward()
            optimizer.step()
        total_loss += train_loss.item()
        iter_num += 1
    loss = total_loss / iter_num
    return loss


def Evaluation(output, labels):
    preds = output.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()
    num_correct = 0
    binary_pred = np.zeros(preds.shape).astype('int')
    for i in range(preds.shape[0]):
        k = labels[i].sum().astype('int')
        topk_idx = preds[i].argsort()[-k:]
        binary_pred[i][topk_idx] = 1
        for pos in list(labels[i].nonzero()[0]):
            if labels[i][pos] and labels[i][pos] == binary_pred[i][pos]:
                num_correct += 1

    return binary_pred


def evaluate(model_pred, labels, dataset):
    labels = labels.cpu()
    if dataset == "IMDB":
        # evaluation from HALO
        pred_result = Evaluation(model_pred, labels)
        # pred_result = (model_pred > 0).int().cpu()
    else:
        pred_result = model_pred.argmax(dim=1).cpu()
    macro = f1_score(labels, pred_result, average='macro')
    micro = f1_score(labels, pred_result, average='micro')

    return macro, micro


class SparseCalcUtil:
    def __init__(self, device):
        self.device = device

    def matmul(self, src, other):
        return torch_sparse.matmul(src.to(self.device), other.to(self.device)).cpu()

    def mul(self, src, other):
        return torch_sparse.mul(src.to(self.device), other.to(self.device)).cpu()

    def add(self, src, other):
        return torch_sparse.add(src.to(self.device), other.to(self.device)).cpu()


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience, delta=0, save_path='checkpoint.pt'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.counter = 0
        self.best_epoch = 0
        self.best_score = None
        self.best_pred = None
        self.early_stop = False
        self.delta = delta
        self.save_path = save_path

    def __call__(self, epoch, val_loss, val_acc, raw_pred, model, judge_loss=True, save_model=False):
        if judge_loss:
            score = -val_loss
        else:
            score = sum(val_acc)
        if self.best_score is None:
            self.best_epoch = epoch
            self.best_score = score
            self.best_pred = raw_pred
            if save_model:
                self.save_checkpoint(model)
        elif score < self.best_score - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_epoch = epoch
            self.best_score = score
            self.best_pred = raw_pred
            if save_model:
                self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        """Saves model when validation loss decrease."""
        torch.save(model.state_dict(), self.save_path)
