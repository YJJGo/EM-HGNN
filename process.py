import torch
import dgl
import numpy as np
from data_loader import data_loader
from torch_sparse import SparseTensor
from utils import sparse_tensor_row_norm


def load_data(args):
    dl = data_loader(f'{args.root}/{args.dataset}')

    # use one-hot index vectors for nodes with no attributes
    # === feats ===
    features_list = []
    for i in range(len(dl.nodes['count'])):
        th = dl.nodes['attr'][i]
        if th is None:
            features_list.append(torch.eye(dl.nodes['count'][i]))
        else:
            features_list.append(torch.FloatTensor(th))

    idx_shift = np.zeros(len(dl.nodes['count']) + 1, dtype=np.int32)
    for i in range(len(dl.nodes['count'])):
        idx_shift[i + 1] = idx_shift[i] + dl.nodes['count'][i]

    # === labels ===
    num_classes = dl.labels_train['num_classes']
    val_ratio = 0.2
    train_val_idx = np.nonzero(dl.labels_train['mask'])[0]
    test_idx = np.nonzero(dl.labels_test['mask'])[0]
    labels = np.zeros((dl.nodes['count'][0], num_classes), dtype=int)
    labels[train_val_idx] = dl.labels_train['data'][train_val_idx]
    labels[test_idx] = dl.labels_test['data'][test_idx]
    if args.dataset == 'IMDB':
        labels = torch.FloatTensor(labels)
    else:
        labels = torch.LongTensor(labels).argmax(dim=1)

    np.random.shuffle(train_val_idx)
    split = int(train_val_idx.shape[0] * val_ratio)
    train_idx, val_idx = train_val_idx[split:], train_val_idx[:split]
    train_idx, val_idx, test_idx = np.sort(train_idx), np.sort(val_idx), np.sort(test_idx)
    train_val_test_idx = {'train_idx': train_idx, 'val_idx': val_idx, 'test_idx': test_idx}

    # === adjs ===
    adjs = [] if args.dataset not in ['Freebase'] else {}
    for i, (k, v) in enumerate(dl.links['data'].items()):
        v = v.tocoo()
        src_type_idx = np.where(idx_shift > v.col[0])[0][0] - 1
        dst_type_idx = np.where(idx_shift > v.row[0])[0][0] - 1
        row = v.row - idx_shift[dst_type_idx]
        col = v.col - idx_shift[src_type_idx]
        sparse_sizes = (dl.nodes['count'][dst_type_idx], dl.nodes['count'][src_type_idx])
        adj = SparseTensor(row=torch.LongTensor(row), col=torch.LongTensor(col), sparse_sizes=sparse_sizes)
        if args.dataset not in ['Freebase']:
            adjs.append(adj)
        else:
            adjs[f'{dst_type_idx}{src_type_idx}'] = adj

    idx_shift_dict = {}
    features_dict = {}
    tgt_type = None
    if args.dataset == 'DBLP':
        # A* --- P --- T
        #        |
        #        V
        # author: [4057, 334]
        # paper : [14328, 4231]
        # term  : [7723, 50]
        # venue(conference) : None
        tgt_type = 'A'
        A, P, T, V = features_list
        AP, PA, PT, PV, TP, VP = adjs

        idx_shift_dict = {
            'A': (idx_shift[0], idx_shift[1]),
            'P': (idx_shift[1], idx_shift[2]),
            'T': (idx_shift[2], idx_shift[3]),
            'V': (idx_shift[3], idx_shift[4]),
        }
        features_dict = {'A': A, 'P': P, 'T': T, 'V': V}
        adjs = {'AP': AP, 'PA': PA, 'PT': PT, 'PV': PV, 'TP': TP, 'VP': VP}

    elif args.dataset == 'ACM':
        # A --- P* --- C
        #       |
        #       K
        # paper     : [3025, 1902]
        # author    : [5959, 1902]
        # conference: [56, 1902]
        # field     : None
        tgt_type = 'P'
        P, A, C, K = features_list
        PP, PP_r, PA, AP, PC, CP, PK, KP = adjs

        row0, col0, _ = PP.coo()
        row1, col1, _ = PP_r.coo()
        row, col, sparse_sizes = torch.cat((row0, row1)), torch.cat((col0, col1)), PP.sparse_sizes()
        PP = SparseTensor(row=row, col=col, sparse_sizes=sparse_sizes)
        PP = PP.coalesce()
        PP = PP.set_diag()

        idx_shift_dict = {
            'P': (idx_shift[0], idx_shift[1]),
            'A': (idx_shift[1], idx_shift[2]),
            'C': (idx_shift[2], idx_shift[3]),
            'K': (idx_shift[3], idx_shift[4]),
        }
        if args.ACM_keep_F:
            features_dict = {'P': P, 'A': A, 'C': C, 'K': K}
            adjs = {'PP': PP, 'PA': PA, 'AP': AP, 'PC': PC, 'CP': CP, 'PK': PK, 'KP': KP}
        else:
            features_dict = {'P': P, 'A': A, 'C': C}
            adjs = {'PP': PP, 'PA': PA, 'AP': AP, 'PC': PC, 'CP': CP}

    elif args.dataset == 'IMDB':
        # A --- M* --- D
        #       |
        #       K
        # movie    : [4932, 3489]
        # director : [2393, 3341]
        # actor    : [6124, 3341]
        # keywords : None
        tgt_type = 'M'
        M, D, A, K = features_list
        MD, DM, MA, AM, MK, KM = adjs

        idx_shift_dict = {
            'M': (idx_shift[0], idx_shift[1]),
            'D': (idx_shift[1], idx_shift[2]),
            'A': (idx_shift[2], idx_shift[3]),
            'K': (idx_shift[3], idx_shift[4]),
        }
        features_dict = {'M': M, 'D': D, 'A': A, 'K': K}
        adjs = {'MD': MD, 'DM': DM, 'MA': MA, 'AM': AM, 'MK': MK, 'KM': KM}

    elif args.dataset == 'AMiner':
        # A --- P*
        #       |
        #       R
        # paper     : None
        # author    : None
        # reference : None
        tgt_type = 'P'
        P, A, R = features_list
        PA, AP, PR, RP = adjs

        idx_shift_dict = {
            'P': (idx_shift[0], idx_shift[1]),
            'A': (idx_shift[1], idx_shift[2]),
            'R': (idx_shift[2], idx_shift[3])
        }
        features_dict = {'P': P, 'A': A, 'R': R}
        adjs = {'PA': PA, 'AP': AP, 'PR': PR, 'RP': RP}

    for key, adj in adjs.items():
        adjs[key] = sparse_tensor_row_norm(adj)

    return features_dict, adjs, idx_shift_dict, labels, num_classes, tgt_type, train_val_test_idx, dl
