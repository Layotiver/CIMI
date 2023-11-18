import numpy as np
import pandas as pd
import torch

def setup_seed(seed=0):
    import random

    torch.manual_seed(seed)  # 为CPU设置随机种子
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    if torch.cuda.is_available():
        # torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
        torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子
        # os.environ['PYTHONHASHSEED'] = str(seed)

def get_stack_mask(valid_mask):
    return valid_mask.detach().cpu().numpy()

def cal_mono(model, scores, batch_x, batch_y, length, device):
    for i in range(len(scores)):
        scores[i, 0] = -1e9
        scores[i, length[i] - 1:] = -1e9
    labels = np.argmax(batch_y, axis=1)
    dims = len(scores[0])
    idx_sort = np.argsort(-scores, axis=-1)
    pred_import = np.zeros_like(scores)
    for i in range(dims):
        idx = idx_sort[:, :i]
        mask = np.ones_like(scores)
        for j in range(len(idx)):
            mask[j, idx[j]] = 0
        mask = torch.tensor(mask).float().to(device)
        preds = model.forward_mask(batch_x, mask).detach().cpu().numpy()

        for j in range(len(labels)):
            pred_import[j, i] = preds[j, labels[j]]
    cor_list = []
    for i in range(len(scores)):
        try:
            if (pred_import[i, 0] >= 0.5):
                idx = np.min(np.where(pred_import[i] < 0.5)[0])
                cor_list.append(np.minimum(idx, length[i] - 2))
            else:
                idx = np.min(np.where(pred_import[i] >= 0.5)[0])
                cor_list.append(np.minimum(idx, length[i] - 2))
        except:
            cor_list.append(length[i] - 2)
    return cor_list

def generate_sim(embed1, embed2):
    embeddings1 = embed1
    embeddings2 = embed2
    pos_sim = torch.cosine_similarity(embeddings1, embeddings2, dim=0)
    return -pos_sim.mean()

def generate_imdb_data():
    np.random.seed(100)
    data = pd.read_csv("data/IMDB/IMDB.csv")
    data = data.sample(len(data), replace=False)
    text = list(data["text"])
    labels = np.array(list(data["label"]))
    idx = np.where(np.isnan(labels))[0]
    labels[idx] = 0
    labels_onehot = np.zeros((len(labels), 2))
    idx1 = np.where(labels == 0)[0]
    labels_onehot[idx1, 0] = 1
    idx2 = np.where(labels == 1)[0]
    labels_onehot[idx2, 1] = 1
    return text[:5000], labels_onehot[:5000]


def generate_hate_data():
    np.random.seed(100)
    data = pd.read_csv("data/hate/labeled_data.csv")
    data = data.sample(len(data), replace=False)
    text = list(data["tweet"])
    labels = np.array(list(data["class"]))
    idx = np.where(np.isnan(labels))[0]
    labels[idx] = 0
    labels_onehot = np.zeros((len(labels), 3))
    idx1 = np.where(labels == 0)[0]
    labels_onehot[idx1, 0] = 1
    idx2 = np.where(labels == 1)[0]
    labels_onehot[idx2, 1] = 1
    idx3 = np.where(labels == 2)[0]
    labels_onehot[idx3, 2] = 1
    return text[:5000], labels_onehot[:5000]


def generate_yelp_data():
    np.random.seed(100)
    train_data = pd.read_csv("data/yelp/train.csv", encoding="utf-8-sig")
    test_data = pd.read_csv("data/yelp/test.csv", encoding="utf-8-sig")
    data = train_data.append(test_data)
    data = data.sample(len(data), replace=False)
    text = list(data["text"])
    labels = np.array(list(data["label"])) - 1
    idx = np.where(np.isnan(labels))[0]
    labels[idx] = 0
    labels_onehot = np.zeros((len(labels), 2))
    idx1 = np.where(labels == 0)[0]
    labels_onehot[idx1, 0] = 1
    idx2 = np.where(labels == 1)[0]
    labels_onehot[idx2, 1] = 1
    return text[:5000], labels_onehot[:5000]


def generate_clickbait_data():
    np.random.seed(100)
    data = pd.read_csv("data/clickbait/clickbait_data.csv", encoding="utf-8-sig")
    data = data.sample(len(data), replace=False)
    text = list(data["headline"])
    labels = np.array(list(data["clickbait"]))
    idx = np.where(np.isnan(labels))[0]
    labels[idx] = 0
    labels_onehot = np.zeros((len(labels), 2))
    idx1 = np.where(labels == 0)[0]
    labels_onehot[idx1, 0] = 1
    idx2 = np.where(labels == 1)[0]
    labels_onehot[idx2, 1] = 1
    return text[:5000], labels_onehot[:5000]
