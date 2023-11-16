import numpy as np
import pandas as pd


def setup_seed(seed=0):
    import torch
    import os
    import numpy as np
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
