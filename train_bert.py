import os
os.environ["ALL_PROXY"] = "http://localhost:10809"

import torch
from torch.utils.data import DataLoader
from transformers import BertModel, BertTokenizer, BertConfig
import tqdm

from utils import *
from parse import *
from dataset import *
from model import *

get_data_map = {
    "yelp": generate_yelp_data,
    "clickbait": generate_clickbait_data,
    "imdb": generate_imdb_data,
    "hate": generate_hate_data,
}

get_epoch_map = {"yelp": 50, "clickbait": 100, "imdb": 50, "hate": 100}
get_dis_map = {"yelp": 1.0, "clickbait": 1.0, "imdb": 0.1, "hate": 1.0}

args = parse_args()
device = torch.device("cuda", args.device)
setup_seed()

text, label = get_data_map[args.dataset]()
split = int(len(text) * 0.8)
train_titles, train_labels = text[:split], label[:split]
test_titles, test_labels = text[split:], label[split:]

train_dataset = My_dataset(train_titles, train_labels)
test_dataset = My_dataset(test_titles, test_labels)

model = BertClassifier(args).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
loss_ce = torch.nn.BCELoss()

best_acc = 0.0
for epoch in range(20):
    model.train()
    optimizer.zero_grad()

    loader = DataLoader(dataset=train_dataset, batch_size=8, num_workers=2, shuffle=True)
    pbar = tqdm.tqdm(loader, "train", total=len(train_dataset) // 8)
    acc_list = []

    for batch_x, batch_y in pbar:
        probs = model(batch_x)  # (B,2)
        batch_y = batch_y.float().to(device)
        loss = loss_ce(probs[:, 1], batch_y[:, 1])
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        batch_y = batch_y.cpu().numpy()
        probs = probs.detach().cpu().numpy()
        acc = np.mean(np.argmax(probs, axis=1) == np.argmax(batch_y, axis=1))
        acc_list.append(acc)
        pbar.set_postfix(mse_loss=loss.cpu().item(), acc=acc)
    print("after epoch %d, train acc=%f" % (epoch, np.mean(acc_list)))
    model.eval()
    loader = DataLoader(dataset=test_dataset, batch_size=2, shuffle=False, num_workers=2)
    pbar = tqdm.tqdm(loader, "test", total=len(test_dataset) // 2)
    probs_list = []
    label_list = []
    for batch_x, batch_y in pbar:
        probs = model(batch_x)  # (B,2)

        probs = probs.detach().cpu().numpy()
        probs_list.append(probs)
        label_list.append(batch_y)

    pred = np.concatenate(probs_list, axis=0)
    label = np.concatenate(label_list, axis=0)
    acc = np.mean(np.argmax(pred, axis=1) == np.argmax(label, axis=1))

    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), f"save/{args.dataset}_bert.pt")
    print("after epoch %d, test acc=%f" % (epoch, acc))
