import os
import torch
from torch.utils.data import DataLoader
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
device = torch.device("cuda")
setup_seed()

text, label = get_data_map[args.dataset]()
split = int(len(text) * 0.8)
train_titles, train_labels = text[:split], label[:split]
test_titles, test_labels = text[split:], label[split:]

train_dataset = My_dataset(train_titles, train_labels)
test_dataset = My_dataset(test_titles, test_labels)

model = Bert_stack(args).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
loss_mse = torch.nn.MSELoss(reduction="mean")

weights = torch.load(f"save/{args.dataset}_bert.pt", map_location=device)
model.load_state_dict(weights, strict=False)

model_path = f"save/{args.dataset}_stack"
if not os.path.exists(model_path):
    os.makedirs(model_path)

nb_epochs = get_epoch_map[args.dataset]
for epoch in range(nb_epochs):
    model.train()
    optimizer.zero_grad()
    loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, num_workers=2, shuffle=True)
    pbar = tqdm.tqdm(loader, "train", total=len(test_dataset) // args.batch_size)    #这里的变量名pbar是一个缩写，表示进度条（progress bar）
    acc_list = []
    iou_list = []
    comp_list = []
    suff_list = []
    length_list = []
    mse_list = []
    select_list = []
    trans_list = []
    for batch_x,batch_y in pbar:
        ...    