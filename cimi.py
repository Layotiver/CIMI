import os

os.environ["ALL_PROXY"] = "http://localhost:10809"

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
device = torch.device("cuda", args.device)
setup_seed()

text, label = get_data_map[args.dataset]()
split = int(len(text) * 0.8)
train_titles, train_labels = text[:split], label[:split]
test_titles, test_labels = text[split:], label[split:]

train_dataset = My_dataset(train_titles, train_labels)
test_dataset = My_dataset(test_titles, test_labels)

model = BertClassifier(args).to(device)
model.requires_grad_(False)
cimi = CIMI().to(device)
optimizer = torch.optim.AdamW(cimi.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
loss_mse = torch.nn.MSELoss(reduction="mean")

weights = torch.load(f"save/{args.dataset}_bert.pt", map_location=device)
model.load_state_dict(weights)

model_path = f"save/{args.dataset}_stack"
if not os.path.exists(model_path):
    os.makedirs(model_path)

nb_epochs = get_epoch_map[args.dataset]


def get_Ls(llm_input, llm_output, saliency):
    """causal sufficiency loss"""
    saliency = saliency[..., 1:]
    init_state = llm_output["hidden_states"][0]
    last_state = llm_output["hidden_states"][-1]

    pooler_output = llm_output["pooler_output"]
    prob = model.forward_linear(pooler_output)

    attention_mask = llm_input["attention_mask"].unsqueeze(-1)
    init_state_pos = init_state * ((saliency - 1.0) * attention_mask + 1.0)
    llm_output_pos = model.forward_embeds_bert(init_state_pos, llm_input)
    pooler_output_pos = model.forward_pooler(llm_output_pos["last_hidden_state"])
    prob_pos = model.forward_linear(pooler_output_pos)

    init_state_neg = init_state * ((1.0 - saliency - 1.0) * attention_mask + 1.0)
    llm_output_neg = model.forward_embeds_bert(init_state_neg, llm_input)
    pooler_output_neg = model.forward_pooler(llm_output_neg["last_hidden_state"])
    prob_neg = model.forward_linear(pooler_output_neg)

    return loss_mse(prob_pos, prob) - loss_mse(prob_neg, prob)  # 这里换成交叉熵损失会不会比较好


def get_Li(llm_input, llm_output, saliency):
    """ "causal intervention loss"""
    get_noise_map = {"religion": 0.2, "rotten": 0.2, "yelp": 0.2, "clickbait": 0.1, "sentiment": 0.1, "food": 0.2, "imdb": 0.2, "hate": 0.1}
    init_state = llm_output["hidden_states"][0]
    last_state = llm_output["hidden_states"][-1]

    attention_mask = llm_input["attention_mask"].unsqueeze(-1)
    vmask = (saliency[..., 1:] - 1.0) * attention_mask + 1.0

    ii = np.random.randint(1, init_state.shape[0])
    init_state_shuffle = torch.cat([init_state[ii:], init_state[:ii]])
    lam = torch.tensor(np.random.random([init_state.shape[0], 1, 1])).to(args.device).float() * get_noise_map[args.dataset]
    init_state_noise = vmask * init_state + (1 - vmask) * ((1 - lam) * init_state + lam * init_state_shuffle)
    llm_output_noise = model.forward_embeds_bert(init_state_noise, llm_input)
    last_state_noise = llm_output_noise["last_hidden_state"]
    io_cat_noise = torch.cat([init_state_noise, last_state_noise], dim=-1)
    saliency_noise = cimi(io_cat_noise)

    return generate_sim(saliency, saliency_noise)


def get_Lp(llm_input, llm_output, saliency):
    """weakly supervising loss"""
    init_state = llm_output["hidden_states"][0]
    last_state = llm_output["hidden_states"][-1]
    saliency = saliency[..., 1:]

    ii = np.random.randint(1, init_state.shape[0])
    init_state_shuffle = torch.cat([init_state[ii:], init_state[:ii]])
    io_cat_shuffle = torch.cat([init_state_shuffle, last_state], dim=-1)
    saliency_shuffle = cimi(io_cat_shuffle)[..., 1:]

    return torch.mean(-F.logsigmoid(saliency - saliency_shuffle))


def get_all_loss(batch_x):
    llm_input = model.tokenize(batch_x)
    legnths = torch.sum(llm_input["attention_mask"], dim=1)
    llm_output = model.forward_ids_bert(**llm_input)

    init_state = llm_output["hidden_states"][0]
    last_state = llm_output["hidden_states"][-1]
    io_cat = torch.cat([init_state, last_state], dim=-1)
    saliency = cimi(io_cat)

    loss_s = get_Ls(llm_input, llm_output, saliency)
    loss_i = get_Li(llm_input, llm_output, saliency)
    loss_p = get_Lp(llm_input, llm_output, saliency)

    loss = loss_s + loss_p + loss_i * get_dis_map[args.dataset]
    return loss


for epoch in range(nb_epochs):
    optimizer.zero_grad()
    loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, num_workers=2, shuffle=True)
    pbar = tqdm.tqdm(loader, "train", total=len(test_dataset) // args.batch_size)  # 这里的变量名pbar是一个缩写，表示进度条（progress bar）
    acc_list = []
    iou_list = []
    comp_list = []
    suff_list = []
    length_list = []
    mse_list = []
    select_list = []
    trans_list = []
    for batch_x, batch_y in pbar:
        loss = get_all_loss(batch_x)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

torch.save(model.state_dict(), f"{model_path}/{args.dataset}_cimi.pt")
