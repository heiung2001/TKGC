import torch
import numpy as np
import random
from torch.utils.data import DataLoader
from dataset import TemporalKGDataset
from utils.params import Params
from trainer import Trainer
from tester import Tester
from models.proposed import ImprovedDESimplE
from utils.helper import AugmentCollator


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

params = Params()

kg_name = "ivews14"
kg_root = f"data/{kg_name}/"
train_set = TemporalKGDataset(kg_root, 'train', params.neg_ratio)
valid_set = TemporalKGDataset(kg_root, 'valid', params.neg_ratio)
test_set  = TemporalKGDataset(kg_root, 'test', params.neg_ratio)

params.num_ent = train_set.get_num_ent()
params.num_rel = train_set.get_num_rel()
params.num_date = train_set.get_num_date()

model = ImprovedDESimplE(
    params.num_ent, params.num_rel, params.num_date,
    params.s_emb_dim, params.t_emb_dim,
    params.cycle, params.dropout
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3, weight_decay=params.reg_lambda
)

trainer = Trainer(model, optimizer, train_set, valid_set, params, kg_name)
tester = Tester(model, valid_set, test_set, params)

for epoch in range(1, params.num_epoch + 1):
    trainer.train_epoch(epoch)

    if epoch % params.save_each == 0:
        tester.validate_epoch(epoch)
