import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from utils.params import Params
from utils.helper import AugmentCollator


class Trainer:
    def __init__(self,
                 model: nn.Module,
                 optimizer: optim.Optimizer,
                 train_set: Dataset,
                 valid_set: Dataset,
                 params: Params,
                 kg_name: str) -> None:

        self.kg_name = kg_name
        self.model = model
        self.optimizer = optimizer
        self.params = params
        self.criteria = nn.CrossEntropyLoss()

        self.train_set = train_set
        self.valid_set = valid_set

        collate_fn = AugmentCollator(params.num_ent, params.neg_ratio)
        self.train_loader = DataLoader(train_set, self.params.batch_size, shuffle=True, collate_fn=collate_fn)
        self.valid_loader = DataLoader(valid_set, self.params.batch_size, shuffle=False, collate_fn=collate_fn)

    def train_epoch(self,
                    epoch: int) -> None:
        self.model.train()

        running_loss = 0.0
        with tqdm(self.train_loader, desc=f'Epoch {epoch}', unit='batch') as tepoch:
            for batch_idx, batch in enumerate(tepoch):
                self.optimizer.zero_grad()

                num_examples = int(batch[0].shape[0] / (1 + self.params.neg_ratio))
                scores = self.model(batch)
                scores = scores.view(num_examples, self.params.neg_ratio + 1)

                label = torch.zeros(num_examples).long()
                loss = self.criteria(scores, label)

                loss.backward()
                self.optimizer.step()
                running_loss += loss.cpu().item()
                tepoch.set_postfix(loss=running_loss)

        if epoch % self.params.save_each == 0:
            self.save_model(epoch)

    def save_model(self, epoch):
        directory = f"/kaggle/working/heiung/{self.params.model_name}/{self.kg_name}/"
        if not os.path.exists(directory):
            os.makedirs(directory)
        torch.save(self.model, directory + self.params.str_() + '_' + str(epoch) + '.chkpnt')
