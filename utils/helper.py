import os
import torch
import numpy as np
from .params import Params


class AugmentCollator:
    def __init__(self, num_ent, neg_rattio):
        self.num_ent = num_ent
        self.neg_ratio = neg_rattio

    def __call__(self, batch):
        pos_neg_group_size = 1 + self.neg_ratio
        head_replaced = np.repeat(np.copy(batch), pos_neg_group_size, axis=0)
        tail_replaced = np.copy(head_replaced)

        head_candidates = np.random.randint(low=1, high=self.num_ent, size=head_replaced.shape[0])
        tail_candidates = np.random.randint(low=1, high=self.num_ent, size=tail_replaced.shape[0])

        for i in range(head_replaced.shape[0] // pos_neg_group_size):
            head_candidates[i * pos_neg_group_size] = 0
            tail_candidates[i * pos_neg_group_size] = 0

        head_replaced[:, 0] = (head_replaced[:, 0] + head_candidates) % self.num_ent
        tail_replaced[:, 2] = (tail_replaced[:, 2] + tail_candidates) % self.num_ent

        return np.concatenate((head_replaced, tail_replaced), axis=0)

def shred_facts(batch):
    np_batch = np.concatenate(batch, axis=0)
    heads = torch.tensor(np_batch[:, 0]).long()
    rels = torch.tensor(np_batch[:, 1]).long()
    tails = torch.tensor(np_batch[:, 2]).long()
    years = torch.tensor(np_batch[:, 3]).float()
    months = torch.tensor(np_batch[:, 4]).float()
    days = torch.tensor(np_batch[:, 5]).float()
    date_id = torch.tensor(np_batch[:, 6]).long()
    return heads, rels, tails, years, months, days, date_id


def save_model(model, name, chkpnt):
    print("Saving the model")
    directory = "/kaggle/working/heiung/" + name + '/'
    if not os.path.exists(directory):
        os.makedirs(directory)

    torch.save(model, directory + '_' + str(chkpnt) + '.chkpnt')
