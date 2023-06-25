import torch
import numpy as np


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
