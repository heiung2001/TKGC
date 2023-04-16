import torch


def shred_facts(facts):
    heads = torch.tensor(facts[:, 0]).long()
    rels = torch.tensor(facts[:, 1]).long()
    tails = torch.tensor(facts[:, 2]).long()
    years = torch.tensor(facts[:, 3]).float()
    months = torch.tensor(facts[:, 4]).float()
    days = torch.tensor(facts[:, 5]).float()
    return heads, rels, tails, years, months, days
