import torch


def shred_facts(facts):
    heads = torch.tensor(facts[:, 0]).long().cuda()
    rels = torch.tensor(facts[:, 1]).long().cuda()
    tails = torch.tensor(facts[:, 2]).long().cuda()
    years = torch.tensor(facts[:, 3]).float().cuda()
    months = torch.tensor(facts[:, 4]).float().cuda()
    days = torch.tensor(facts[:, 5]).float().cuda()
    return heads, rels, tails, years, months, days
