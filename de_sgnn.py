import numpy as np
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict as ddict


class DE_SGraph(nn.Module):
    def __init__(self, dataset, params):
        super(DE_SGraph, self).__init__()
        self.dataset = dataset
        self.params  = params

        self.train_link = self.get_link()
        self.gold_rels  = self.get_gold_relations()

        self.ent_embs = nn.Embedding(dataset.num_ent(), params.s_emb_dim).cuda()
        self.rel_embs = nn.Embedding(dataset.num_rel(), params.s_emb_dim+params.t_emb_dim).cuda()
        nn.init.xavier_uniform_(self.ent_embs.weight)
        nn.init.xavier_uniform_(self.rel_embs.weight)

        self.m_freq = nn.Embedding(self.dataset.num_ent(), self.params.t_emb_dim).cuda()
        self.d_freq = nn.Embedding(self.dataset.num_ent(), self.params.t_emb_dim).cuda()
        self.y_freq = nn.Embedding(self.dataset.num_ent(), self.params.t_emb_dim).cuda()
        nn.init.xavier_uniform_(self.m_freq.weight)
        nn.init.xavier_uniform_(self.d_freq.weight)
        nn.init.xavier_uniform_(self.y_freq.weight)

        self.m_phi = nn.Embedding(self.dataset.num_ent(), self.params.t_emb_dim).cuda()
        self.d_phi = nn.Embedding(self.dataset.num_ent(), self.params.t_emb_dim).cuda()
        self.y_phi = nn.Embedding(self.dataset.num_ent(), self.params.t_emb_dim).cuda()
        nn.init.xavier_uniform_(self.m_phi.weight)
        nn.init.xavier_uniform_(self.d_phi.weight)
        nn.init.xavier_uniform_(self.y_phi.weight)

        self.m_amp = nn.Embedding(self.dataset.num_ent(), self.params.t_emb_dim).cuda()
        self.d_amp = nn.Embedding(self.dataset.num_ent(), self.params.t_emb_dim).cuda()
        self.y_amp = nn.Embedding(self.dataset.num_ent(), self.params.t_emb_dim).cuda()
        nn.init.xavier_uniform_(self.m_amp.weight)
        nn.init.xavier_uniform_(self.d_amp.weight)
        nn.init.xavier_uniform_(self.y_amp.weight)

        self.time_nl = torch.sin
        self.encode = Encoder(params, dataset.num_rel()).cuda()
        # self.decode = ...

    def forward(self, heads, rels, tails, years, months, days):
        set_of_entity = torch.cat([heads, tails]).unique()
        neighbor, neighbor_dict, assign = self.get_context(set_of_entity)

        eemb = self.ent_embs(torch.tensor(neighbor, dtype=torch.long).cuda())
        eemb = self.encode(eemb, neighbor, neighbor_dict, assign, set_of_entity, self.gold_rels)
        eemb = torch.split(eemb, 1, dim=0)
        edict = dict()
        for i, emb in zip(set_of_entity.tolist(), eemb):
            edict[i] = emb

        size = heads.shape[0]
        h, r, t = list(), list(), list()
        for i in range(size):
            r.append(rels[i].item())
            h.append(edict[heads[i].item()])
            t.append(edict[tails[i].item()])
        h = torch.cat(h, dim=0)
        t = torch.cat(t, dim=0)
        r = self.rel_embs(torch.tensor(r, dtype=torch.long).cuda())

        years  = years.view(-1, 1)
        months = months.view(-1, 1)
        days   = days.view(-1, 1)
        h_t = self.get_time_embedd(heads, years, months, days)
        t_t = self.get_time_embedd(tails, years, months, days)

        h = torch.cat([h, h_t], dim=1)
        t = torch.cat([t, t_t], dim=1)

        scores = h + r - t
        scores = F.dropout(scores, p=self.params.dropout, training=self.training)
        scores = -torch.norm(scores, dim=1)
        return scores

    def get_time_embedd(self, entities, year, month, day):
        y = self.y_amp(entities) * self.time_nl(self.y_freq(entities) * year + self.y_phi(entities))
        m = self.m_amp(entities) * self.time_nl(self.m_freq(entities) * month + self.m_phi(entities))
        d = self.d_amp(entities) * self.time_nl(self.d_freq(entities) * day + self.d_phi(entities))
        return y + m + d

    def get_context(self, entity):
        assign = ddict(list)
        neighbor_dict = ddict(int)
        for i, e in enumerate(entity):
            if len(self.train_link[e.item()]) <= self.params.num_neighbor_samples:
                near = self.train_link[e.item()]
            else:
                near = random.sample(self.train_link[e.item()], self.params.num_neighbor_samples)
            for k in near:
                if k not in neighbor_dict:
                    neighbor_dict[k] = len(neighbor_dict)
                assign[neighbor_dict[k]].append(i)

        neighbor = list()
        for k, v in sorted(neighbor_dict.items(), key=lambda l: l[1]):
            neighbor.append(k)
        return neighbor, neighbor_dict, assign

    def get_link(self):
        train_link = ddict(set)
        for triple in self.dataset.data['train']:
            train_link[triple[0]].add(triple[2])
            train_link[triple[2]].add(triple[0])
        train_link = ddict(list, ((k, list(v)) for k, v in train_link.items()))
        return train_link

    def get_gold_relations(self):
        gold_rels = dict()
        for triple in self.dataset.data['train']:
            gold_rels[(triple[0], triple[2])] = triple[1]
        return gold_rels


class Encoder(nn.Module):
    def __init__(self, params, rel_num):
        super(Encoder, self).__init__()
        self.params = params

        self.transform = TransitionLayer(params.s_emb_dim, rel_num).cuda()
        self.pooling   = PoolingLayer(params.pooling_method).cuda()

    def forward(self, x, neighbor_entities, neighbor_dict, assign, entities, relations):
        if len(neighbor_dict) == 1:
            x = [x]
        else:
            x = torch.split(x, 1, dim=0)

        assignR = dict()
        bundle = ddict(list)    # relation space
        for v, k in enumerate(neighbor_entities):
            for i in assign[v]:
                e = entities[i].item()
                if (e, k) in relations:
                    r = relations[(e, k)] * 2
                else:
                    r = relations[(k, e)] * 2 + 1
                assignR[(r, len(bundle[r]))] = v
                bundle[r].append(x[v])

        result = [0 for _ in range(len(neighbor_dict))]
        result = self.transform(bundle, assignR, result)
        result = self.pooling(result, assign)
        result = torch.cat(result, dim=0)

        return result


class TransitionBlock(nn.Module):
    def __init__(self, dim):
        super(TransitionBlock, self).__init__()

        self.linear = nn.Linear(dim, dim).cuda()
        self.norm   = nn.BatchNorm1d(dim).cuda()
        self.act    = nn.ReLU().cuda()

    def forward(self, x):
        z = self.linear(x)
        z = self.norm(z) if z.shape[0] > 1 else z
        z = self.act(z)
        return z


class TransitionLayer(nn.Module):
    def __init__(self, dim, rel_num):
        super(TransitionLayer, self).__init__()
        self.rel_num = rel_num

        self.transformH = nn.ModuleList(
            [TransitionBlock(dim).cuda() for _ in range(rel_num)]
        )
        self.transformT = nn.ModuleList(
            [TransitionBlock(dim).cuda() for _ in range(rel_num)]
        )

    def forward(self, rel_spaces, assignR, result):
        for space, rx in rel_spaces.items():
            if len(rx) == 1:
                rx = rx[0]
                if space % 2 == 0:
                    rx = self.transformH[int(space // 2)](rx)
                else:
                    rx = self.transformT[int(space // 2)](rx)
                result[assignR[(space, 0)]] = rx
            else:
                rx = torch.cat(rx, dim=0)
                if space % 2 == 0:
                    rx = self.transformH[int(space//2)](rx)
                else:
                    rx = self.transformT[int(space//2)](rx)
                rx = torch.split(rx, 1, dim=0)
                for i, x in enumerate(rx):
                    result[assignR[(space, i)]] = x
        return result


class PoolingLayer(nn.Module):
    def __init__(self, pooling_method):
        super(PoolingLayer, self).__init__()
        self.method = pooling_method

    def forward(self, xs, neighbor):
        if self.method == "avg":
            sources = ddict(list)
            for ee in neighbor:
                for i in neighbor[ee]:
                    sources[i].append(xs[ee])
            result = []
            for i, xxs in sorted(sources.items(), key=lambda l: l[0]):
                result.append(sum(xxs)/len(xxs))
        else:
            raise NotImplementedError
        return result


# class Decoder(nn.Module):
#     def __init__(self, method):
#         super(Decoder, self).__init__()
#         self.method = method
#
#     def forward(self, head, rels, tail):
#         if self.method == 'transe':
#             scores = (head + rels - tail).pow(2).sum(dim=1)
#         else:
#             raise NotImplementedError
#         return scores
