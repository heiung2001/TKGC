import os
import numpy as np

from torch.utils.data import Dataset
from datetime import datetime


class TemporalKGDataset(Dataset):
    ent2id  = dict()
    rel2id  = dict()
    date2id = dict()

    def __init__(self, kg_root, mode, neg_ratio):
        super(TemporalKGDataset, self).__init__()
        self.kg_root = kg_root
        self.mode = mode
        self.file_name = f"{mode}.txt"
        self.neg_ratio = neg_ratio

        self.facts = self._read_kg()

    def __len__(self):
        return len(self.facts)

    def __getitem__(self, idx):
        return self.facts[idx]

    # def __getitem__(self, idx):
    #     pos = self.facts[idx]
    #     num_ent = len(self.ent2id)
    #
    #     pos_neg_group_size = 1 + self.neg_ratio
    #     head_replaced = np.repeat(np.expand_dims(np.copy(pos), 0), pos_neg_group_size, axis=0)
    #     tail_replaced = np.copy(head_replaced)
    #
    #     head_candidates = np.random.randint(low=1, high=num_ent, size=head_replaced.shape[0])
    #     tail_candidates = np.random.randint(low=1, high=num_ent, size=tail_replaced.shape[0])
    #     head_candidates[0] = 0
    #     tail_candidates[0] = 0
    #
    #     head_replaced[:, 0] = (head_replaced[:, 0] + head_candidates) % num_ent
    #     tail_replaced[:, 2] = (tail_replaced[:, 2] + tail_candidates) % num_ent
    #     sample = np.concatenate([head_replaced, tail_replaced], axis=0)
    #
    #     return sample

    def _read_kg(self):
        data_path = os.path.join(self.kg_root, self.file_name)
        with open(data_path, 'r', encoding='utf-8') as f:
            kg = f.readlines()

        facts = list()
        for quadruple in kg:
            s, p, o, t = quadruple.strip().split('\t')

            subject_id   = self.__get_ent_id(s)
            predicate_id = self.__get_rel_id(p)
            object_id    = self.__get_ent_id(o)

            facts.append([subject_id, predicate_id, object_id, t])
        facts = self.__get_date_id(facts)
        return facts

    @classmethod
    def __get_date_id(cls, facts):
        time_space = set()
        for i, fact in enumerate(facts):
            date = fact[-1]
            time_space.add(date)

            facts[i] = facts[i][:-1]
            date = list(map(float, date.split('-')))
            facts[i] += date
        time_space = sorted(time_space, key=lambda x: datetime.strptime(x, '%Y-%m-%d'))
        time_space = [tuple(map(float, d.split('-'))) for d in time_space]
        cls.date2id = {t: i for i, t in enumerate(time_space)}

        for i, fact in enumerate(facts):
            date = fact[-3:]
            date_id = cls.date2id[tuple(date)]
            facts[i].append(date_id)
        return facts

    @classmethod
    def __get_ent_id(cls, ent_name):
        if ent_name in cls.ent2id:
            return cls.ent2id[ent_name]
        cls.ent2id[ent_name] = len(cls.ent2id)
        return cls.ent2id[ent_name]

    @classmethod
    def __get_rel_id(cls, rel_name):
        if rel_name in cls.rel2id:
            return cls.rel2id[rel_name]
        cls.rel2id[rel_name] = len(cls.rel2id)
        return cls.rel2id[rel_name]

    @classmethod
    def get_num_ent(cls):
        return len(cls.ent2id)

    @classmethod
    def get_num_rel(cls):
        return len(cls.rel2id)

    @classmethod
    def get_num_date(cls):
        return len(cls.date2id)
