import os
from utils.helper import shred_facts
from torch.utils.data import DataLoader, Dataset
from datetime import datetime


class TemporalKGDataset(Dataset):
    ent2id  = dict()
    rel2id  = dict()
    date2id = dict()

    def __init__(self, kg_root, file_name):
        super(TemporalKGDataset, self).__init__()
        self.kg_root = kg_root
        self.file_name = file_name

        self.facts = self._read_kg()

    def __len__(self):
        return len(self.facts)

    def __getitem__(self, idx):
        return self.facts[idx]

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


if __name__ == "__main__":
    train_dataset = TemporalKGDataset(r'data\icews14', 'train.txt')
    val_dataset   = TemporalKGDataset(r'data\icews14', 'valid.txt')
    test_dataset  = TemporalKGDataset(r'data\icews14', 'test.txt')

    batch_size = 5
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, collate_fn=shred_facts)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=shred_facts)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=shred_facts)

    heads, rels, tails, years, months, days, date_ids = next(iter(train_loader))
