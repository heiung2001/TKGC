import torch
import numpy as np
from scripts import shred_facts
from measure import Measure


class Tester:
    def __init__(self, dataset, model_path, valid_or_test):
        self.model = torch.load(model_path)
        self.model.eval()
        self.dataset = dataset
        self.valid_or_test = valid_or_test
        self.measure = Measure()

    def get_rank(self,
                 sim_scores):  # assuming the test fact is the first one
        return (sim_scores > sim_scores[0]).sum() + 1

    def replace_and_shred(self,
                          fact,
                          raw_or_fil,
                          head_or_tail):
        head, rel, tail, years, months, days = fact
        if head_or_tail == "head":
            ret_facts = [(i, rel, tail, years, months, days) for i in range(self.dataset.num_ent())]
        elif head_or_tail == "tail":
            ret_facts = [(head, rel, i, years, months, days) for i in range(self.dataset.num_ent())]
        else:
            raise ValueError

        if raw_or_fil == "raw":
            ret_facts = [tuple(fact)] + ret_facts
        elif raw_or_fil == "fil":
            ret_facts = [tuple(fact)] + list(set(ret_facts) - self.dataset.all_facts_as_tuples)

        return shred_facts(np.array(ret_facts))

    def test(self):
        with torch.no_grad():
            for i, fact in enumerate(self.dataset.data[self.valid_or_test][:100]):
                settings = ["fil"]
                for raw_or_fil in settings:
                    for head_or_tail in ["head", "tail"]:
                        heads, rels, tails, years, months, days = self.replace_and_shred(fact, raw_or_fil, head_or_tail)
                        sim_scores = self.model(heads, rels, tails, years, months, days).cpu().data.numpy()
                        rank = self.get_rank(sim_scores)
                        self.measure.update(rank, raw_or_fil)

        self.measure.print_()
        print("~~~~~~~~~~~~~")
        self.measure.normalize(len(self.dataset.data[self.valid_or_test]))
        self.measure.print_()

        return self.measure.mrr["fil"]
