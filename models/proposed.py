import torch
import torch.nn as nn
import torch.nn.functional as F


class ImprovedDESimplE(nn.Module):
    def __init__(self,
                 num_ent: int,
                 num_rel: int,
                 num_date: int,
                 s_emb_dim: int,
                 t_emb_dim: int,
                 cycle: int,
                 dropout: float):
        super(ImprovedDESimplE, self).__init__()

        self.num_ent = num_ent
        self.num_rel = num_rel
        self.num_date = num_date

        self.s_emb_dim = s_emb_dim
        self.t_emb_dim = t_emb_dim

        self.cycle = cycle
        self.dropout = dropout
        self.time_nl = torch.sin

        # Base Embedding
        self.ent_embs_h = nn.Embedding(num_ent, s_emb_dim).cuda()
        self.ent_embs_t = nn.Embedding(num_ent, s_emb_dim).cuda()
        self.rel_embs_f = nn.Embedding(num_rel, s_emb_dim + t_emb_dim).cuda()
        self.rel_embs_i = nn.Embedding(num_rel, s_emb_dim + t_emb_dim).cuda()

        nn.init.xavier_uniform_(self.ent_embs_h.weight)
        nn.init.xavier_uniform_(self.ent_embs_t.weight)
        nn.init.xavier_uniform_(self.rel_embs_f.weight)
        nn.init.xavier_uniform_(self.rel_embs_i.weight)

        # Share Time Embedding
        self.stw = nn.Embedding(num_date//cycle + 1, t_emb_dim)
        nn.init.xavier_uniform_(self.stw.weight)

        # Relation-Timestamp Composition
        self.rtc = nn.Embedding(num_date, s_emb_dim + t_emb_dim)
        nn.init.xavier_uniform_(self.rtc.weight)

        # Time Embedding (frequency)
        self.m_freq_h = nn.Embedding(num_ent, t_emb_dim).cuda()
        self.m_freq_t = nn.Embedding(num_ent, t_emb_dim).cuda()
        self.d_freq_h = nn.Embedding(num_ent, t_emb_dim).cuda()
        self.d_freq_t = nn.Embedding(num_ent, t_emb_dim).cuda()
        self.y_freq_h = nn.Embedding(num_ent, t_emb_dim).cuda()
        self.y_freq_t = nn.Embedding(num_ent, t_emb_dim).cuda()

        nn.init.xavier_uniform_(self.m_freq_h.weight)
        nn.init.xavier_uniform_(self.d_freq_h.weight)
        nn.init.xavier_uniform_(self.y_freq_h.weight)
        nn.init.xavier_uniform_(self.m_freq_t.weight)
        nn.init.xavier_uniform_(self.d_freq_t.weight)
        nn.init.xavier_uniform_(self.y_freq_t.weight)

        # Time Embedding (phi)
        self.m_phi_h = nn.Embedding(num_ent, t_emb_dim).cuda()
        self.m_phi_t = nn.Embedding(num_ent, t_emb_dim).cuda()
        self.d_phi_h = nn.Embedding(num_ent, t_emb_dim).cuda()
        self.d_phi_t = nn.Embedding(num_ent, t_emb_dim).cuda()
        self.y_phi_h = nn.Embedding(num_ent, t_emb_dim).cuda()
        self.y_phi_t = nn.Embedding(num_ent, t_emb_dim).cuda()

        nn.init.xavier_uniform_(self.m_phi_h.weight)
        nn.init.xavier_uniform_(self.d_phi_h.weight)
        nn.init.xavier_uniform_(self.y_phi_h.weight)
        nn.init.xavier_uniform_(self.m_phi_t.weight)
        nn.init.xavier_uniform_(self.d_phi_t.weight)
        nn.init.xavier_uniform_(self.y_phi_t.weight)

        # Time Embedding (amplitude)
        self.m_amps_h = nn.Embedding(num_ent, t_emb_dim).cuda()
        self.m_amps_t = nn.Embedding(num_ent, t_emb_dim).cuda()
        self.d_amps_h = nn.Embedding(num_ent, t_emb_dim).cuda()
        self.d_amps_t = nn.Embedding(num_ent, t_emb_dim).cuda()
        self.y_amps_h = nn.Embedding(num_ent, t_emb_dim).cuda()
        self.y_amps_t = nn.Embedding(num_ent, t_emb_dim).cuda()

        nn.init.xavier_uniform_(self.m_amps_h.weight)
        nn.init.xavier_uniform_(self.d_amps_h.weight)
        nn.init.xavier_uniform_(self.y_amps_h.weight)
        nn.init.xavier_uniform_(self.m_amps_t.weight)
        nn.init.xavier_uniform_(self.d_amps_t.weight)
        nn.init.xavier_uniform_(self.y_amps_t.weight)

    def forward(self, batch):
        heads, rels, tails, years, months, days, date_ids = batch
        h_embs1, r_embs1, t_embs1, h_embs2, r_embs2, t_embs2 = self.__get_embeddings(heads, rels, tails, years, months, days, date_ids)

        scores = ((h_embs1 * r_embs1) * t_embs1 + (h_embs2 * r_embs2) * t_embs2) / 2.0
        scores = F.dropout(scores, p=self.dropout, training=self.training)
        scores = torch.sum(scores, dim=1)

        return scores

    def __get_embeddings(self, heads, rels, tails, years, months, days, date_ids):
        years  = years.view(-1, 1)
        months = months.view(-1, 1)
        days   = days.view(-1, 1)

        h_embs1 = self.ent_embs_h(heads)
        r_embs1 = self.rel_embs_f(rels)
        t_embs1 = self.ent_embs_t(tails)
        h_embs2 = self.ent_embs_h(tails)
        r_embs2 = self.rel_embs_i(rels)
        t_embs2 = self.ent_embs_t(heads)

        time = self.rtc(date_ids)
        r_embs1 = r_embs1 + r_embs1 * time
        r_embs2 = r_embs2 + r_embs2 * time

        h_embs1 = torch.cat((h_embs1, self.__get_ent_time_emb(heads, years, months, days, date_ids, "head")), dim=1)
        t_embs1 = torch.cat((t_embs1, self.__get_ent_time_emb(tails, years, months, days, date_ids, "tail")), dim=1)
        h_embs2 = torch.cat((h_embs2, self.__get_ent_time_emb(tails, years, months, days, date_ids, "head")), dim=1)
        t_embs2 = torch.cat((t_embs2, self.__get_ent_time_emb(heads, years, months, days, date_ids, "tail")), dim=1)

        return h_embs1, r_embs1, t_embs1, h_embs2, r_embs2, t_embs2

    def __get_ent_time_emb(self, entities, years, months, days, date_ids, h_or_t):
        shared_window = self.stw(torch.div(date_ids, self.cycle, rounding_mode='floor'))

        if h_or_t == "head":
            emb  = self.y_amps_h(entities) * self.time_nl(self.y_freq_h(entities) * years  + self.y_phi_h(entities))
            emb += self.m_amps_h(entities) * self.time_nl(self.m_freq_h(entities) * months + self.m_phi_h(entities))
            emb += self.d_amps_h(entities) * self.time_nl(self.d_freq_h(entities) * days   + self.d_phi_h(entities))
        else:
            emb  = self.y_amps_t(entities) * self.time_nl(self.y_freq_t(entities) * years  + self.y_phi_t(entities))
            emb += self.m_amps_t(entities) * self.time_nl(self.m_freq_t(entities) * months + self.m_phi_t(entities))
            emb += self.d_amps_t(entities) * self.time_nl(self.d_freq_t(entities) * days   + self.d_phi_t(entities))
        emb += shared_window

        return emb
