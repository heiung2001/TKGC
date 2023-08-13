class Params:
    def __init__(self,
                 num_epoch=500,
                 batch_size=512,
                 lr=0.001,
                 reg_lambda=0.0,
                 emb_dim=100,
                 neg_ratio=20,
                 dropout=0.4,
                 save_each=50,
                 se_prop=0.9,
                 model_name='TimeFactorDE') -> None:
        self.model_name = model_name
        self.num_epoch = num_epoch
        self.batch_size = batch_size
        self.lr = lr
        self.reg_lambda = reg_lambda
        self.s_emb_dim = int(se_prop * emb_dim)
        self.t_emb_dim = emb_dim - int(se_prop * emb_dim)
        self.save_each = save_each
        self.neg_ratio = neg_ratio
        self.dropout = dropout
        self.se_prop = se_prop

        self.pooling_method = 'avg'
        self.num_neighbor_samples = 64

        self.num_ent = None
        self.num_rel = None
        self.num_date = None

        self.cycle = 122

    def str_(self) -> str:
        return str(self.num_epoch) + "_" + str(self.batch_size) + "_" + str(self.lr) + "_" + str(self.reg_lambda) + "_" + str(
            self.s_emb_dim) + "_" + str(self.neg_ratio) + "_" + str(self.dropout) + "_" + str(
            self.t_emb_dim) + "_" + str(self.save_each) + "_" + str(self.se_prop)
