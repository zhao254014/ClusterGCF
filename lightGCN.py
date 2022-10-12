import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F


from utility.helper import *
from utility.batch_test import *

import warnings
warnings.filterwarnings('ignore')
from time import time


class lightGCN(nn.Module):
    def __init__(self,data_config, pretrain_data, args):
        super(lightGCN, self).__init__()

        # argument settings
        self.model_type = 'lightGCN'
        self.adj_type = args.adj_type
        self.alg_type = args.alg_type
        self.device = args.device
        self.group = args.groups

        self.pretrain_data = pretrain_data

        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']

        self.n_fold = 20

        self.norm_adj = data_config['norm_adj']
        self.n_nonzero_elems = self.norm_adj.count_nonzero()

        self.lr = args.lr

        self.emb_dim = args.embed_size
        self.batch_size = args.batch_size

        self.dropout = args.mlp_dropout
        self.node_dropout = eval(args.node_dropout)


        self.weight_size = eval(args.layer_size)
        self.n_layers = len(self.weight_size)

        self.model_type += '_%s_%s_l%d' % (self.adj_type, self.alg_type, self.n_layers)

        self.regs = eval(args.regs)
        self.decay = self.regs[0]

        self.verbose = args.verbose
        """
        *********************************************************
        Init the weight of user-item. 初始化权重和embedding
        """
        self.embedding_dict, self.weight_dict = self.init_weight()

        self.embedding_user = self.embedding_dict['user_emb']
        #print( self.embedding_user)
        self.embedding_item = self.embedding_dict['item_emb']
        #print(self.embedding_item)
        """
        *********************************************************
        Get sparse adj. 得到邻接矩阵
        """
        self.sparse_norm_adj = self._convert_sp_mat_to_sp_tensor(self.norm_adj).to(self.device)

    def init_weight(self):
        # xavier init
        initializer = nn.init.xavier_uniform_
        if self.pretrain_data is None:
            embedding_dict = nn.ParameterDict({
                'user_emb': nn.Parameter(initializer(torch.empty(self.n_users,
                                                 self.emb_dim))),
                'item_emb': nn.Parameter(initializer(torch.empty(self.n_items,
                                                 self.emb_dim)))
            })
            print('using xavier initialization')
        else:
            pretrain_datas = torch.load(args.dataset+'.npz')
            embedding_dict = nn.ParameterDict({
                'user_emb': nn.Parameter(pretrain_datas['embedding_user']),
                'item_emb': nn.Parameter(pretrain_datas['embedding_item'])
            })
            print('using pretrained initialization')
        weight_dict = nn.ParameterDict({
            'W_gc_1': nn.Parameter(initializer(torch.empty(self.emb_dim,
                                                           self.emb_dim))),
            'b_gc_1': nn.Parameter(initializer(torch.empty(1,self.emb_dim))),
            'W_gc': nn.Parameter(initializer(torch.empty(self.emb_dim,
                                                           self.group))),
            'b_gc': nn.Parameter(initializer(torch.empty(1,self.group)))
        })

        return embedding_dict, weight_dict

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)

    def _split_A_hat(self, X):     #将邻接矩阵分为20份
        A_fold_hat = []
        fold_len = (self.n_users + self.n_items) // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold -1:
                end = self.n_users + self.n_items
            else:
                end = (i_fold + 1) * fold_len
            A_fold_hat.append(self._convert_sp_mat_to_sp_tensor(X[start:end]))
        return A_fold_hat

    def _split_A_hat_node_dropout(self, X):
        A_fold_hat = []
        fold_len = (self.n_users + self.n_items) // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold - 1:
                end = self.n_users + self.n_items
            else:
                end = (i_fold + 1) * fold_len
            # A_fold_hat.append(self._convert_sp_mat_to_sp_tensor(X[start:end]))
            temp = self._convert_sp_mat_to_sp_tensor(X[start:end])
            n_nonzero_temp = X[start:end].count_nonzero()
            A_fold_hat.append(self._dropout_sparse(temp, 1 - self.node_dropout[0], n_nonzero_temp))
        return A_fold_hat
    """
    *********************************************************
    使用结点失活
    """
    def sparse_dropout(self, x, rate, noise_shape):         #邻接矩阵失活0.1
        random_tensor = 1 - rate
        random_tensor += torch.rand(noise_shape).to(x.device)
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        i = x._indices()
        v = x._values()

        i = i[:, dropout_mask]
        v = v[dropout_mask]

        out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)
        return out * (1. / (1 - rate))

    """
    *********************************************************
    brp损失函数
    """
    def create_bpr_loss(self,  u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings,users,pos_items,neg_items):#都是1024*256
        pos_scores = torch.sum(torch.mul(u_g_embeddings, pos_i_g_embeddings), axis=1)  #用户与正样本相乘 再求和   [1024]
        neg_scores = torch.sum(torch.mul(u_g_embeddings, neg_i_g_embeddings), axis=1)  #用户与负样本相乘 再求和

        mf_loss = torch.mean(torch.nn.functional.softplus(-(pos_scores - neg_scores)))
        #maxi = nn.LogSigmoid()(pos_scores - neg_scores)

        #mf_loss = -1 * torch.mean(maxi)

        users_emb_ego = self.embedding_user[users, :]
        pos_emb_ego = self.embedding_item[pos_items, :]
        neg_emb_ego = self.embedding_item[neg_items, :]

        regularizer = (torch.norm(users_emb_ego) ** 2
                       + torch.norm(pos_emb_ego) ** 2
                       + torch.norm(neg_emb_ego) ** 2) / 2
        emb_loss = self.decay * regularizer / self.batch_size
        """
        # cul regularizer
        regularizer = (torch.norm(users) ** 2
                       + torch.norm(pos_items) ** 2
                       + torch.norm(neg_items) ** 2) / 2
        emb_loss = self.decay * regularizer / self.batch_size
        """
        return mf_loss + emb_loss, mf_loss, emb_loss

    def rating(self, u_g_embeddings, pos_i_g_embeddings):
        return torch.matmul(u_g_embeddings, pos_i_g_embeddings.t())
    """
    *********************************************************
    处理更新权重的邻接矩阵
    """
    def sparse_dense_mul_col(self,s, d):
        i = s._indices()
        v = s._values()
        dv = d[i[0, :]]
        return torch.sparse.FloatTensor(i, v * dv, s.size())

    def sparse_dense_mul_row(self,s, d):
        i = s._indices()
        v = s._values()
        dv = d[i[1, :]]
        return torch.sparse.FloatTensor(i, v * dv, s.size())

    def forward(self, users, pos_items, neg_items, drop_flag=True):

        A_hat = self.sparse_dropout(self.sparse_norm_adj,
                                    self.node_dropout,
                                    self.sparse_norm_adj._nnz()) if drop_flag else self.sparse_norm_adj  #结点失活

        ego_embeddings = torch.cat([self.embedding_dict['user_emb'],
                                    self.embedding_dict['item_emb']], 0)


        all_embeddings = [ego_embeddings]

        for k in range(0, self.n_layers):

            side_embeddings =torch.sparse.mm(A_hat,ego_embeddings)
            ego_embeddings = side_embeddings

            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, 1)
        all_embeddings =torch.mean(all_embeddings, axis=1, keepdims=False)

        #分用户和物品em
        u_g_embeddings = all_embeddings[:self.n_users, :]
        i_g_embeddings = all_embeddings[self.n_users:, :]

        """
        *********************************************************
        look up.
        """
        # 查询此批次1024数据的embed
        u_g_embeddings = u_g_embeddings[users, :]
        pos_i_g_embeddings = i_g_embeddings[pos_items, :]
        neg_i_g_embeddings = i_g_embeddings[neg_items, :]

        return u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings


def load_pretrained_data():
    #pretrain_path = '%spretrain/%s/%s.npz' % (args.proj_path, args.dataset, 'embedding')
    pretrain_path = args.dataset+'.npz'
    try:
        pretrain_data = np.load(pretrain_path)
        print(pretrain_data)
        print('load the pretrained embeddings.')
    except Exception:
        pretrain_data = None
    return pretrain_data

if __name__ == '__main__':
    args.device = torch.device('cuda:' + str(args.gpu_id))



    config = dict()
    config['n_users'] = data_generator.n_users
    config['n_items'] = data_generator.n_items

    """
    *********************************************************
    生成拉普拉斯矩阵
    """
    plain_adj, norm_adj, mean_adj, pre_adj = data_generator.get_adj_mat()
    if args.adj_type == 'plain':
        config['norm_adj'] = plain_adj
        print('use the plain adjacency matrix')

    elif args.adj_type == 'norm':
        config['norm_adj'] = norm_adj
        print('use the normalized adjacency matrix')

    elif args.adj_type == 'gcmc':
        config['norm_adj'] = mean_adj
        print('use the gcmc adjacency matrix')

    elif args.adj_type == 'pre':
        config['norm_adj'] = pre_adj
        print('use the pre adjcency matrix')

    else:
        config['norm_adj'] = mean_adj + sp.eye(mean_adj.shape[0])
        print('use the mean adjacency matrix')

    t0 = time()
    if args.pretrain == -1:
        pretrain_data = load_pretrained_data()
    else:
        pretrain_data = None
    model = lightGCN(data_config=config, pretrain_data=pretrain_data, args=args).to(args.device)
    # 存储初始的embedding
    #torch.save(model.state_dict(), args.dataset + '.npz')
    """
    *********************************************************
    Train.
    """
    cur_best_pre_0, stopping_step = 0, 0
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_loger, pre_loger, rec_loger, ndcg_loger, hit_loger = [], [], [], [], []
    for epoch in range(args.epoch):
        t1 = time()
        loss, mf_loss, emb_loss = 0., 0., 0.
        n_batch = data_generator.n_train // args.batch_size + 1
        for idx in range(n_batch):
            users, pos_items, neg_items = data_generator.sample()
            u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings = model(users,
                                                                           pos_items,
                                                                           neg_items,
                                                                           drop_flag=args.node_dropout_flag)

            batch_loss, batch_mf_loss, batch_emb_loss = model.create_bpr_loss(u_g_embeddings,
                                                                              pos_i_g_embeddings,
                                                                              neg_i_g_embeddings,
                                                                              users, pos_items, neg_items )
            optimizer.zero_grad()
            #torch.cuda.empty_cache()
            batch_loss.backward()
            optimizer.step()

            loss += batch_loss/n_batch
            mf_loss += batch_mf_loss/n_batch
            emb_loss += batch_emb_loss/n_batch

        if (epoch + 1) % 10 != 0:
            if args.verbose > 0 and epoch % args.verbose == 0:
                perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f]' % (
                    epoch, time() - t1, loss, mf_loss, emb_loss)
                print(perf_str)
            continue

        t2 = time()
        users_to_test = list(data_generator.test_set.keys())   # 测试集中字典的索引（用户）4001个
        ret = test(model, users_to_test, drop_flag=False)

        t3 = time()

        loss_loger.append(loss)
        rec_loger.append(ret['recall'])
        pre_loger.append(ret['precision'])
        ndcg_loger.append(ret['ndcg'])
        hit_loger.append(ret['hit_ratio'])

        if args.verbose > 0:
            perf_str = 'Epoch %d [%.1fs + %.1fs]: train==[%.5f=%.5f + %.5f], recall=[%.5f, %.5f], ' \
                       'precision=[%.5f, %.5f], hit=[%.5f, %.5f], ndcg=[%.5f, %.5f]' % \
                       (epoch, t2 - t1, t3 - t2, loss, mf_loss, emb_loss, ret['recall'][0], ret['recall'][-1],
                        ret['precision'][0], ret['precision'][-1], ret['hit_ratio'][0], ret['hit_ratio'][-1],
                        ret['ndcg'][0], ret['ndcg'][-1])
            print(perf_str)

        cur_best_pre_0, stopping_step, should_stop = early_stopping(ret['recall'][0], cur_best_pre_0,
                                                                    stopping_step, expected_order='acc', flag_step=5)

        # *********************************************************
        #early stopping when cur_best_pre_0 is decreasing for ten successive steps.
        if should_stop == True:
            break

        # *********************************************************
        # save the user & item embeddings for pretraining.
        if ret['recall'][0] == cur_best_pre_0 and args.save_flag == 1:
            torch.save(model.state_dict(), args.weights_path + str(epoch) + '.pkl')
            print('save the weights in path: ', args.weights_path + str(epoch) + '.pkl')

    recs = np.array(rec_loger)
    pres = np.array(pre_loger)
    ndcgs = np.array(ndcg_loger)
    hit = np.array(hit_loger)

    best_rec_0 = max(recs[:, 0])
    idx = list(recs[:, 0]).index(best_rec_0)

    final_perf = "Best Iter=[%d]@[%.1f]\trecall=[%s], precision=[%s], hit=[%s], ndcg=[%s]" % \
                 (idx, time() - t0, '\t'.join(['%.5f' % r for r in recs[idx]]),
                  '\t'.join(['%.5f' % r for r in pres[idx]]),
                  '\t'.join(['%.5f' % r for r in hit[idx]]),
                  '\t'.join(['%.5f' % r for r in ndcgs[idx]]))
    print(final_perf)