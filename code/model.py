"""
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Jianbai Ye (gusye@mail.ustc.edu.cn)

Define models here
"""
import world
import torch
from dataloader import BasicDataset
from torch import nn
import numpy as np

from torch.nn import Module
import torch.nn.functional as F

class BasicModel(nn.Module):    
    def __init__(self):
        super(BasicModel, self).__init__()
    
    def getUsersRating(self, users):
        raise NotImplementedError
    
class PairWiseModel(BasicModel):
    def __init__(self):
        super(PairWiseModel, self).__init__()
    def bpr_loss(self, users, pos, neg):
        """
        Parameters:
            users: users list 
            pos: positive items for corresponding users
            neg: negative items for corresponding users
        Return:
            (log-loss, l2-loss)
        """
        raise NotImplementedError
    
class PureMF(BasicModel):
    def __init__(self, 
                 config:dict, 
                 dataset:BasicDataset):
        super(PureMF, self).__init__()
        self.num_users  = dataset.n_users
        self.num_items  = dataset.m_items
        self.latent_dim = config['latent_dim_rec']
        self.f = nn.Sigmoid()
        self.__init_weight()
        
    def __init_weight(self):
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        print("using Normal distribution N(0,1) initialization for PureMF")
        
    def getUsersRating(self, users):
        users = users.long()
        users_emb = self.embedding_user(users)
        items_emb = self.embedding_item.weight
        scores = torch.matmul(users_emb, items_emb.t())
        return self.f(scores)
    
    def bpr_loss(self, users, pos, neg):
        users_emb = self.embedding_user(users.long())
        pos_emb   = self.embedding_item(pos.long())
        neg_emb   = self.embedding_item(neg.long())
        pos_scores= torch.sum(users_emb*pos_emb, dim=1)
        neg_scores= torch.sum(users_emb*neg_emb, dim=1)
        loss = torch.mean(nn.functional.softplus(neg_scores - pos_scores))
        reg_loss = (1/2)*(users_emb.norm(2).pow(2) + 
                          pos_emb.norm(2).pow(2) + 
                          neg_emb.norm(2).pow(2))/float(len(users))
        return loss, reg_loss
        

    def bpr_loss_gcl_Kpos(self,users_emb0,pos_emb0,users,pos_items,alpha):
       # users_emb0,pos_emb0 =  self.computer(nd,fd)   #几次？

        users_emb=   users_emb0[users] 
        loss=self.bpr_loss_gcl_unit(users_emb,pos_emb0[pos_items[:,0]],alpha)
        for i in range(pos_items.size()[1])[1:]:
            pos_emb=   pos_emb0[pos_items[:,i]]  
            loss=loss+self.bpr_loss_gcl_unit(users_emb,pos_emb,alpha)
        
       # losii=self.iiloss(pos_emb0,pos_items)
               
        return  loss.mean() ,0
  
    def bpr_loss_gcl_unit(self,users_emb,pos_emb,alpha): 
        T=self.T 
        sim_batch=torch.exp(torch.mm(users_emb,pos_emb.t() ) /T )  
        posself=sim_batch.diag() 
        neg=  sim_batch.sum(dim=1)  
       # lossRS=-torch.log(( posself+0.00001) /(neg+0.00001) ).mean()

        lossRS= -alpha*torch.log( posself+0.00001)+(1-alpha)*torch.log(neg+0.00001) 
        lossRS=lossRS.mean()
 
        return  lossRS 

    def forward(self, users, items):
        users = users.long()
        items = items.long()
        users_emb = self.embedding_user(users)
        items_emb = self.embedding_item(items)
        scores = torch.sum(users_emb*items_emb, dim=1)
        return self.f(scores)
 


class LightGCN(BasicModel):
    def __init__(self, 
                 config:dict, 
                 dataset:BasicDataset):
        super(LightGCN, self).__init__()
        self.config = config
        self.dataset : dataloader.BasicDataset = dataset
        self.__init_weight()

    # def getSparseEye(self,num):
    #     i = torch.LongTensor([[k for k in range(0,num)],[j for j in range(0,num)]])
    #     val = torch.FloatTensor([1]*num)
    #     return torch.sparse.FloatTensor(i,val)
    


    def __init_weight(self):
        self.num_users  = self.dataset.n_users
        self.num_items  = self.dataset.m_items
        self.latent_dim = self.config['latent_dim_rec']
        self.n_layers = self.config['lightGCN_n_layers']
        self.keep_prob = self.config['keep_prob']
        self.A_split = self.config['A_split']

        self.T=self.config['temperature']

      

        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        if self.config['pretrain'] == 0:
#             nn.init.xavier_uniform_(self.embedding_user.weight, gain=1)
#             nn.init.xavier_uniform_(self.embedding_item.weight, gain=1)
#             print('use xavier initilizer')
# random normal init seems to be a better choice when lightGCN actually don't use any non-linear activation function
            nn.init.normal_(self.embedding_user.weight, std=0.1)
            nn.init.normal_(self.embedding_item.weight, std=0.1)
            world.cprint('use NORMAL distribution initilizer')
        else:
            self.embedding_user.weight.data.copy_(torch.from_numpy(self.config['user_emb']))
            self.embedding_item.weight.data.copy_(torch.from_numpy(self.config['item_emb']))
            print('use pretarined data')
        self.f = nn.Sigmoid()
        self.Graph = self.dataset.getSparseGraph()

 
        self.alphapara = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.alphapara.data.fill_(0.5)
       
         # print("save_txt")
    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index]#/keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g
 

    def drop_feature(self,x, drop_prob):
        drop_mask = torch.empty(
            (x.size(1), ),
            dtype=torch.float32,
            device=x.device).uniform_(0, 1) < drop_prob
        x = x.clone()
        x[:, drop_mask] = 0 
        return x


    def __dropout(self, keep_prob):
        if self.A_split:
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.Graph, keep_prob)
        return graph
     

    def computer(self):
        """
        propagate methods for lightGCN
        """       
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
 
        all_emb = torch.cat([users_emb, items_emb])
        embs = [all_emb]

        if self.config['dropout']:
            if self.training:
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.Graph     
        else:
            g_droped = self.Graph 

        for layer in range(self.n_layers):
            if self.A_split:
                temp_emb = []
                for f in range(len(g_droped)):
                    temp_emb.append(torch.sparse.mm(g_droped[f], all_emb))
                side_emb = torch.cat(temp_emb, dim=0)
                all_emb = side_emb
            else:
                all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)

        light_out=all_emb

                # #mean
                # embs = torch.stack(embs, dim=1) 
                # light_out = torch.mean(embs, dim=1)
                #sg
        light_out= embs[-1] 
        users, items = torch.split(light_out, [self.num_users, self.num_items])

        return users, items 


    def getUsersRating(self, users):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
   
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating
    
    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego
    
    
    def bpr_loss(self, users, pos, neg):
        (users_emb, pos_emb, neg_emb, 
        userEmb0,  posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())
        reg_loss = (1/2)*(userEmb0.norm(2).pow(2) + 
                         posEmb0.norm(2).pow(2)  +
                         negEmb0.norm(2).pow(2))/float(len(users))
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)
        
        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        
        return loss, reg_loss

#fei qita zhegnli
############################################################################################################################################
    def bpr_loss_gcl_Kpos(self,users_emb0,pos_emb0,users,pos_items,alpha):

        alpha=self.alphapara
        users_emb=   users_emb0[users] 
        loss=self.bpr_loss_gcl_unit(users_emb,pos_emb0[pos_items[:,0]],alpha)
        for i in range(pos_items.size()[1])[1:]:
            pos_emb=   pos_emb0[pos_items[:,i]]  
            loss=loss+self.bpr_loss_gcl_unit(users_emb,pos_emb,alpha)

        return  loss.mean() 
    
 
 
 
    def bpr_loss_gcl_unit(self,users_emb,pos_emb,alpha): 
        T=self.T 
 
        ##ori
        sim_batch=torch.exp(torch.mm(users_emb,pos_emb.t() ) /T )  
 
        posself=sim_batch.diag() 
        neg=  sim_batch.sum(dim=1)    ########################################
        
        lossRS= -alpha*torch.log( posself+0.00001)+(1-alpha)*torch.log(neg+0.00001) 
        lossRS=lossRS.mean()
 
        return  lossRS 
 
####################################################################################
#
    def forward(self, users, items):
        # compute embedding
        all_users, all_items = self.computer()
        # print('forward')
        #all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb)
        gamma     = torch.sum(inner_pro, dim=1)
        return gamma


        # pos=torch.exp((torch.mm(users_emb,pos_emb.t() ) /T)).diag()
        # pos=torch.cat([pos,pos]) 
        # allui=torch.cat([users_emb,pos_emb])
        # neg=torch.exp((torch.mm(allui,allui.t() ) /T)).sum(dim=1)
        # loss= -torch.log(pos/neg).mean() 
        