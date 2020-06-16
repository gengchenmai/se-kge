import torch
import torch.nn as nn
from torch.nn import init

import math
import numpy as np
import torch.nn.functional as F

from netquery.encoders import LayerNorm


'''
A set of attention method used for query attention learning
'''
class IntersectConcatAttention(nn.Module):
    def __init__(self, query_dims, key_dims, num_attn = 1, activation = "leakyrelu", f_activation = "sigmoid", 
        layernorm = False, use_post_mat = False):
        '''
        The attention method used by Graph Attention network (LeakyReLU)
        Args:
            query_dims: a dict() mapping: node type --> pre-computed variable embeddings dimention
            key_dims: a dict() mapping: node type --> embeddings dimention computed from different query path for the same variables
            num_attn: number of attention head
            activation: the activation function to atten_vecs * torch.cat(query_embed, key_embed), see GAT paper Equ 3
            f_activation: the final activation function applied to get the final result, see GAT paper Equ 6
        ''' 
        super(IntersectConcatAttention, self).__init__()
        self.atten_vecs = {}
        self.query_dims = query_dims
        self.key_dims = key_dims
        self.num_attn = num_attn

        # define the layer normalization
        self.layernorm = layernorm
        if self.layernorm:
            self.lns = {}
            for mode in query_dims:
                self.lns[mode] = LayerNorm(query_dims[mode])
                self.add_module(mode+"_ln", self.lns[mode])

        self.use_post_mat = use_post_mat
        if self.use_post_mat:
            self.post_W = {}
            self.post_B = {}
            if self.layernorm:
                self.post_lns = {}
            for mode in query_dims:
                self.post_W[mode] = nn.Parameter(torch.FloatTensor(query_dims[mode], query_dims[mode]))
                init.xavier_uniform(self.post_W[mode])
                self.register_parameter(mode+"_attnPostW", self.post_W[mode])

                self.post_B[mode] = nn.Parameter(torch.FloatTensor(query_dims[mode], 1))
                init.xavier_uniform(self.post_B[mode])
                self.register_parameter(mode+"_attnPostB",self.post_B[mode])
                if self.layernorm:
                    self.post_lns[mode] = LayerNorm(query_dims[mode])
                    self.add_module(mode+"_attnPostln", self.post_lns[mode])


        if activation == "leakyrelu":
            self.activation = nn.LeakyReLU(negative_slope=0.2)
        elif activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        else:
            raise Exception("activation not recognized.")
        if f_activation == "leakyrelu":
            self.f_activation = nn.LeakyReLU(negative_slope=0.2)
        elif f_activation == "relu":
            self.f_activation = nn.ReLU()
        elif f_activation == "sigmoid":
            self.f_activation = nn.Sigmoid()
        else:
            raise Exception("attention final activation not recognized.")
        self.softmax = nn.Softmax(dim=0)
        for mode in query_dims:
            # each column represent an attention vector for one attention head: [embed_dim*2, num_attn]
            self.atten_vecs[mode] = nn.Parameter(torch.FloatTensor(query_dims[mode]+key_dims[mode], self.num_attn))
            init.xavier_uniform(self.atten_vecs[mode])
            self.register_parameter(mode+"_attenvecs", self.atten_vecs[mode])

    def forward(self, query_embed, key_embeds, mode):
        '''
        Args:
            query_embed: the pre-computed variable embeddings, [embed_dim, batch_size]
            key_embeds: a list of embeddings computed from different query path for the same variables, [num_query_path, embed_dim, batch_size]
            mode: node type
        Return:
            combined: the multi-head attention based embeddings for a variable [embed_dim, batch_size]
        '''
        tensor_size = key_embeds.size()
        num_query_path = tensor_size[0]
        batch_size = tensor_size[2]
        # query_embed_expand: [num_query_path, embed_dim, batch_size]
        query_embed_expand = query_embed.unsqueeze(0).expand_as(key_embeds)
        # concat: [num_query_path, batch_size, embed_dim*2]
        # concat = torch.cat((query_embed_expand, key_embeds), dim=1).view((num_query_path, batch_size, -1))
        concat = torch.cat((query_embed_expand, key_embeds), dim=1).transpose(1,2)
        # 1. compute the attention score for each key embeddings
        # attn: [num_query_path, batch_size, num_attn]
        attn = torch.einsum("nbd,dk->nbk", (concat, self.atten_vecs[mode]))
        # attn: [num_query_path, batch_size, num_attn]
        attn = self.softmax(self.activation(attn))
        # attn = attn.view(batch_size, self.num_attn, num_query_path)
        # attn: [batch_size, num_attn, num_context_pt]
        attn = attn.transpose(0,1).transpose(1,2)
        # key_embeds_: [batch_size, num_query_path, embed_dim]
        # key_embeds_ = key_embeds.view(batch_size, num_query_path, -1)
        key_embeds_ = key_embeds.transpose(1,2).transpose(0,1)
        # 2. using the attention score to compute the weighted average of the key embeddings
        # combined: [batch_size, num_attn, embed_dim]
        combined = torch.einsum("bkn,bnd->bkd", (attn, key_embeds_))
        # combined: [batch_size, embed_dim]
        combined = torch.sum(combined, dim=1,keepdim=False) * (1.0/self.num_attn)
        # combined: [embed_dim, batch_size]
        combined =  self.f_activation(combined).t()

        if self.layernorm:
            combined = combined + query_embed
            combined = self.lns[mode](combined.t()).t()

        if self.use_post_mat:
            linear = self.post_W[mode].mm(combined) + self.post_B[mode]
            if self.layernorm:
                linear = linear + combined
                linear = self.post_lns[mode](linear.t()).t()
            return linear

        
        return combined


class IntersectDotProductAttention(nn.Module):
    def __init__(self, query_dims, key_dims, num_attn = 1, f_activation = "sigmoid", 
        dotproduct_scaled = True, layernorm = False, use_post_mat = False):
        '''
        The attention method used by "Attention Is All You Need" paper (dotproduct_scaled = True),
        dotproduct_scaled = False, for a normal dot product attention
        Args:
            query_dims: a dict() mapping: node type --> pre-computed variable embeddings dimention
            key_dims: a dict() mapping: node type --> embeddings dimention computed from different query path for the same variables
            num_attn: number of attention head
            activation: the activation function to atten_vecs * torch.cat(query_embed, key_embed), see GAT paper Equ 3
            f_activation: the final activation function applied to get the final result, see GAT paper Equ 6
        ''' 
        super(IntersectDotProductAttention, self).__init__()
        assert num_attn == 1
        self.query_dims = query_dims
        self.key_dims = key_dims
        self.num_attn = num_attn
        self.dotproduct_scaled = dotproduct_scaled
        # define the layer normalization
        self.layernorm = layernorm
        if self.layernorm:
            self.lns = {}
            for mode in query_dims:
                self.lns[mode] = LayerNorm(query_dims[mode])
                self.add_module(mode+"_ln", self.lns[mode])

        self.use_post_mat = use_post_mat
        if self.use_post_mat:
            self.post_W = {}
            self.post_B = {}
            if self.layernorm:
                self.post_lns = {}
            for mode in query_dims:
                self.post_W[mode] = nn.Parameter(torch.FloatTensor(query_dims[mode], query_dims[mode]))
                init.xavier_uniform(self.post_W[mode])
                self.register_parameter(mode+"_attnPostW", self.post_W[mode])

                self.post_B[mode] = nn.Parameter(torch.FloatTensor(query_dims[mode], 1))
                init.xavier_uniform(self.post_B[mode])
                self.register_parameter(mode+"_attnPostB",self.post_B[mode])
                if self.layernorm:
                    self.post_lns[mode] = LayerNorm(query_dims[mode])
                    self.add_module(mode+"_attnPostln", self.post_lns[mode])

        if f_activation == "leakyrelu":
            self.f_activation = nn.LeakyReLU(negative_slope=0.2)
        elif f_activation == "relu":
            self.f_activation = nn.ReLU()
        elif f_activation == "sigmoid":
            self.f_activation = nn.Sigmoid()
        else:
            raise Exception("attention final activation not recognized.")
        self.softmax = nn.Softmax(dim=0)
        for mode in query_dims:
            assert query_dims[mode] == key_dims[mode]
        

    def forward(self, query_embed, key_embeds, mode):
        '''
        Do the Dot product based attention based on "Attention Is All You Need" paper, Equ. 1
        Args:
            query_embed: the pre-computed variable embeddings, [embed_dim, batch_size]
            key_embeds: a list of embeddings computed from different query path for the same variables, [num_query_path, embed_dim, batch_size]
            mode: node type
        Return:
            combined: the multi-head attention based embeddings for a variable [embed_dim, batch_size]
        '''
        tensor_size = key_embeds.size()
        num_query_path = tensor_size[0]
        batch_size = tensor_size[2]
        # query_embed_expand: [num_query_path, embed_dim, batch_size]
        query_embed_expand = query_embed.unsqueeze(0).expand_as(key_embeds)
        #  do dot product between the query embedding with key embeddings
        # attn: [num_query_path, 1, batch_size]
        attn = torch.sum(query_embed_expand * key_embeds, dim=1, keepdim=True)
        if self.dotproduct_scaled:
            attn /= math.sqrt(self.query_dims[mode])
        # normalize the attention score across all query paths, [1, num_query_path, batch_size]
        attn = self.softmax(attn).view(1, num_query_path, batch_size)
        # [1, embed_dim, batch_size]
        combined = torch.einsum("inb,ndb->idb", (attn, key_embeds))

        combined = combined.squeeze(0)

        if self.layernorm:
            combined = combined + query_embed
            combined = self.lns[mode](combined.t()).t()

        if self.use_post_mat:
            linear = self.post_W[mode].mm(combined) + self.post_B[mode]
            if self.layernorm:
                linear = linear + combined
                linear = self.post_lns[mode](linear.t()).t()
            return linear

        return combined
        






        