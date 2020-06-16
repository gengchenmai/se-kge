import torch
import torch.nn as nn
from torch.nn import init

import numpy as np
import torch.nn.functional as F
"""
A set of decoder modules.
Each decoder takes pairs of embeddings and predicts relationship scores given these embeddings.
"""

""" 
*Edge decoders*
For all edge decoders, the forward method returns a simple relationships score, 
i.e. the likelihood of an edge, between a pair of nodes.
"""

class CosineEdgeDecoder(nn.Module):
    """
    Simple decoder where the relationship score is just the cosine
    similarity between the two embeddings.
    Note: this does not distinguish between edges types
    """

    def __init__(self):
        super(CosineEdgeDecoder, self).__init__()
        self.cos = nn.CosineSimilarity(dim=0)

    def forward(self, embeds1, embeds2, rel):
        # cosine, the larger, the better
        return self.cos(embeds1, embeds2)

class DotProductEdgeDecoder(nn.Module):
    """
    Simple decoder where the relationship score is just the dot product
    between the embeddings (i.e., unnormalized version of cosine)
    Note: this does not distinguish between edges types
    """

    def __init__(self):
        super(DotProductEdgeDecoder, self).__init__()

    def forward(self, embeds1, embeds2, rel):
        dots = torch.sum(embeds1 * embeds2, dim=0)
        return dots

class BilinearEdgeDecoder(nn.Module):
    """
    Decoder where the relationship score is given by a bilinear form
    between the embeddings (i.e., one learned matrix per relationship type).
    """

    def __init__(self, relations, dims):
        super(BilinearEdgeDecoder, self).__init__()
        self.relations = relations
        self.mats = {}
        self.cos = nn.CosineSimilarity(dim=0)
        for r1 in relations:
            for r2 in relations[r1]:
                rel = (r1, r2[1], r2[0])
                self.mats[rel] = nn.Parameter(
                        torch.FloatTensor(dims[rel[0]], dims[rel[2]]))
                init.xavier_uniform(self.mats[rel])
                self.register_parameter("_".join(rel), self.mats[rel])


    def forward(self, embeds1, embeds2, rel):
        '''
        embeds1, embeds2 shape: [embed_dim, batch_size]
        '''
        acts = embeds1.t().mm(self.mats[rel])
        return self.cos(acts.t(), embeds2)

class TransEEdgeDecoder(nn.Module):
    """
    Decoder where the relationship score is given by translation of
    the embeddings (i.e., one learned vector per relationship type).
    """

    def __init__(self, relations, dims):
        super(TransEEdgeDecoder, self).__init__()
        self.relations = relations
        self.vecs = {}
        for r1 in relations:
            for r2 in relations[r1]:
                rel = (r1, r2[1], r2[0])
                self.vecs[rel] = nn.Parameter(torch.FloatTensor(dims[rel[0]]))
                init.uniform(self.vecs[rel], a=-6.0/np.sqrt(dims[rel[0]]), b=6.0/np.sqrt(dims[rel[0]]))
                self.register_parameter("_".join(rel), self.vecs[rel])

    def forward(self, embeds1, embeds2, rel):
        '''
        embeds1, embeds2 shape: [embed_dim, batch_size]
        '''
        trans_embed = embeds1 + self.vecs[rel].unsqueeze(1).expand(self.vecs[rel].size(0), embeds1.size(1))
        trans_dist = (trans_embed - embeds2).pow(2).sum(0)
        # trans_dist shape: [batch_size]
        # TransE distance, the smaller, the better
        return -trans_dist

    

class BilinearDiagEdgeDecoder(nn.Module):
    """
    Decoder where the relationship score is given by a bilinear form
    between the embeddings (i.e., one learned diagonal matrix per relationship type).
    """

    def __init__(self, relations, dims):
        super(BilinearDiagEdgeDecoder, self).__init__()
        self.relations = relations
        self.vecs = {}
        for r1 in relations:
            for r2 in relations[r1]:
                rel = (r1, r2[1], r2[0])
                self.vecs[rel] = nn.Parameter(torch.FloatTensor(dims[rel[0]]))
                init.uniform(self.vecs[rel], a=-6.0/np.sqrt(dims[rel[0]]), b=6.0/np.sqrt(dims[rel[0]]))
                self.register_parameter("_".join(rel), self.vecs[rel])

    def forward(self, embeds1, embeds2, rel):
        '''
        embeds1, embeds2 shape: [embed_dim, batch_size]
        '''
        acts = (embeds1*self.vecs[rel].unsqueeze(1).expand(self.vecs[rel].size(0), embeds1.size(1))*embeds2).sum(0)
        return acts

 

""" 
*Metapath decoders*
For all metapath encoders, the forward method returns a compositonal relationships score, 
i.e. the likelihood of compositonional relationship or metapath, between a pair of nodes.
"""

class BilinearMetapathDecoder(nn.Module):
    """
    Each edge type is represented by a matrix, and
    compositional relationships are a product matrices.
    """

    def __init__(self, relations, dims):
        '''
        Args:
            relations: a dict() of all triple templates
                key:    domain entity type
                value:  a list of tuples (range entity type, predicate)
            dims: a dict(), node type => embed_dim of node embedding
        '''
        super(BilinearMetapathDecoder, self).__init__()
        self.relations = relations
        self.mats = {}
        self.sigmoid = torch.nn.Sigmoid()
        self.cos = nn.CosineSimilarity(dim=0)
        for r1 in relations:
            for r2 in relations[r1]:
                rel = (r1, r2[1], r2[0])
                self.mats[rel] = nn.Parameter(torch.FloatTensor(dims[rel[0]], dims[rel[2]]))
                init.xavier_uniform(self.mats[rel])
                self.register_parameter("_".join(rel), self.mats[rel])

    def forward(self, embeds1, embeds2, rels):
        '''
        embeds1, embeds2 shape: [embed_dim, batch_size]
        rels: a list of triple templates, a n-length metapath
        '''
        act = embeds1.t()
        for i_rel in rels:
            act = act.mm(self.mats[i_rel])
        act = self.cos(act.t(), embeds2)
        return act

    def project(self, embeds, rel):
        '''
        embeds shape: [embed_dim, batch_size]
        rel: triple template
        '''
        return self.mats[rel].mm(embeds)


class BilinearBlockDiagMetapathDecoder(nn.Module):
    """
    This is only used for enc_agg_func == "concat"
    Each edge type is represented by two matrix:
    1) feature matrix for node featur embed
    2) position matrix for node position embed
    It can be seen as a block-diagal matrix
    compositional relationships are a product matrices.
    """

    def __init__(self, relations, dims, feat_dims, spa_embed_dim):
        '''
        Args:
            relations: a dict() of all triple templates
                key:    domain entity type
                value:  a list of tuples (range entity type, predicate)
            dims: a dict(), node type => embed_dim of node embedding
            feat_dims: a dict(), node type => embed_dim of feature embedding
            spa_embed_dim: the embed_dim of position embedding
        '''
        super(BilinearBlockDiagMetapathDecoder, self).__init__()
        self.relations = relations
        self.dims = dims
        self.feat_dims = feat_dims
        self.spa_embed_dim = spa_embed_dim

        self.feat_mats = {}
        self.pos_mats = {}
        self.sigmoid = torch.nn.Sigmoid()
        self.cos = nn.CosineSimilarity(dim=0)
        for r1 in relations:
            for r2 in relations[r1]:
                rel = (r1, r2[1], r2[0])
                self.feat_mats[rel] = nn.Parameter(torch.FloatTensor(feat_dims[rel[0]], feat_dims[rel[2]]))
                init.xavier_uniform(self.feat_mats[rel])
                self.register_parameter("feat-"+"_".join(rel), self.feat_mats[rel])

                self.pos_mats[rel] = nn.Parameter(torch.FloatTensor(spa_embed_dim, spa_embed_dim))
                init.xavier_uniform(self.pos_mats[rel])
                self.register_parameter("pos-"+"_".join(rel), self.pos_mats[rel])

    def forward(self, embeds1, embeds2, rels):
        '''
        embeds1, embeds2 shape: [embed_dim, batch_size]
        rels: a list of triple templates, a n-length metapath
        '''
        # act: [batch_size, embed_dim]
        act = embeds1.t()
        feat_act, pos_act = torch.split(act, 
            [self.feat_dims[rels[0][0]],self.spa_embed_dim], dim=1)
        for i_rel in rels:
            feat_act = feat_act.mm(self.feat_mats[i_rel])
            pos_act = pos_act.mm(self.pos_mats[i_rel])
        #  act: [batch_size, embed_dim]
        act = torch.cat([feat_act, pos_act], dim=1)
        act = self.cos(act.t(), embeds2)
        return act

    def project(self, embeds, rel):
        '''
        embeds shape: [embed_dim, batch_size]
        rel: triple template
        '''
        feat_act, pos_act = torch.split(embeds.t(), 
            [self.feat_dims[rel[0]],self.spa_embed_dim], dim=1)
        feat_act = feat_act.mm(self.feat_mats[rel])
        pos_act = pos_act.mm(self.pos_mats[rel])
        act = torch.cat([feat_act, pos_act], dim=1)
        return act.t()

class BilinearBlockDiagPos2FeatMatMetapathDecoder(nn.Module):
    """
    This is only used for enc_agg_func == "concat"
    Each edge type is represented by two matrix:
    1) feature matrix for node featur embed
    2) position matrix for node position embed
    It can be seen as a block-diagal matrix
    compositional relationships are a product matrices.
    """

    def __init__(self, relations, dims, feat_dims, spa_embed_dim):
        '''
        Args:
            relations: a dict() of all triple templates
                key:    domain entity type
                value:  a list of tuples (range entity type, predicate)
            dims: a dict(), node type => embed_dim of node embedding
            feat_dims: a dict(), node type => embed_dim of feature embedding
            spa_embed_dim: the embed_dim of position embedding
        '''
        super(BilinearBlockDiagPos2FeatMatMetapathDecoder, self).__init__()
        self.relations = relations
        self.dims = dims
        self.feat_dims = feat_dims
        self.spa_embed_dim = spa_embed_dim

        self.feat_mats = {}
        self.pos_mats = {}
        self.pos2feat_mats = {}
        self.sigmoid = torch.nn.Sigmoid()
        self.cos = nn.CosineSimilarity(dim=0)
        for r1 in relations:
            for r2 in relations[r1]:
                rel = (r1, r2[1], r2[0])
                self.feat_mats[rel] = nn.Parameter(torch.FloatTensor(feat_dims[rel[0]], feat_dims[rel[2]]))
                init.xavier_uniform(self.feat_mats[rel])
                self.register_parameter("feat-"+"_".join(rel), self.feat_mats[rel])

                self.pos_mats[rel] = nn.Parameter(torch.FloatTensor(spa_embed_dim, spa_embed_dim))
                init.xavier_uniform(self.pos_mats[rel])
                self.register_parameter("pos-"+"_".join(rel), self.pos_mats[rel])

                self.pos2feat_mats[rel] = nn.Parameter(torch.FloatTensor(spa_embed_dim, feat_dims[rel[2]]))
                init.xavier_uniform(self.pos2feat_mats[rel])
                self.register_parameter("pos2feat-"+"_".join(rel), self.pos2feat_mats[rel])

    def forward(self, embeds1, embeds2, rels, do_spa_sem_lift = False):
        '''
        embeds1, embeds2 shape: [embed_dim, batch_size]
        rels: a list of triple templates, a n-length metapath
        do_spa_sem_lift: whether to do pos embed to feat embed prediction
        '''
        # act: [batch_size, embed_dim]
        act = embeds1.t()
        if do_spa_sem_lift:
            # make sure the input is a pos_embed
            assert act.size()[1] == self.spa_embed_dim
            # act: [batch_size, spa_embed_dim]
            rel = rels[0]
            # feat_act: [batch_size, embed_dim]
            feat_act = act.mm(self.pos2feat_mats[rel])
            # pos_act: [batch_size, spa_embed_dim]
            pos_act = act.mm(self.pos_mats[rel])
            for i_rel in rels[1:]:
                feat_act = feat_act.mm(self.feat_mats[i_rel])
                pos_act = pos_act.mm(self.pos_mats[i_rel])
        else:
            feat_act, pos_act = torch.split(act, 
                [self.feat_dims[rels[0][0]],self.spa_embed_dim], dim=1)
            for i_rel in rels:
                feat_act = feat_act.mm(self.feat_mats[i_rel])
                pos_act = pos_act.mm(self.pos_mats[i_rel])
        #  act: [batch_size, embed_dim+spa_embed_dim]
        act = torch.cat([feat_act, pos_act], dim=1)
        act = self.cos(act.t(), embeds2)
        return act

    def project(self, embeds, rel, do_spa_sem_lift = False):
        '''
        embeds shape: [embed_dim, batch_size]
        rel: triple template
        do_spa_sem_lift: whether to do pos embed to feat embed prediction
        '''
        # act: [batch_size, embed_dim]
        act = embeds.t()
        if do_spa_sem_lift:
            # make sure the input is a pos_embed
            assert act.size()[1] == self.spa_embed_dim
            # feat_act: [batch_size, embed_dim]
            feat_act = act.mm(self.pos2feat_mats[rel])
            # pos_act: [batch_size, spa_embed_dim]
            pos_act = act.mm(self.pos_mats[rel])
        else:
            feat_act, pos_act = torch.split(act, 
                [self.feat_dims[rel[0]],self.spa_embed_dim], dim=1)
            feat_act = feat_act.mm(self.feat_mats[rel])
            pos_act = pos_act.mm(self.pos_mats[rel])
        
        act = torch.cat([feat_act, pos_act], dim=1)
        return act.t()

   
class DotBilinearMetapathDecoder(nn.Module):
    """
    Each edge type is represented by a matrix, and
    compositional relationships are a product matrices.
    """

    def __init__(self, relations, dims):
        super(DotBilinearMetapathDecoder, self).__init__()
        self.relations = relations
        self.mats = {}
        self.sigmoid = torch.nn.Sigmoid()
        for r1 in relations:
            for r2 in relations[r1]:
                rel = (r1, r2[1], r2[0])
                self.mats[rel] = nn.Parameter(torch.FloatTensor(dims[rel[0]], dims[rel[2]]))
                #init.xavier_uniform(self.mats[rel])
                init.normal(self.mats[rel], std=0.1)
                self.register_parameter("_".join(rel), self.mats[rel])

    def forward(self, embeds1, embeds2, rels):
        act = embeds1.t()
        for i_rel in rels:
            act = act.mm(self.mats[i_rel])
        dots = torch.sum(act * embeds2, dim=0)
        return dots


class TransEMetapathDecoder(nn.Module):
    """
    Decoder where the relationship score is given by translation of
    the embeddings, each relation type is represented by a vector, and
    compositional relationships are addition of these vectors
    """

    def __init__(self, relations, dims):
        super(TransEMetapathDecoder, self).__init__()
        self.relations = relations
        self.vecs = {}
        for r1 in relations:
            for r2 in relations[r1]:
                rel = (r1, r2[1], r2[0])
                self.vecs[rel] = nn.Parameter(torch.FloatTensor(dims[rel[0]]))
                init.uniform(self.vecs[rel], a=-6.0/np.sqrt(dims[rel[0]]), b=6.0/np.sqrt(dims[rel[0]]))
                self.register_parameter("_".join(rel), self.vecs[rel])
        self.cos = nn.CosineSimilarity(dim=0)

    def forward(self, embeds1, embeds2, rels):
        trans_embed = embeds1
        for i_rel in rels:
            trans_embed += self.vecs[i_rel].unsqueeze(1).expand(self.vecs[i_rel].size(0), embeds1.size(1))
        trans_dist = self.cos(embeds2, trans_embed)
        return trans_dist

    def project(self, embeds, rel):
        return embeds + self.vecs[rel].unsqueeze(1).expand(self.vecs[rel].size(0), embeds.size(1))


class BilinearDiagMetapathDecoder(nn.Module):
    """
    Decoder where the relationship score is given by a bilinear form
    between the embeddings (i.e., one learned diagonal matrix per relationship type).
    """

    def __init__(self, relations, dims):
        super(BilinearDiagMetapathDecoder, self).__init__()
        self.relations = relations
        self.vecs = {}
        for r1 in relations:
            for r2 in relations[r1]:
                rel = (r1, r2[1], r2[0])
                self.vecs[rel] = nn.Parameter(torch.FloatTensor(dims[rel[0]]))
                init.uniform(self.vecs[rel], a=-6.0/np.sqrt(dims[rel[0]]), b=6.0/np.sqrt(dims[rel[0]]))
                self.register_parameter("_".join(rel), self.vecs[rel])

    def forward(self, embeds1, embeds2, rels):
        acts = embeds1
        for i_rel in rels:
            acts = acts*self.vecs[i_rel].unsqueeze(1).expand(self.vecs[i_rel].size(0), embeds1.size(1))
        acts = (acts*embeds2).sum(0)
        return acts

    def project(self, embeds, rel):
        return embeds*self.vecs[rel].unsqueeze(1).expand(self.vecs[rel].size(0), embeds.size(1))



"""
Set intersection operators. (Experimental)
"""

class TensorIntersection(nn.Module):
    """
    Decoder that computes the implicit intersection between two state vectors
    Uses a symmetric tensor operation.
    """
    def __init__(self, dims):
        super(TensorIntersection, self).__init__()
        self.inter_tensors = {}
        for mode in dims:
            dim = dims[mode]
            self.inter_tensors[mode] = nn.Parameter(torch.FloatTensor(dim, dim, dim))
            init.xavier_uniform(self.inter_tensors[mode])
            self.register_parameter(mode+"_mat", self.inter_tensors[mode])

    def forward(self, embeds1, embeds2, mode):
        '''
        embeds1, embeds2 shape: [embed_dim, batch_size]
        '''
        inter_tensor = self.inter_tensors[mode] 
        tensor_size = inter_tensor.size()
        inter_tensor = inter_tensor.view(tensor_size[0]*tensor_size[1], tensor_size[2])

        temp1 = inter_tensor.mm(embeds1)
        temp1 = temp1.view(tensor_size[0], tensor_size[1], embeds2.size(1))
        temp2 = inter_tensor.mm(embeds2)
        temp2 = temp2.view(tensor_size[0], tensor_size[1], embeds2.size(1))
        result = (temp1*temp2).sum(dim=1)
        return result

class SetIntersection(nn.Module):
    """
    Decoder that computes the implicit intersection between two state vectors
    Applies an MLP and takes elementwise mins. then another MLP
    """
    def __init__(self, mode_dims, expand_dims, use_relu = True, use_post_mat = True, agg_func=torch.min):
        '''
        Args:
            mode_dims: the input embedding dim
            expand_dims: the internal hidden state dim
        '''
        super(SetIntersection, self).__init__()
        self.pre_mats = {}
        self.use_post_mat = use_post_mat
        if self.use_post_mat:
            self.post_mats = {}
        self.use_relu = use_relu
        self.agg_func = agg_func
        for mode in mode_dims:
            self.pre_mats[mode] = nn.Parameter(torch.FloatTensor(expand_dims[mode], mode_dims[mode])) # [expand_embed_dim, embed_dim]
            init.xavier_uniform(self.pre_mats[mode])
            self.register_parameter(mode+"_premat", self.pre_mats[mode])
            if self.use_post_mat:
                self.post_mats[mode] = nn.Parameter(torch.FloatTensor(mode_dims[mode], expand_dims[mode]))
                init.xavier_uniform(self.post_mats[mode])
                self.register_parameter(mode+"_postmat", self.post_mats[mode])

    # def forward(self, embeds1, embeds2, mode, embeds3 = []):
    #     '''
    #     Args:
    #         embeds1, embeds2 shape: [embed_dim, batch_size]
    #         embeds3: a list of [embed_dim, batch_size]
    #     Return:
    #         aggs: the computed embedding for the intersection variable, [mode_dims, batch_size] 
    #         combined: the pre-intersect embeddings for each path, [num_query_path, expand_embed_dim, batch_size]
    #     '''
        
    #     # temp1, temp2 shape: [expand_embed_dim, batch_size]
    #     temp1_ = self.pre_mats[mode].mm(embeds1)
    #     temp1 = F.relu(temp1_)
    #     temp2_ = self.pre_mats[mode].mm(embeds2)
    #     temp2 = F.relu(temp2_)
    #     if len(embeds3) > 0:
    #         temp3_ = self.pre_mats[mode].mm(embeds3)
    #         temp3 = F.relu(temp3_)
    #         # concatenate sequence of tensors along a new dimension(dim=0 default)
    #         if not self.use_relu:
    #             combined_ = torch.stack([temp1_, temp2_, temp3_])
    #         combined = torch.stack([temp1, temp2, temp3])
    #     else:
    #         if not self.use_relu:
    #             combined_ = torch.stack([temp1_, temp2_])
    #         combined = torch.stack([temp1, temp2])
    #     aggs = self.agg_func(combined,dim=0)
    #     if type(aggs) == tuple:
    #         # For torch.min, the result is a tuple (min_value, index_tensor), we just get the 1st
    #         # For torch.mean, the result is just mean_value
    #         # so we need to check the result type
    #         aggs = aggs[0]
    #     if self.use_post_mat:
    #         aggs = self.post_mats[mode].mm(aggs)
    #     # aggs: [mode_dims, batch_size]
    #     if self.use_relu:
    #         return aggs, combined
    #     else:
    #         return aggs, combined_

    def forward(self, mode, embeds_list):
        '''
        Args:
            embeds_list: a list of embeds with shape [embed_dim, batch_size]
        Return:
            aggs: the computed embedding for the intersection variable, [mode_dims, batch_size] 
            combined: the pre-intersect embeddings for each path, [num_query_path, expand_embed_dim, batch_size]
        '''
        if len(embeds_list) < 2:
            raise Exception("The intersection needs more than one embeding")

        combined = []
        combined_ = []
        for i in range(len(embeds_list)):
            embeds = embeds_list[i]
            temp_ = self.pre_mats[mode].mm(embeds)
            temp = F.relu(temp_)

            if not self.use_relu:
                combined_.append(temp_)
            combined.append(temp)

        if not self.use_relu:
            combined_ =  torch.stack(combined_)
        combined =  torch.stack(combined)
        aggs = self.agg_func(combined,dim=0)
        if type(aggs) == tuple:
            # For torch.min, the result is a tuple (min_value, index_tensor), we just get the 1st
            # For torch.mean, the result is just mean_value
            # so we need to check the result type
            aggs = aggs[0]
        if self.use_post_mat:
            aggs = self.post_mats[mode].mm(aggs)
        # aggs: [mode_dims, batch_size]
        if self.use_relu:
            return aggs, combined
        else:
            return aggs, combined_


       
class SimpleSetIntersection(nn.Module):
    """
    Decoder that computes the implicit intersection between two state vectors.
    Takes a simple element-wise min.
    """
    def __init__(self, agg_func=torch.min):
        super(SimpleSetIntersection, self).__init__()
        self.agg_func = agg_func

    # def forward(self, embeds1, embeds2, mode, embeds3 = []):
    #     if len(embeds3) > 0:
    #         combined = torch.stack([embeds1, embeds2, embeds3])
    #     else:
    #         combined = torch.stack([embeds1, embeds2])
    #     aggs = self.agg_func(combined, dim=0)
    #     if type(aggs) == tuple:
    #         aggs = aggs[0]
    #     return aggs, combined

    def forward(self, mode, embeds_list):
        if len(embeds_list) < 2:
            raise Exception("The intersection needs more than one embeding")

        combined = torch.stack(embeds_list)
        aggs = self.agg_func(combined, dim=0)
        if type(aggs) == tuple:
            aggs = aggs[0]
        return aggs, combined