import torch
import torch.nn as nn
import itertools
from torch.nn import init
from torch.autograd import Variable
import torch.nn.functional as F

import random
import math
import numpy as np

"""
Set of modules for aggregating embeddings of neighbors.
These modules take as input embeddings of neighbors.
"""

class MeanAggregator(nn.Module):
    """
    Aggregates a node's embeddings using mean of neighbors' embeddings
    Neighborhood sample:
    sample min(max_keep, len(to_neigh)*keep_prob) neighbors WITHOUT replacement as neighbor for each center node
    """
    def __init__(self, features, device = "cpu"): 
        """
        Initializes the aggregator for a specific graph.

        features         -- function mapping (node_list, features, offset) to feature values
                            see torch.nn.EmbeddingBag and forward function below docs for offset meaning.
        features(nodes, mode): a embedding lookup function to make a dict() from node type to embeddingbag
            nodes: a lists of global node id which are in type (mode)
            mode: node type
            return: embedding vectors, shape [num_node, embed_dim]
        """

        super(MeanAggregator, self).__init__()

        self.features = features
        self.device = device
        
    def forward(self, to_neighs, rel, keep_prob=0.5, max_keep=10):
        """
        Given a list of neighbors of nodes (to_neighs), 
        compute the average embedding of center nodes using the embeddings of neighbors

        Aggregates embeddings for a batch of nodes.
        keep_prob and max_keep are the parameters for edge/neighbour dropout.

        to_neighs -- list of neighbors of nodes
        keep_prob -- probability of keeping a neighbor
        max_keep  -- maximum number of neighbors kept per node

        rel: triple template, i.e. ('drug', 'psoriatic_arthritis', 'drug')
        """

        # Local pointers to functions (speed hack)
        _int = int
        _set = set
        _min = min
        _len = len
        _ceil = math.ceil
        _sample = random.sample
        # sample nodes from to_neighs
        samp_neighs = [_set(_sample(to_neigh, 
                        _min(_int(_ceil(_len(to_neigh)*keep_prob)), max_keep)
                        )) for to_neigh in to_neighs]
        unique_nodes_list = list(set.union(*samp_neighs))
        unique_nodes = {n:i for i,n in enumerate(unique_nodes_list)}
        # make an adjecent matrix from the original nodes to their sampled neighbors
        # row id: the index of original nodes, column: the index of sample neighbors in unique_nodes_list
        mask = Variable(torch.zeros(len(samp_neighs), len(unique_nodes)))
        column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]   
        row_indices = [i for i in range(len(samp_neighs)) for j in range(len(samp_neighs[i]))]
        mask[row_indices, column_indices] = 1
        
        mask = mask.to(self.device)

        num_neigh = mask.sum(1, keepdim=True) # [num_node, 1]
        mask = mask.div(num_neigh)
        # embedding lookup for all neighbor nodes
        embed_matrix = self.features(unique_nodes_list, rel[-1])
        # if only return one embeddeding vector, add one dim in the front
        if len(embed_matrix.size()) == 1:
            embed_matrix = embed_matrix.unsqueeze(dim=0) # [embed_dim] -> [1, embed_dim]
        # matrix multiplication, shape [len(to_neighs), embed_dim]
        to_feats = mask.mm(embed_matrix)
        return to_feats

class FastMeanAggregator(nn.Module):
    """
    Aggregates a node's embeddings using mean of neighbors' embeddings, 
    Compare to MeanAggregator, this just sample max_keep neighbors WITH replacement as neighbor for each center node
    """
    def __init__(self, features, device = "cpu"): 
        """
        Initializes the aggregator for a specific graph.

        features         -- function mapping (node_list, features, offset) to feature values
                            see torch.nn.EmbeddingBag and forward function below docs for offset meaning.
        """

        super(FastMeanAggregator, self).__init__()

        self.features = features
        
        self.device = device
        
    def forward(self, to_neighs, rel, keep_prob=None, max_keep=25):
        """
        Aggregates embeddings for a batch of nodes.
        keep_prob and max_keep are the parameters for edge/neighbour dropout.

        to_neighs -- list of neighbors of nodes
        keep_prob -- probability of keeping a neighbor
        max_keep  -- maximum number of neighbors kept per node

        rel: triple template, i.e. ('drug', 'psoriatic_arthritis', 'drug')
        keep_prob: do not use here
        """
        _random = random.random
        _int = int 
        _len = len
        samp_neighs = [to_neigh[_int(_random()*_len(to_neigh))] for i in itertools.repeat(None, max_keep) 
                for to_neigh in to_neighs]
        # embed_matrix shape: [len(to_neighs) * max_keep, embed_dim]
        embed_matrix = self.features(samp_neighs, rel[-1])
        # reshape embed_matrix
        # to_feats shape: [max_keep, len(to_neighs), embed_dim]
        to_feats = embed_matrix.view(max_keep, len(to_neighs), embed_matrix.size()[1])
        # output shape: [len(to_neighs), embed_dim]
        return to_feats.mean(dim=0)

class PoolAggregator(nn.Module):
    """
    Aggregates a node's embeddings using mean pooling of neighbors' embeddings
    Neighborhood sample:
    sample min(max_keep, len(to_neigh)*keep_prob) neighbors WITHOUT replacement as neighbor for each center node
    Mean pooling, pass all neighbor embedding to a weight matrix, then sum their vectors and use relu
    """
    def __init__(self, features, feature_dims, device = "cpu"): 
        """
        Initializes the aggregator for a specific graph.

        features         -- function mapping (node_list, features, offset) to feature values
                            see torch.nn.EmbeddingBag and forward function below docs for offset meaning.
        """

        super(PoolAggregator, self).__init__()

        self.features = features
        self.feat_dims = feature_dims
        self.pool_matrix = {}
        for mode, feat_dim in self.feat_dims.iteritems():
            self.pool_matrix[mode] = nn.Parameter(torch.FloatTensor(feat_dim, feat_dim))
            init.xavier_uniform(self.pool_matrix[mode])
            self.register_parameter(mode+"_pool", self.pool_matrix[mode])
        
        self.device = device
        
    def forward(self, to_neighs, rel, keep_prob=0.5, max_keep=10):
        """
        Aggregates embeddings for a batch of nodes.
        keep_prob and max_keep are the parameters for edge/neighbour dropout.

        to_neighs -- list of neighbors of nodes
        keep_prob -- probability of keeping a neighbor
        max_keep  -- maximum number of neighbors kept per node
        """
        _int = int
        _set = set
        _min = min
        _len = len
        _ceil = math.ceil
        _sample = random.sample
        samp_neighs = [_set(_sample(to_neigh, 
                        _min(_int(_ceil(_len(to_neigh)*keep_prob)), max_keep)
                        )) for to_neigh in to_neighs]
        unique_nodes_list = list(set.union(*samp_neighs))
        unique_nodes = {n:i for i,n in enumerate(unique_nodes_list)}
        # make an adjecent matrix from the original nodes to their sampled neighbors
        # row id: the index of original nodes, column: the index of sample neighbors in unique_nodes_list
        mask = Variable(torch.zeros(len(samp_neighs), len(unique_nodes)))
        column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]   
        row_indices = [i for i in range(len(samp_neighs)) for j in range(len(samp_neighs[i]))]
        mask[row_indices, column_indices] = 1
        mode = rel[0]
       
        mask = mask.to(self.device)

        # embed_matrix shape: [len(unique_nodes), embed_dim] = [len(unique_nodes), embed_dim] * [embed_dim, embed_dim]
        embed_matrix = self.features(unique_nodes, rel[-1]).mm(self.pool_matrix[mode])
        to_feats = F.relu(mask.mm(embed_matrix))
        # output shape: [len(to_neighs), embed_dim]
        return to_feats

class FastPoolAggregator(nn.Module):
    """
    Aggregates a node's embeddings using mean pooling of neighbors' embeddings
    Compare to PoolAggregator, this just sample max_keep neighbors WITH replacement as neighbor for each center node
    """
    def __init__(self, features, feature_dims,
            device = "cpu"): 
        """
        Initializes the aggregator for a specific graph.

        features         -- function mapping (node_list, features, offset) to feature values
                            see torch.nn.EmbeddingBag and forward function below docs for offset meaning.
        """

        super(FastPoolAggregator, self).__init__()

        self.features = features
        self.feat_dims = feature_dims
        self.pool_matrix = {}
        for mode, feat_dim in self.feat_dims.iteritems():
            self.pool_matrix[mode] = nn.Parameter(torch.FloatTensor(feat_dim, feat_dim))
            init.xavier_uniform(self.pool_matrix[mode])
            self.register_parameter(mode+"_pool", self.pool_matrix[mode])
        
        self.device = device
        
    def forward(self, to_neighs, rel, keep_prob=0.5, max_keep=10):
        """
        Aggregates embeddings for a batch of nodes.
        keep_prob and max_keep are the parameters for edge/neighbour dropout.

        to_neighs -- list of neighbors of nodes
        keep_prob -- probability of keeping a neighbor
        max_keep  -- maximum number of neighbors kept per node
        """
        _random = random.random
        _int = int 
        _len = len
        samp_neighs = [to_neigh[_int(_random()*_len(to_neigh))] for i in itertools.repeat(None, max_keep) 
                for to_neigh in to_neighs]
        mode = rel[0]
        embed_matrix = self.features(samp_neighs, rel[-1]).mm(self.pool_matrix[mode])
        to_feats = embed_matrix.view(max_keep, len(to_neighs), embed_matrix.size()[1])
        # output shape: [len(to_neighs), embed_dim]
        return to_feats.mean(dim=0)
