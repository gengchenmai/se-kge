import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from sets import Set
import random

from netquery.module import *
from netquery.SpatialRelationEncoder import *

"""
Set of modules for encoding nodes.
These modules take as input node ids and output embeddings.
"""

class DirectEncoder(nn.Module):
    """
    Encodes a node as a embedding via direct lookup.
    (i.e., this is just like basic node2vec or matrix factorization)
    """
    def __init__(self, features, feature_modules): 
        """
        Initializes the model for a specific graph.

        features         -- function mapping (node_list, features, offset) to feature values
                            see torch.nn.EmbeddingBag and forward function below docs for offset meaning.
        feature_modules  -- This should be a map from mode -> torch.nn.EmbeddingBag 

        features(nodes, mode): a embedding lookup function to make a dict() from node type to embeddingbag
            nodes: a lists of global node id which are in type (mode)
            mode: node type
            return: embedding vectors, shape [num_node, embed_dim]
        feature_modules: a dict of embedding matrix by node type, each embed matrix shape: [num_ent_by_type + 2, embed_dim]
        """
        super(DirectEncoder, self).__init__()
        for name, module in feature_modules.iteritems():
            self.add_module("feat-"+name, module)
        self.features = features

    def forward(self, nodes, mode, offset=None, **kwargs):
        """
        Generates embeddings for a batch of nodes.

        nodes     -- list of nodes
        mode      -- string desiginating the mode of the nodes
        offsets   -- specifies how the embeddings are aggregated. 
                     see torch.nn.EmbeddingBag for format. 
                     No aggregation if offsets is None
        """

        if offset is None:
            # the output is a dict() map: node type --> embedding tensor [num_ent, embed_dim]
            # t() transpose embedding tensor as [embed_dim, num_ent]
            embeds = self.features(nodes, mode).t()
            # calculate the L2-norm for each embedding vector, [1, num_ent]
            norm = embeds.norm(p=2, dim=0, keepdim=True)
            # normalize the embedding vectors
            # shape: [embed_dim, num_ent]
            # return embeds.div(norm.expand_as(embeds))
            return embeds.div(norm)
        else:
            return self.features(nodes, mode, offset).t()


class SimpleSpatialEncoder(nn.Module):
    """
    Encodes a node as a embedding via direct lookup. Encode its geographic coordicate, and sum them up
    (i.e., this is just like basic node2vec or matrix factorization)
    """
    def __init__(self, features, feature_modules, out_dims, id2geo): 
        """
        Initializes the model for a specific graph.

        features         -- function mapping (node_list, features, offset) to feature values
                            see torch.nn.EmbeddingBag and forward function below docs for offset meaning.
        feature_modules  -- This should be a map from mode -> torch.nn.EmbeddingBag 

        features(nodes, mode): a embedding lookup function to make a dict() from node type to embeddingbag
            nodes: a lists of global node id which are in type (mode)
            mode: node type
            return: embedding vectors, shape [num_node, embed_dim]
        feature_modules: a dict of embedding matrix by node type, each embed matrix shape: [num_ent_by_type + 2, embed_dim]
        out_dims: a dict()
            key: node type
            value: embedding dimention
        id2geo: a dict()
            key: node id
            value: a list, [longitude, lantitude]
        """
        super(SimpleSpatialEncoder, self).__init__()
        for name, module in feature_modules.iteritems():
            self.add_module("feat-"+name, module)
        self.features = features

        self.id2geo = id2geo

        # as for position encoding, we give a geographic coordinate transformation matrix
        # this require that different node type have the same embedding dimention
        self.embed_dim = out_dims[out_dims.keys()[0]]
        for mode in out_dims:
            assert out_dims[mode] == self.embed_dim
        # position encoding weight matrix
        self.geo_W = nn.Parameter(torch.FloatTensor(2, self.embed_dim))
        init.xavier_uniform(self.geo_W)
        self.register_parameter("geo_W", self.geo_W)
        # position encoding biase tem
        self.geo_B = nn.Parameter(torch.FloatTensor(1, self.embed_dim,))
        init.xavier_uniform(self.geo_B)
        self.register_parameter("geo_B", self.geo_B)

        # the position embedding for no geographic entity, same as the out-of-vocab token
        self.nogeo_embed = nn.Parameter(torch.FloatTensor(1, self.embed_dim))
        init.xavier_uniform(self.nogeo_embed)
        self.register_parameter("nogeo_embed", self.nogeo_embed)


    def forward(self, nodes, mode, offset=None, **kwargs):
        """
        Generates embeddings for a batch of nodes.

        nodes     -- list of nodes
        mode      -- string desiginating the mode of the nodes
        offsets   -- specifies how the embeddings are aggregated. 
                     see torch.nn.EmbeddingBag for format. 
                     No aggregation if offsets is None
        """

        if offset is None:
            # the output is a dict() map: node type --> embedding tensor [num_ent, embed_dim]
            # t() transpose embedding tensor as [embed_dim, num_ent]
            embeds = self.features(nodes, mode).t()
            # calculate the L2-norm for each embedding vector, [1, num_ent]
            norm = embeds.norm(p=2, dim=0, keepdim=True)
            # normalize the embedding vectors
            # shape: [embed_dim, num_ent]
            ent_embeds = embeds.div(norm.expand_as(embeds))
        else:
            ent_embeds = self.features(nodes, mode, offset).t()

        # coord_tensor: [batch_size, 2]
        # nogeo_khot: [batch_size]
        coord_tensor, nogeo_khot = geo_lookup(nodes, self.id2geo)
        

        # pos_embeds: [batch_size, embed_dim]
        pos_embeds = torch.FloatTensor(coord_tensor).mm(self.geo_W) + self.geo_B

        # nogeo_khot: [batch_size, 1]
        nogeo_khot = torch.FloatTensor(nogeo_khot).unsqueeze(1)
        # nogeo_tensor: [batch_size, embed_dim]
        nogeo_tensor = nogeo_khot * (self.nogeo_embed - self.geo_B)

        pos_embeds = nogeo_tensor + pos_embeds
        # pos_embeds: [embed_dim, batch_size]
        pos_embeds = pos_embeds.t()

        return pos_embeds + ent_embeds

def geo_lookup(nodes, id2geo, add_dim = -1, id2extent = None, doExtentSample = False):
    '''
    Given a list of node id, make a coordinate tensor and a nogeo indicator tensor
    Args:
        nodes: list of nodes id
        id2geo: a dict()
            key: node id
            value: a list, [longitude, lantitude]
    Return:
        coord_tensor: [batch_size, 2], geographic coordicate for geo_ent, [0.0, 0.0] for nogeo_ent
        nogeo_khot: [batch_size], 0 for geo_ent, 1 for nogeo_ent
    '''
    coord_tensor = []
    nogeo_khot = []
    for i, eid in enumerate(nodes):
        if eid in id2geo:
            if id2extent is None:
                coords = list(id2geo[eid])
            else:
                if eid in id2extent:
                    if doExtentSample:
                        xmin, xmax, ymin, ymax = id2extent[eid]
                        x = random.uniform(xmin, xmax)
                        y = random.uniform(ymin, ymax)
                        coords = [x, y]
                    else:
                        coords = list(id2geo[eid])
                else:
                    coords = list(id2geo[eid])
            if add_dim == -1:
                coord_tensor.append(coords)
            elif add_dim == 1:
                coord_tensor.append([coords])
            nogeo_khot.append(0)
        else:
            if add_dim == -1:
                coord_tensor.append([0.0, 0.0])
            elif add_dim == 1:
                coord_tensor.append([[0.0, 0.0]])
            nogeo_khot.append(1)

    return coord_tensor, nogeo_khot


class ExtentPositionEncoder(nn.Module):
    '''
    This is position encoder, a wrapper for different space encoder,
    Given a list of node ids, return their embedding
    '''
    def __init__(self, spa_enc_type, id2geo, id2extent, spa_enc, graph, spa_enc_embed_norm, device = "cpu"):
        '''
        Args:
            out_dims: a dict()
                key: node type
                value: embedding dimention
            
            spa_enc_type: the type of space encoder
            id2geo: a dict(): node id -> [longitude, latitude]
            id2extent: a dict(): node id -> (xmin, xmax, ymin, ymax)
            spa_enc: one space encoder
            graph: Graph()
            spa_enc_embed_norm: whether to do position embedding normalization
            
            
        '''
        super(ExtentPositionEncoder, self).__init__()
        self.spa_enc_type = spa_enc_type
        self.id2geo = id2geo
        self.id2extent = id2extent
        self.spa_embed_dim = spa_enc.spa_embed_dim # the output space embedding
        self.spa_enc = spa_enc
        self.graph = graph
        self.spa_enc_embed_norm = spa_enc_embed_norm
        self.device = device

        self.nogeo_idmap = self.make_nogeo_idmap(self.id2geo, self.graph)
        # random initialize the position embedding for nogeo entities
        # last index: indicate the geo-entity, use this for convinience
        self.nogeo_spa_embed_module = torch.nn.Embedding(len(self.nogeo_idmap)+1, self.spa_embed_dim).to(self.device)
        self.add_module("nogeo_pos_embed_matrix", self.nogeo_spa_embed_module)
        # define embedding initialization method: normal dist
        self.nogeo_spa_embed_module.weight.data.normal_(0, 1./self.spa_embed_dim)

    def nogeo_embed_lookup(self, nodes):
        '''
        nogeo_spa_embeds: the spa embed for no-geo entity, [batch_size, spa_embed_dim]
        Note for geo-entity, we use the last embedding in self.nogeo_spa_embed_module
        
        '''
        id_list = []
        for node in nodes:
            if node in self.nogeo_idmap:
                # if this is nogeo entity
                id_list.append(self.nogeo_idmap[node])
            else:
                # if this is geo entity
                id_list.append(len(self.nogeo_idmap))

        # nogeo_spa_embeds: (batch_size, spa_embed_dim)
        nogeo_spa_embeds = self.nogeo_spa_embed_module(
            torch.autograd.Variable(torch.LongTensor(id_list).to(self.device)))
        # calculate the L2-norm for each embedding vector, (batch_size, 1)
        norm = nogeo_spa_embeds.norm(p=2, dim=1, keepdim=True)
        # normalize the embedding vectors
        # shape: [batch_size, spa_embed_dim]
        return nogeo_spa_embeds.div(norm)

    def make_nogeo_idmap(self, id2geo, graph):
        '''
        nogeo_idmap: dict(), nogeo-entity id => local id
        '''
        id_set = Set()
        for mode in graph.full_sets:
            id_set.union(graph.full_sets[mode])
        # id_list = sorted(list(id_set))
        geo_set = Set(id2geo.keys())
        nogeo_set = id_set - geo_set
        nogeo_idmap = {nogeo_id: i for i, nogeo_id in enumerate(nogeo_set)}
        return nogeo_idmap

    def forward(self, nodes, do_test = False):
        '''
        Args:
            nodes: a list of node ids
        Return:
            pos_embeds: the position embedding for all nodes, (spa_embed_dim, batch_size)
                    geo_ent => space embedding from geographic coordinates
                    nogeo_ent => [0,0..,0]
        '''
        # coord_tensor: [batch_size, 1, 2], geographic coordicate for geo_ent, [0.0, 0.0] for nogeo_ent
        # nogeo_khot: [batch_size], 0 for geo_ent, 1 for nogeo_ent
        coord_tensor, nogeo_khot = geo_lookup(nodes, 
                                    id2geo = self.id2geo, 
                                    add_dim = 1, 
                                    id2extent = self.id2extent, 
                                    doExtentSample = True)

        # spa_embeds: (batch_size, 1, spa_embed_dim)
        spa_embeds = self.spa_enc(coord_tensor)
        # spa_embeds: (batch_size, spa_embed_dim)
        spa_embeds = torch.squeeze(spa_embeds, dim = 1)

        nogeo_khot = torch.FloatTensor(nogeo_khot).to(self.device)
        # mask: (batch_size, 1)
        mask = torch.unsqueeze(nogeo_khot, dim=1)
        # pos_embeds: (batch_size, spa_embed_dim), erase nogeo embed as [0,0...]
        pos_embeds = spa_embeds * (1 - mask)

        # nogeo_spa_embeds: (batch_size, spa_embed_dim)
        nogeo_spa_embeds = self.nogeo_embed_lookup(nodes)
        # nogeo_pos_embeds: (batch_size, spa_embed_dim), erase geo embed as [0,0...]
        nogeo_pos_embeds = nogeo_spa_embeds * mask

        # pos_embeds: (batch_size, spa_embed_dim)
        pos_embeds = pos_embeds + nogeo_pos_embeds

        # pos_embeds: (spa_embed_dim, batch_size)
        pos_embeds = pos_embeds.t()

        if self.spa_enc_embed_norm:
            # calculate the L2-norm for each embedding vector, (1, batch_size)
            norm = pos_embeds.norm(p=2, dim=0, keepdim=True)
            # normalize the embedding vectors
            # shape: (spa_embed_dim, batch_size)
            return pos_embeds.div(norm)

        return pos_embeds


class PositionEncoder(nn.Module):
    '''
    This is position encoder, a wrapper for different space encoder,
    Given a list of node ids, return their embedding
    '''
    def __init__(self, spa_enc_type, id2geo, spa_enc, graph, spa_enc_embed_norm, device = "cpu"):
        '''
        Args:
            out_dims: a dict()
                key: node type
                value: embedding dimention
            
            spa_enc_type: the type of space encoder
            id2geo: a dict(): node id -> [longitude, latitude]
            spa_enc: one space encoder
            graph: Graph()
            spa_enc_embed_norm: whether to do position embedding normalization
            
            
        '''
        super(PositionEncoder, self).__init__()
        self.spa_enc_type = spa_enc_type
        self.id2geo = id2geo
        self.spa_embed_dim = spa_enc.spa_embed_dim # the output space embedding
        self.spa_enc = spa_enc
        self.graph = graph
        self.spa_enc_embed_norm = spa_enc_embed_norm
        self.device = device

        self.nogeo_idmap = self.make_nogeo_idmap(self.id2geo, self.graph)
        # random initialize the position embedding for nogeo entities
        # last index: indicate the geo-entity, use this for convinience
        self.nogeo_spa_embed_module = torch.nn.Embedding(len(self.nogeo_idmap)+1, self.spa_embed_dim).to(self.device)
        self.add_module("nogeo_pos_embed_matrix", self.nogeo_spa_embed_module)
        # define embedding initialization method: normal dist
        self.nogeo_spa_embed_module.weight.data.normal_(0, 1./self.spa_embed_dim)

    def nogeo_embed_lookup(self, nodes):
        '''
        nogeo_spa_embeds: the spa embed for no-geo entity, [batch_size, spa_embed_dim]
        Note for geo-entity, we use the last embedding in self.nogeo_spa_embed_module
        
        '''
        id_list = []
        for node in nodes:
            if node in self.nogeo_idmap:
                # if this is nogeo entity
                id_list.append(self.nogeo_idmap[node])
            else:
                # if this is geo entity
                id_list.append(len(self.nogeo_idmap))

        # nogeo_spa_embeds: (batch_size, spa_embed_dim)
        nogeo_spa_embeds = self.nogeo_spa_embed_module(
            torch.autograd.Variable(torch.LongTensor(id_list).to(self.device)))
        # calculate the L2-norm for each embedding vector, (batch_size, 1)
        norm = nogeo_spa_embeds.norm(p=2, dim=1, keepdim=True)
        # normalize the embedding vectors
        # shape: [batch_size, spa_embed_dim]
        return nogeo_spa_embeds.div(norm.expand_as(nogeo_spa_embeds))

    def make_nogeo_idmap(self, id2geo, graph):
        '''
        nogeo_idmap: dict(), nogeo-entity id => local id
        '''
        id_set = Set()
        for mode in graph.full_sets:
            id_set.union(graph.full_sets[mode])
        # id_list = sorted(list(id_set))
        geo_set = Set(id2geo.keys())
        nogeo_set = id_set - geo_set
        nogeo_idmap = {nogeo_id: i for i, nogeo_id in enumerate(nogeo_set)}
        return nogeo_idmap

    def forward(self, nodes):
        '''
        Args:
            nodes: a list of node ids
        Return:
            pos_embeds: the position embedding for all nodes, (spa_embed_dim, batch_size)
                    geo_ent => space embedding from geographic coordinates
                    nogeo_ent => [0,0..,0]
        '''
        # coord_tensor: [batch_size, 1, 2], geographic coordicate for geo_ent, [0.0, 0.0] for nogeo_ent
        # nogeo_khot: [batch_size], 0 for geo_ent, 1 for nogeo_ent
        coord_tensor, nogeo_khot = geo_lookup(nodes, self.id2geo, add_dim = 1)

        # spa_embeds: (batch_size, 1, spa_embed_dim)
        spa_embeds = self.spa_enc(coord_tensor)
        # spa_embeds: (batch_size, spa_embed_dim)
        spa_embeds = torch.squeeze(spa_embeds, dim = 1)

        nogeo_khot = torch.FloatTensor(nogeo_khot).to(self.device)
        # mask: (batch_size, 1)
        mask = torch.unsqueeze(nogeo_khot, dim=1)
        # pos_embeds: (batch_size, spa_embed_dim), erase nogeo embed as [0,0...]
        pos_embeds = spa_embeds * (1 - mask)

        # nogeo_spa_embeds: (batch_size, spa_embed_dim)
        nogeo_spa_embeds = self.nogeo_embed_lookup(nodes)
        # nogeo_pos_embeds: (batch_size, spa_embed_dim), erase geo embed as [0,0...]
        nogeo_pos_embeds = nogeo_spa_embeds * mask

        # pos_embeds: (batch_size, spa_embed_dim)
        pos_embeds = pos_embeds + nogeo_pos_embeds

        # pos_embeds: (spa_embed_dim, batch_size)
        pos_embeds = pos_embeds.t()

        if self.spa_enc_embed_norm:
            # calculate the L2-norm for each embedding vector, (1, batch_size)
            norm = pos_embeds.norm(p=2, dim=0, keepdim=True)
            # normalize the embedding vectors
            # shape: (spa_embed_dim, batch_size)
            return pos_embeds.div(norm)

        return pos_embeds


'''
End for space encoding
'''
########################



class NodeEncoder(nn.Module):
    """
    This is the encoder for each entity or node which has two components"
    1. feature encoder (DirectEncoder): feat_enc
    2. position encoder (PositionEncoder): pos_enc
    """
    def __init__(self, feat_enc, pos_enc, agg_type = "add"):
        '''
        Args:
            feat_enc:feature encoder
            pos_enc: position encoder
            agg_type: how to combine the feature embedding and space embedding of a node/entity
        '''
        super(NodeEncoder, self).__init__()
        self.feat_enc = feat_enc
        self.pos_enc = pos_enc
        self.agg_type = agg_type
        if feat_enc is None and pos_enc is None:
            raise Exception("pos_enc and feat_enc are both None!!")

    def forward(self, nodes, mode, offset=None):
        '''
        Args:
            nodes: a list of node ids
        Return:
            
            embeds: node embedding
                if agg_type in ["add", "min", "max", "mean"]:
                    # here we assume spa_embed_dim == embed_dim 
                    shape [embed_dim, num_ent]
                if agg_type == "concat":
                    shape [embed_dim + spa_embed_dim, num_ent]
        '''
        if self.feat_enc is not None and self.pos_enc is not None:
            # we have both feature encoder and position encoder
            
            # feat_embeds: [embed_dim, num_ent]
            feat_embeds = self.feat_enc(nodes, mode, offset=offset)

            # # there is no space encoder
            # if self.pos_enc is None:
            #     return feat_embeds


            # pos_embeds: [embed_dim, num_ent]
            pos_embeds = self.pos_enc(nodes)
            if self.agg_type == "add":
                embeds = feat_embeds + pos_embeds
            elif self.agg_type in ["min", "max", "mean"]:
                
                if self.agg_type == "min":
                    agg_func = torch.min
                elif self.agg_type == "max":
                    agg_func = torch.max
                elif self.agg_type == "mean":
                    agg_func = torch.mean
                combined = torch.stack([feat_embeds, pos_embeds])
                aggs = agg_func(combined, dim=0)
                if type(aggs) == tuple:
                    aggs = aggs[0]
                embeds = aggs
            elif self.agg_type == "concat":
                embeds = torch.cat([feat_embeds, pos_embeds], dim=0)
            else:
                raise Exception("The Node Encoder Aggregation type is not supported!!")
        elif self.feat_enc is None and self.pos_enc is not None:
            # we only have position encoder

            # pos_embeds: [embed_dim, num_ent]
            pos_embeds = self.pos_enc(nodes)

            embeds = pos_embeds
        elif self.feat_enc is not None and self.pos_enc is None:
            # we only have feature encoder

            # feat_embeds: [embed_dim, num_ent]
            feat_embeds = self.feat_enc(nodes, mode, offset=offset)

            embeds = feat_embeds

        return embeds


class NodeAndLocationEncoder(nn.Module):
    """
    This is the encoder for each entity or node which has two components"
    1. feature encoder (DirectEncoder): feat_enc
    2. position encoder (PositionEncoder): pos_enc
    """
    def __init__(self, feat_enc, pos_enc, out_dims, agg_type = "add"):
        '''
        Args:
            feat_enc:feature encoder
            pos_enc: position encoder
            out_dims: a dict() from node type to embed_dim
            agg_type: how to combine the feature embedding and space embedding of a node/entity
        '''
        super(NodeAndLocationEncoder, self).__init__()
        self.feat_enc = feat_enc
        self.pos_enc = pos_enc
        self.out_dims = out_dims
        self.agg_type = agg_type
        if feat_enc is None and pos_enc is None:
            raise Exception("pos_enc and feat_enc are both None!!")

    # def make_zero_feat_embed(self, nodes, mode):
    #     '''
    #     Args:
    #         nodes: a list of node ids
    #         mode: node type
    #     '''
    #     embed_dim = self.out_dims[mode]
    #     # feat_embed = [embed_dim, num_ent]
    #     feat_embed = torch.zeros((embed_dim, len(nodes)))
    #     return feat_embed

    # def make_zero_feat_embed_all_mode(self, num_ent):
    #     '''
    #     Args: 
    #         num_ent: number of entities
    #     '''
    #     mode = list(self.out_dims.keys())[0]
    #     embed_dim = self.out_dims[mode]
    #     # make sure every mode have the same embed_dim
    #     for m in self.out_dims:
    #         assert self.out_dims[m] == embed_dim

    #     # feat_embed = [embed_dim, num_ent]
    #     feat_embed = torch.zeros((embed_dim, num_ent))
    #     return feat_embed

    def encode_location_to_node_embed(self, coord_tensor):
        '''
        Args:
            coord_tensor: [batch_size, 1, 2], geographic coordicate for geo_ent
        '''
        assert self.pos_enc is not None
        # spa_embeds: (batch_size, 1, spa_embed_dim)
        spa_embeds = self.pos_enc.spa_enc(coord_tensor)

        # spa_embeds: (batch_size, spa_embed_dim)
        spa_embeds = torch.squeeze(spa_embeds, dim = 1)

        # pos_embeds: (spa_embed_dim, batch_size)
        pos_embeds = spa_embeds.t()

        # batch_size = pos_embeds.size()[1]
        
        # # feat_embeds: (embed_dim, batch_size)
        # feat_embeds = self.make_zero_feat_embed_all_mode(batch_size)

        # embeds = self.combine_feat_pos_embed(feat_embeds, pos_embeds)

        return pos_embeds

    def combine_feat_pos_embed(self, feat_embeds, pos_embeds):
        '''
        Args:
            feat_embeds: (embed_dim, batch_size)
            pos_embeds: (spa_embed_dim, batch_size)
        '''
        embed_dim = feat_embeds.size()[0]
        spa_embed_dim = pos_embeds.size()[0]

        if self.agg_type == "add":
            assert embed_dim == spa_embed_dim
            # embeds: (embed_dim, batch_size)
            embeds = feat_embeds + pos_embeds
        elif self.agg_type in ["min", "max", "mean"]:
            assert embed_dim == spa_embed_dim
            if self.agg_type == "min":
                agg_func = torch.min
            elif self.agg_type == "max":
                agg_func = torch.max
            elif self.agg_type == "mean":
                agg_func = torch.mean
            combined = torch.stack([feat_embeds, pos_embeds])
            aggs = agg_func(combined, dim=0)
            if type(aggs) == tuple:
                aggs = aggs[0]
            # embeds: (embed_dim, batch_size)
            embeds = aggs
        elif self.agg_type == "concat":
            # embeds: (embed_dim+spa_embed_dim, batch_size)
            embeds = torch.cat([feat_embeds, pos_embeds], dim=0)
        else:
            raise Exception("The Node Encoder Aggregation type is not supported!!")
        return embeds

    def forward(self, nodes, mode, offset=None, enc_pos_embeds_only = False):
        '''
        Args:
            nodes: a list of node ids
            do_zero_feat_embed: whether to make [0,0..,0] vector as the feature embedding
                True: feat_embeds = [0,0....,0]
                False: get the correct feature embedding
        Return:
            
            embeds: node embedding
                if agg_type in ["add", "min", "max", "mean"]:
                    # here we assume spa_embed_dim == embed_dim 
                    shape [embed_dim, num_ent]
                if agg_type == "concat":
                    shape [embed_dim + spa_embed_dim, num_ent]
        '''
        if self.feat_enc is not None and self.pos_enc is not None:
            # we have both feature encoder and position encoder
            # pos_embeds: [embed_dim, num_ent]
            pos_embeds = self.pos_enc(nodes)
            if enc_pos_embeds_only:
                embeds = pos_embeds
            else:
                # feat_embeds: [embed_dim, num_ent]
                feat_embeds = self.feat_enc(nodes, mode, offset=offset)

                embeds = self.combine_feat_pos_embed(feat_embeds, pos_embeds)
        elif self.feat_enc is None and self.pos_enc is not None:
            # we only have position encoder

            # pos_embeds: [embed_dim, num_ent]
            pos_embeds = self.pos_enc(nodes)

            embeds = pos_embeds
        elif self.feat_enc is not None and self.pos_enc is None:
            # we only have feature encoder
            assert  enc_pos_embeds_only == False

            # feat_embeds: [embed_dim, num_ent]
            feat_embeds = self.feat_enc(nodes, mode, offset=offset)

            embeds = feat_embeds

        return embeds





class Encoder(nn.Module):
    """
    Encodes a node's using a GCN/GraphSage approach, 1 layer
    """
    def __init__(self, features, feature_dims, 
            out_dims, relations, adj_lists, aggregator,
            base_model=None, 
            layer_norm=False,
            feature_modules={},
            device = "cpu"): 
        """
        Initializes the model for a specific graph.

        features         -- function mapping (node_list, features, offset) to feature values
                            see torch.nn.EmbeddingBag and forward function below docs for offset meaning.
        feature_dims     -- output dimension of each of the feature functions (for each node type). 
        out_dims         -- embedding dimensions for each mode (i.e., output dimensions)
        relations        -- map from mode -> out_going_relations
        adj_lists        -- map from relation_tuple -> node -> list of node's neighbors
        base_model       -- if features are from another encoder, pass it here for training
        cuda             -- whether or not to move params to the GPU
        feature_modules  -- if features come from torch.nn module, pass the modules here for training

        Args: 
            relations: a dict() of all triple templates
                key:    domain entity type
                value:  a list of tuples (range entity type, predicate)
            adj_lists: a dict about the edges in KG
                key: triple template, i.e. ('drug', 'psoriatic_arthritis', 'drug')
                value: a defaultdict about all the edges instance of thos metapath
                    key: the head entity id
                    value: a set of tail entity ids
            feature_modules: a dict of embedding matrix by node type, each embed matrix shape: [num_ent_by_type + 2, embed_dim]
            aggregator: different aggregator object
            device: cpu or cuda or cuda:0 or cuda:1, whether or not to move params to the GPU

        Return:
            self.compress_params: a dict, mapping node type -> the weight matrix in GraphSAGE aggregator, 
                                shape [out_dims[mode], self.compress_dims[mode]]
            self.compress_dims: a dict, mapping node type -> the second dim of self.compress_params
                                                             (self.relations[mode] + 1) * embed_dim
        """

        super(Encoder, self).__init__()

        self.features = features
        self.feat_dims = feature_dims
        self.adj_lists = adj_lists
        self.relations = relations
        self.aggregator = aggregator
        for name, module in feature_modules.iteritems():
            self.add_module("feat-"+name, module)
        if base_model != None:
            self.base_model = base_model

        self.out_dims = out_dims
        
        self.device = device
        self.aggregator.device = device

        self.layer_norm = layer_norm
        self.compress_dims = {}
        for source_mode in relations:
            self.compress_dims[source_mode] = self.feat_dims[source_mode]
            for (to_mode, _) in relations[source_mode]:
                self.compress_dims[source_mode] += self.feat_dims[to_mode]

        self.self_params = {}
        self.compress_params = {} 
        self.lns = {}
        # for each node type
        for mode, feat_dim in self.feat_dims.iteritems():
            # make a type specific layer normalization layer for convolution layer
            if self.layer_norm:
                self.lns[mode] = LayerNorm(out_dims[mode])
                self.add_module(mode+"_ln", self.lns[mode])
            # make the aggregation weight matrix, See GraphSAGE
            self.compress_params[mode] = nn.Parameter(
                    torch.FloatTensor(out_dims[mode], self.compress_dims[mode]))
            init.xavier_uniform(self.compress_params[mode])
            self.register_parameter(mode+"_compress", self.compress_params[mode])

    def forward(self, nodes, mode, keep_prob=0.5, max_keep=10):
        """
        Generates embeddings for a batch of nodes by using its sampled neghborhoods, GraphSAGE way.
        But aggregation is based on each triple template, in other words, based on each predicate

        nodes     -- list of nodes
        mode      -- string desiginating the mode of the nodes
        """
        self_feat = self.features(nodes, mode).t()
        neigh_feats = []
        # For a neighbor set with each triple template, 
        # use aggregater to compute the aggregated embedding for each node in nodes
        for to_r in self.relations[mode]:
            rel = (mode, to_r[1], to_r[0])
            to_neighs = [[-1] if node == -1 else self.adj_lists[rel][node] for node in nodes]
            
            # Special null neighbor for nodes with no edges of this type
            to_neighs = [[-1] if len(l) == 0 else l for l in to_neighs]
            # to_feats shaps: [len(to_neighs), embed_dim]
            # compute the aggregated embedding for each node in nodes by aggregating from its sampled neighborhoods
            to_feats = self.aggregator.forward(to_neighs, rel, keep_prob, max_keep)
            # to_feats shaps: [embed_dim, len(to_neighs)]
            to_feats = to_feats.t()
            neigh_feats.append(to_feats)
        
        # Append the self embeddings
        neigh_feats.append(self_feat)
        # Concatenate the aggregated embeddings for center nodes (per triple template) and its self nodes
        # See GraphSAGE, combined shape: [(self.relations[mode] + 1)*embed_dim, len(to_neighs)]
        #                             or [self.compress_dims[mode]            , len(to_neighs)]
        combined = torch.cat(neigh_feats, dim=0)
        # self.compress_params[mode] shape: [out_dims[mode], self.compress_dims[mode]]
        # combined shape: [out_dims[mode], len(to_neighs)]
        combined = self.compress_params[mode].mm(combined)
        if self.layer_norm:
            combined = self.lns[mode](combined.t()).t()
        combined = F.relu(combined)
        # combined shape: [out_dims[mode], len(to_neighs)]
        #              or [embed_dim,      num_ent]
        return combined


class LayerNorm(nn.Module):
    """
    layer normalization
    Simple layer norm object optionally used with the convolutional encoder.
    """

    def __init__(self, feature_dim, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones((feature_dim,)))
        self.beta = nn.Parameter(torch.zeros((feature_dim,)))
        self.eps = eps

    def forward(self, x):
        # x: [batch_size, embed_dim]
        # normalize for each embedding
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        # output shape is the same as x
        # Type not match for self.gamma and self.beta??????????????????????
        # output: [batch_size, embed_dim]
        return self.gamma * (x - mean) / (std + self.eps) + self.beta
