import torch
import torch.nn as nn 
import numpy as np
import re

import random
# from netquery.graph import _reverse_relation

# from utils import entity_embeding_lookup

EPS = 10e-6

"""
End-to-end autoencoder models for representation learning on
heteregenous graphs/networks
"""

class MetapathEncoderDecoder(nn.Module):
    """
    Encoder decoder model that reasons over metapaths
    """

    def __init__(self, graph, enc, dec):
        """
        graph -- simple graph object; see graph.py
        enc --- an encoder module that generates embeddings (see encoders.py) 
        dec --- an decoder module that predicts compositional relationships, i.e. metapaths, between nodes given embeddings. (see decoders.py)
                Note that the decoder must be an *compositional/metapath* decoder (i.e., with name Metapath*.py)
        """
        super(MetapathEncoderDecoder, self).__init__()
        self.enc = enc
        self.dec = dec
        self.graph = graph

    def forward(self, nodes1, nodes2, rels):
        """
        Returns a vector of 'relationship scores' for pairs of nodes being connected by the given metapath (sequence of relations).
        Essentially, the returned scores are the predicted likelihood of the node pairs being connected
        by the given metapath, where the pairs are given by the ordering in nodes1 and nodes2,
        i.e. the first node id in nodes1 is paired with the first node id in nodes2.
        """

        # output shape: [num_ent]
        return self.dec.forward(self.enc.forward(nodes1, rels[0][0]), 
                self.enc.forward(nodes2, rels[-1][-1]),
                rels)

    def margin_loss(self, nodes1, nodes2, rels):
        """
        Standard max-margin based loss function.
        Maximizes relationaship scores for true pairs vs negative samples.
        """
        affs = self.forward(nodes1, nodes2, rels)
        # there is some problem wil this negative sampling method ?????????????????????????
        # neg_nodes = [random.randint(1,len(self.graph.adj_lists[self.graph._reverse_relation[rels[-1]]])-1) for _ in xrange(len(nodes1))]
        neg_nodes = [random.randint(1,len(self.graph.adj_lists[self.graph._reverse_relation(rels[-1])])-1) for _ in xrange(len(nodes1))]
        neg_affs = self.forward(nodes1, neg_nodes,
            rels)
        margin = 1 - (affs - neg_affs)
        margin = torch.clamp(margin, min=0)
        loss = margin.mean()
        return loss 

class QueryEncoderDecoder(nn.Module):
    """
    Encoder decoder model that reasons about edges, metapaths and intersections
    """

    def __init__(self, graph, enc, path_dec, inter_dec, inter_attn, use_inter_node=False):
        '''
        Args:
            graph: the Graph() object
            enc: the node embedding encoder object
            path_dec: the metapath dencoder object
            inter_dec: the intersection decoder object
            use_inter_node: Whether we use the True nodes in the intersection attention as the query embedding to train the QueryEncoderDecoder

        '''
        super(QueryEncoderDecoder, self).__init__()
        self.enc = enc
        self.path_dec = path_dec
        self.inter_dec = inter_dec
        self.inter_attn = inter_attn
        self.graph = graph
        self.cos = nn.CosineSimilarity(dim=0)
        self.use_inter_node = use_inter_node

    def forward(self, formula, queries, source_nodes, do_modelTraining = False):
        '''
        Args:
            formula: a Fomula() object
            queries: a list of Query() objects with the same formula
            source_nodes: a list of target node for each query (Training), a list of negative sampling nodes (query inferencing)
            do_modelTraining: default is False, we do query inferencing
        return:
            scores: a list of cosine scores with length len(queries)
        '''
        pattern = re.compile("[\d]+-inter$")
        if formula.query_type == "1-chain" or formula.query_type == "2-chain" or formula.query_type == "3-chain":
            # a chain is simply a call to the path decoder
            # If they do this way, then the path decoder begin from the target node, then translate to the anchor node????????????
            # return self.path_dec.forward(
            #         self.enc.forward(source_nodes, formula.target_mode), 
            #         self.enc.forward([query.anchor_nodes[0] for query in queries], formula.anchor_modes[0]),
            #         formula.rels)

            # I think this is the right way to do x-chain
            reverse_rels = tuple([self.graph._reverse_relation(formula.rels[i]) for i in range(len(formula.rels)-1, -1, -1)])
            return self.path_dec.forward(
                    self.enc.forward([query.anchor_nodes[0] for query in queries], formula.anchor_modes[0]),
                    self.enc.forward(source_nodes, formula.target_mode), 
                    reverse_rels)
        # elif formula.query_type == "2-inter" or formula.query_type == "3-inter" or formula.query_type == "3-inter_chain":
        #     target_embeds = self.enc(source_nodes, formula.target_mode)

        #     # project the 1st anchor node to target node in rels: (t, p1, a1)
        #     embeds1 = self.enc([query.anchor_nodes[0] for query in queries], formula.anchor_modes[0])
        #     embeds1 = self.path_dec.project(embeds1, self.graph._reverse_relation(formula.rels[0]))

        #     # project the 2nd anchor node to target node in rels: 
        #     embeds2 = self.enc([query.anchor_nodes[1] for query in queries], formula.anchor_modes[1])
        #     if len(formula.rels[1]) == 2: # '3-inter_chain'
        #         # '3-inter_chain': project a2 to t by following ((t, p2, e1),(e1, p3, a2))
        #         for i_rel in formula.rels[1][::-1]: # loop the formula.rels[1] in the reverse order
        #             embeds2 = self.path_dec.project(embeds2, self.graph._reverse_relation(i_rel))
        #     else: # "2-inter" or "3-inter"
        #         # project a2 to t by following (t, p2, a2)
        #         embeds2 = self.path_dec.project(embeds2, self.graph._reverse_relation(formula.rels[1]))

        #     # project the 3nd anchor node to target node in rels: 
        #     # "3-inter": project a3 to t by following (t, p3, a3)
        #     if formula.query_type == "3-inter":
        #         embeds3 = self.enc([query.anchor_nodes[2] for query in queries], formula.anchor_modes[2])
        #         embeds3 = self.path_dec.project(embeds3, self.graph._reverse_relation(formula.rels[2]))

        #         # intersect different target node embeddings
        #         # query_intersection, embeds_inter = self.inter_dec(embeds1, embeds2, formula.target_mode, embeds3)
        #         query_intersection, embeds_inter = self.inter_dec(formula.target_mode, [embeds1, embeds2, embeds3])
        #     else:
        #         query_intersection, embeds_inter = self.inter_dec(formula.target_mode, [embeds1, embeds2])
            
        #     # this is where the attention take affect
        #     if self.inter_attn is not None:
        #         if self.use_inter_node and do_modelTraining:
        #             # for 2-inter, 3-inter, 3-inter_chain, the inter node is target node
        #             # we we can use source_nodes
        #             query_embeds = self.graph.features([query.target_node for query in queries], formula.target_mode).t()
        #             query_intersection = self.inter_attn(query_embeds, embeds_inter, formula.target_mode)
        #         else:
        #             query_intersection = self.inter_attn(query_intersection, embeds_inter, formula.target_mode)


        #     scores = self.cos(target_embeds, query_intersection)
        #     return scores
        elif formula.query_type == "3-inter_chain":
            target_embeds = self.enc(source_nodes, formula.target_mode)

            # project the 1st anchor node to target node in rels: (t, p1, a1)
            embeds1 = self.enc([query.anchor_nodes[0] for query in queries], formula.anchor_modes[0])
            embeds1 = self.path_dec.project(embeds1, self.graph._reverse_relation(formula.rels[0]))

            # project the 2nd anchor node to target node in rels: 
            embeds2 = self.enc([query.anchor_nodes[1] for query in queries], formula.anchor_modes[1])
            # '3-inter_chain': project a2 to t by following ((t, p2, e1),(e1, p3, a2))
            for i_rel in formula.rels[1][::-1]: # loop the formula.rels[1] in the reverse order
                embeds2 = self.path_dec.project(embeds2, self.graph._reverse_relation(i_rel))
   
            query_intersection, embeds_inter = self.inter_dec(formula.target_mode, [embeds1, embeds2])
            
            # this is where the attention take affect
            if self.inter_attn is not None:
                if self.use_inter_node and do_modelTraining:
                    # for 2-inter, 3-inter, 3-inter_chain, the inter node is target node
                    # we we can use source_nodes
                    query_embeds = self.graph.features([query.target_node for query in queries], formula.target_mode).t()
                    query_intersection = self.inter_attn(query_embeds, embeds_inter, formula.target_mode)
                else:
                    query_intersection = self.inter_attn(query_intersection, embeds_inter, formula.target_mode)


            scores = self.cos(target_embeds, query_intersection)
            return scores
        elif formula.query_type == "3-chain_inter":
            target_embeds = self.enc(source_nodes, formula.target_mode)

            # project the 1st anchor node to inter node (e1, p2, a1)
            embeds1 = self.enc([query.anchor_nodes[0] for query in queries], formula.anchor_modes[0])
            embeds1 = self.path_dec.project(embeds1, self.graph._reverse_relation(formula.rels[1][0]))
            # project the 2nd anchor node to inter node (e1, p3, a2)
            embeds2 = self.enc([query.anchor_nodes[1] for query in queries], formula.anchor_modes[1])
            embeds2 = self.path_dec.project(embeds2, self.graph._reverse_relation(formula.rels[1][1]))
            # intersect different inter node (e1) embeddings
            # query_intersection, embeds_inter = self.inter_dec(embeds1, embeds2, formula.rels[0][-1])
            query_intersection, embeds_inter = self.inter_dec(formula.rels[0][-1], [embeds1, embeds2])

            # this is where the attention take affect
            if self.inter_attn is not None:
                if self.use_inter_node and do_modelTraining:
                    # for 3-chain_inter, the inter node is in the query_graph
                    inter_nodes = [query.query_graph[1][2] for query in queries]
                    inter_node_mode = formula.rels[0][2]
                    query_embeds = self.graph.features(inter_nodes, inter_node_mode).t()
                    query_intersection = self.inter_attn(query_embeds, embeds_inter, formula.target_mode)
                else:
                    query_intersection = self.inter_attn(query_intersection, embeds_inter, formula.rels[0][-1])

            query_intersection = self.path_dec.project(query_intersection, self.graph._reverse_relation(formula.rels[0]))
            scores = self.cos(target_embeds, query_intersection)
            return scores
        elif pattern.match(formula.query_type) is not None:
            # x-inter
            target_embeds = self.enc(source_nodes, formula.target_mode)
            num_edges = int(formula.query_type.replace("-inter", ""))
            embeds_list = []
            for i in range(0, num_edges):
                # project the ith anchor node to target node in rels: (t, pi, ai)
                embeds = self.enc([query.anchor_nodes[i] for query in queries], formula.anchor_modes[i])
                embeds = self.path_dec.project(embeds, self.graph._reverse_relation(formula.rels[i]))
                embeds_list.append(embeds)

            query_intersection, embeds_inter = self.inter_dec(formula.target_mode, embeds_list)

            # this is where the attention take affect
            if self.inter_attn is not None:
                if self.use_inter_node and do_modelTraining:
                    # for x-inter, the inter node is target node
                    # so we can use the real target node
                    query_embeds = self.graph.features([query.target_node for query in queries], formula.target_mode).t()
                    query_intersection = self.inter_attn(query_embeds, embeds_inter, formula.target_mode)
                else:
                    query_intersection = self.inter_attn(query_intersection, embeds_inter, formula.target_mode)


            scores = self.cos(target_embeds, query_intersection)
            return scores




    def margin_loss(self, formula, queries, hard_negatives=False, margin=1):
        if not "inter" in formula.query_type and hard_negatives:
            raise Exception("Hard negative examples can only be used with intersection queries")
        elif hard_negatives:
            neg_nodes = [random.choice(query.hard_neg_samples) for query in queries]
        elif formula.query_type == "1-chain":
            neg_nodes = [random.choice(self.graph.full_lists[formula.target_mode]) for _ in queries]
        else:
            neg_nodes = [random.choice(query.neg_samples) for query in queries]

        affs = self.forward(formula, queries, [query.target_node for query in queries], do_modelTraining=True)
        neg_affs = self.forward(formula, queries, neg_nodes, do_modelTraining=True)
        # affs (neg_affs) is the cosine similarity between golden (negative) node embedding and predicted embedding
        # the larger affs (the smaller the neg_affs), the better the prediction is
        loss = margin - (affs - neg_affs)
        loss = torch.clamp(loss, min=0)
        loss = loss.mean()
        return loss 

class SpatialSemanticLiftingEncoderDecoder(nn.Module):
    """
    Encoder decoder model that reasons about edges, metapaths and intersections
    """

    def __init__(self, graph, enc, path_dec, inter_dec, inter_attn, use_inter_node=False):
        '''
        Args:
            graph: the Graph() object
            enc: the node embedding encoder object
            path_dec: the metapath dencoder object
            inter_dec: the intersection decoder object
            use_inter_node: Whether we use the True nodes in the intersection attention as the query embedding to train the QueryEncoderDecoder

        '''
        super(SpatialSemanticLiftingEncoderDecoder, self).__init__()
        self.enc = enc
        self.path_dec = path_dec
        self.inter_dec = inter_dec
        self.inter_attn = inter_attn
        self.graph = graph
        self.cos = nn.CosineSimilarity(dim=0)
        self.use_inter_node = use_inter_node

    def forward(self, formula, queries, source_nodes, do_spa_sem_lift = False, do_modelTraining = False):
        '''
        Args:
            formula: a Fomula() object
            queries: a list of Query() objects with the same formula
            source_nodes: a list of target node for each query (Training), a list of negative sampling nodes (query inferencing)
            do_spa_sem_lift: whether to do spatial semantic lifting
            do_modelTraining: default is False, we do query inferencing
        return:
            scores: a list of cosine scores with length len(queries)
        '''
        
        if formula.query_type == "1-chain":
            # a chain is simply a call to the path decoder
            # If they do this way, then the path decoder begin from the target node, then translate to the anchor node????????????
            # return self.path_dec.forward(
            #         self.enc.forward(source_nodes, formula.target_mode), 
            #         self.enc.forward([query.anchor_nodes[0] for query in queries], formula.anchor_modes[0]),
            #         formula.rels)

            # I think this is the right way to do x-chain
            reverse_rels = tuple([self.graph._reverse_relation(formula.rels[i]) for i in range(len(formula.rels)-1, -1, -1)])
            head_embeds = self.enc.forward([query.anchor_nodes[0] for query in queries], 
                                formula.anchor_modes[0], 
                                enc_pos_embeds_only = do_spa_sem_lift)
            target_embeds = self.enc.forward(source_nodes, 
                                formula.target_mode, 
                                enc_pos_embeds_only = False)
            
            if self.path_dec.__class__.__name__ == "BilinearBlockDiagPos2FeatMatMetapathDecoder":
                return self.path_dec.forward(
                        head_embeds,
                        target_embeds, 
                        reverse_rels,
                        do_spa_sem_lift = do_spa_sem_lift)
            else:
                return self.path_dec.forward(
                        head_embeds,
                        target_embeds, 
                        reverse_rels)
        
        


    def margin_loss(self, formula, queries, hard_negatives=False, margin=1, do_spa_sem_lift = False):
        if not "inter" in formula.query_type and hard_negatives:
            raise Exception("Hard negative examples can only be used with intersection queries")
        elif hard_negatives:
            neg_nodes = [random.choice(query.hard_neg_samples) for query in queries]
        elif formula.query_type == "1-chain":
            neg_nodes = [random.choice(self.graph.full_lists[formula.target_mode]) for _ in queries]
        else:
            neg_nodes = [random.choice(query.neg_samples) for query in queries]

        affs = self.forward(formula, queries, [query.target_node for query in queries], do_spa_sem_lift = do_spa_sem_lift, do_modelTraining=True)
        neg_affs = self.forward(formula, queries, neg_nodes, do_spa_sem_lift = do_spa_sem_lift, do_modelTraining=True)
        # affs (neg_affs) is the cosine similarity between golden (negative) node embedding and predicted embedding
        # the larger affs (the smaller the neg_affs), the better the prediction is
        loss = margin - (affs - neg_affs)
        loss = torch.clamp(loss, min=0)
        loss = loss.mean()
        return loss 

class SoftAndEncoderDecoder(nn.Module):
    """
    Encoder decoder model that reasons about edges, metapaths and intersections

    Different from QueryEncoderDecoder(), this does not use intersect operator
    Given a intersection, just compute cos between gold sample and the prediction, and multiply them
    So this does not support 3-chain_inter !!!!!!!!!!!
    """

    def __init__(self, graph, enc, path_dec):
        super(SoftAndEncoderDecoder, self).__init__()
        self.enc = enc
        self.path_dec = path_dec
        self.graph = graph
        self.cos = nn.CosineSimilarity(dim=0)

    def forward(self, formula, queries, source_nodes):
        if formula.query_type == "1-chain":
            # a chain is simply a call to the path decoder
            return self.path_dec.forward(
                    self.enc.forward(source_nodes, formula.target_mode), 
                    self.enc.forward([query.anchor_nodes[0] for query in queries], formula.anchor_modes[0]),
                    formula.rels)
        elif formula.query_type == "2-inter" or formula.query_type == "3-inter":
            target_embeds = self.enc(source_nodes, formula.target_mode)

            embeds1 = self.enc([query.anchor_nodes[0] for query in queries], formula.anchor_modes[0])
            embeds1 = self.path_dec.project(embeds1, self.graph._reverse_relation(formula.rels[0]))

            embeds2 = self.enc([query.anchor_nodes[1] for query in queries], formula.anchor_modes[1])
            if len(formula.rels[1]) == 2:
                for i_rel in formula.rels[1][::-1]:
                    embeds2 = self.path_dec.project(embeds2, self.graph._reverse_relation(i_rel))
            else:
                embeds2 = self.path_dec.project(embeds2, self.graph._reverse_relation(formula.rels[1]))

            scores1 = self.cos(target_embeds, embeds1)
            scores2 = self.cos(target_embeds, embeds2)
            if formula.query_type == "3-inter":
                embeds3 = self.enc([query.anchor_nodes[2] for query in queries], formula.anchor_modes[2])
                embeds3 = self.path_dec.project(embeds3, self.graph._reverse_relation(formula.rels[2]))
                scores3 = self.cos(target_embeds, embeds2)
                scores = scores1 * scores2 * scores3
            else:
                scores = scores1 * scores2
            return scores
        else:
            raise Exception("Query type not supported for this model.")

    def margin_loss(self, formula, queries, hard_negatives=False, margin=1):
        if not "inter" in formula.query_type and hard_negatives:
            raise Exception("Hard negative examples can only be used with intersection queries")
        elif hard_negatives:
            neg_nodes = [random.choice(query.hard_neg_samples) for query in queries]
        elif formula.query_type == "1-chain":
            neg_nodes = [random.choice(self.graph.full_lists[formula.target_mode]) for _ in queries]
        else:
            neg_nodes = [random.choice(query.neg_samples) for query in queries]

        affs = self.forward(formula, queries, [query.target_node for query in queries])
        neg_affs = self.forward(formula, queries, neg_nodes)
        loss = margin - (affs - neg_affs)
        loss = torch.clamp(loss, min=0)
        loss = loss.mean()
        return loss 
