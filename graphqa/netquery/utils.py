import numpy as np
import scipy
import scipy.stats as stats
import torch
from sklearn.metrics import roc_auc_score
from netquery.decoders import BilinearMetapathDecoder, TransEMetapathDecoder, BilinearDiagMetapathDecoder, BilinearBlockDiagMetapathDecoder, BilinearBlockDiagPos2FeatMatMetapathDecoder, SetIntersection, SimpleSetIntersection
from netquery.encoders import *
from netquery.aggregators import MeanAggregator
from netquery.attention import IntersectConcatAttention, IntersectDotProductAttention
# from netquery.graph import _reverse_relation
from netquery.module import *
from netquery.SpatialRelationEncoder import *
import cPickle as pickle
import logging
import random
import time
import math

"""
Misc utility functions..
"""
def detect_cuda_device(device):
    if not torch.cuda.is_available():
        device = "cpu"
    else:
        if device == "cpu":
            return device
        elif "cuda" in device:
            if device == "cuda":
                print("Using cuda!!!")
            elif "cuda:" in device:
                cuda_device = int(device.replace("cuda:", ""))
                num_cuda = torch.cuda.device_count()
                if not (cuda_device < num_cuda and cuda_device >= 0):
                    raise Exception("The cuda device number {} is not available!!!".format(device))
                
                device = torch.device(device)

    return device

def cudify(feature_modules, node_maps=None, device = "cuda"):
    '''
    Make the features function with cuda mode
    Args:
        feature_modules: a dict of embedding matrix by node type, each embed matrix shape: [num_ent_by_type + 2, embed_dim]
        node_maps: a dict()
            key: type, 5 types: function, sideeffects, protein, disease, drug
            value: dict():
                key: global node id
                value: local node id for this type
    Return:
        features(nodes, mode): a function to make a dict() from node type to pytorch variable tensor for all (local) node id + 1
            nodes: a lists of global node id which are in type (mode)
            mode: node type
    '''
    if node_maps is None:
        features = lambda nodes, mode : feature_modules[mode](
            torch.autograd.Variable(torch.LongTensor(nodes)+1).to(device))
    else:
        features = lambda nodes, mode : feature_modules[mode](
            torch.autograd.Variable(torch.LongTensor([node_maps[mode][n] for n in nodes])+1).to(device))
    return features

def _get_perc_scores(scores, lengths):
    '''
    percentile rank score: Given a query, one positive target cos score p, x negative target, and their cos score [n1, n2, ..., nx],
    See the rank of p in [n1, n2, ..., nx]

    There are N queries, compute percentiel rank (APR) score for each query
    Args:
        scores: 1st N corespond to cos score for each positive query-target
                scores[N:] correspond to cos score for each negative query-target which append in order, the number is sum(lengths)
        lengths: a list of N int, each indicate the negative sample size for this query
    Return:
        perc_scores: a list of percentile rank score per query, APR are the average of all these score
    '''
    perc_scores = []
    cum_sum = 0
    neg_scores = scores[len(lengths):]
    for i, length in enumerate(lengths):
        # score[i]: the cos score for positive query-target
        # neg_scores[cum_sum:cum_sum+length]: the list of cos score for negative query-target
        perc_scores.append(stats.percentileofscore(neg_scores[cum_sum:cum_sum+length], scores[i]))
        cum_sum += length
    return perc_scores

def entity_embeding_lookup(features, node_list, mode_list):
    # embeds: [batch_size, 1, embed_dim]
    embeds = torch.stack([features([node], mode_list[i]) for i, node in enumerate(node_list)])
    # output: [embed_dim, batch_size]
    return embeds.squeeze(1).t()

def eval_auc_queries(test_queries, enc_dec, batch_size=1000, hard_negatives=False, seed=0):
    '''
    Given a list of queries, run enc_dec, compute AUC score with the negative samples and ground truth labels
    Args:
        test_queries: a dict()
            key: formula template
            value: the query object
    Return:
        formula_aucs: a dict():
            key: (formula.query_type, formula.rels)
            value: AUC for this formula
        overall_auc: overall AUC score for all test queries, overall AUC for all queries for a query type
    '''
    predictions = []
    labels = []
    formula_aucs = {}
    random.seed(seed)
    for formula in test_queries:
        formula_labels = [] # a list of ground truth labels
        formula_predictions = [] # a list of prediction scores
        formula_queries = test_queries[formula]
        offset = 0
        # split the formula_queries intp batches, add collect their ground truth and prediction scores
        while offset < len(formula_queries):
            max_index = min(offset+batch_size, len(formula_queries))
            batch_queries = formula_queries[offset:max_index]
            if hard_negatives:
                # a list for number of negative sample per query
                lengths = [1 for j in range(offset, max_index)]
                negatives = [random.choice(formula_queries[j].hard_neg_samples) for j in xrange(offset, max_index)]
            else:
                lengths = [1 for j in range(offset, max_index)]
                negatives = [random.choice(formula_queries[j].neg_samples) for j  in xrange(offset, max_index)]
            offset += batch_size

            formula_labels.extend([1 for _ in xrange(len(lengths))])
            formula_labels.extend([0 for _ in xrange(len(negatives))])
            batch_scores = enc_dec.forward(formula, 
                    batch_queries+[b for i, b in enumerate(batch_queries) for _ in range(lengths[i])], 
                    [q.target_node for q in batch_queries] + negatives)
            batch_scores = batch_scores.data.tolist()
            formula_predictions.extend(batch_scores)
        formula_key = (formula.query_type, formula.rels)
        formula_aucs[formula_key] = roc_auc_score(formula_labels, np.nan_to_num(formula_predictions))
        labels.extend(formula_labels)
        predictions.extend(formula_predictions)
    overall_auc = roc_auc_score(labels, np.nan_to_num(predictions))
    return overall_auc, formula_aucs

def eval_auc_queries_spa_sem_lift(test_queries, enc_dec, batch_size=1000, hard_negatives=False, seed=0, do_spa_sem_lift = False):
    '''
    Given a list of queries, run enc_dec, compute AUC score with the negative samples and ground truth labels
    Args:
        test_queries: a dict()
            key: formula template
            value: the query object
    Return:
        formula_aucs: a dict():
            key: (formula.query_type, formula.rels)
            value: AUC for this formula
        overall_auc: overall AUC score for all test queries, overall AUC for all queries for a query type
    '''
    predictions = []
    labels = []
    formula_aucs = {}
    random.seed(seed)
    for formula in test_queries:
        formula_labels = [] # a list of ground truth labels
        formula_predictions = [] # a list of prediction scores
        formula_queries = test_queries[formula]
        offset = 0
        # split the formula_queries intp batches, add collect their ground truth and prediction scores
        while offset < len(formula_queries):
            max_index = min(offset+batch_size, len(formula_queries))
            batch_queries = formula_queries[offset:max_index]
            if hard_negatives:
                # a list for number of negative sample per query
                lengths = [1 for j in range(offset, max_index)]
                negatives = [random.choice(formula_queries[j].hard_neg_samples) for j in xrange(offset, max_index)]
            else:
                lengths = [1 for j in range(offset, max_index)]
                negatives = [random.choice(formula_queries[j].neg_samples) for j  in xrange(offset, max_index)]
            offset += batch_size

            formula_labels.extend([1 for _ in xrange(len(lengths))])
            formula_labels.extend([0 for _ in xrange(len(negatives))])
            batch_scores = enc_dec.forward(formula, 
                    batch_queries+[b for i, b in enumerate(batch_queries) for _ in range(lengths[i])], 
                    [q.target_node for q in batch_queries] + negatives,
                    do_spa_sem_lift = do_spa_sem_lift)
            batch_scores = batch_scores.data.tolist()
            formula_predictions.extend(batch_scores)
        formula_key = (formula.query_type, formula.rels)
        formula_aucs[formula_key] = roc_auc_score(formula_labels, np.nan_to_num(formula_predictions))
        labels.extend(formula_labels)
        predictions.extend(formula_predictions)
    overall_auc = roc_auc_score(labels, np.nan_to_num(predictions))
    return overall_auc, formula_aucs

    
def eval_perc_queries(test_queries, enc_dec, batch_size=1000, hard_negatives=False, eval_detail_log = False):
    '''
    Given a list of queries, run enc_dec, compute average percentiel rank (APR) score with the negative samples and ground truth labels
    Args:
        test_queries: a dict()
            key: formula template
            value: the query object
    Return:
        perc_scores: average percentiel rank (APR) score for all test_queries
        the average percentiel rank (APR)

        fm2query_prec: a dict()
            key: (formula.query_type, formula.rels)
            value: a list, each item is [query.serialize(), prec]
                query.serialize(): (query_graph, neg_samples, hard_neg_samples)
                prec: prec score for current query
    '''
    if eval_detail_log:
        fm2query_prec = {}
    perc_scores = []
    for formula in test_queries:
        formula_queries = test_queries[formula]

        if eval_detail_log:
            # save the prec score for each query in each formula
            formula_key = (formula.query_type, formula.rels)
            fm2query_prec[formula_key] = []

        offset = 0
        while offset < len(formula_queries):
            max_index = min(offset+batch_size, len(formula_queries))
            batch_queries = formula_queries[offset:max_index]
            if hard_negatives:
                lengths = [len(formula_queries[j].hard_neg_samples) for j in range(offset, max_index)]
                negatives = [n for j in range(offset, max_index) for n in formula_queries[j].hard_neg_samples]
            else:
                lengths = [len(formula_queries[j].neg_samples) for j in range(offset, max_index)]
                negatives = [n for j in range(offset, max_index) for n in formula_queries[j].neg_samples]
            offset += batch_size

            # the 1st batch_queries is the positive query-target
            # the 2nd               is the negative query-target
            batch_scores = enc_dec.forward(formula, 
                    batch_queries+[b for i, b in enumerate(batch_queries) for _ in range(lengths[i])], 
                    [q.target_node for q in batch_queries] + negatives)
            batch_scores = batch_scores.data.tolist()
            # batch_perc_scores: 
            #       a list of percentile rank score per query, APR are the average of all these score
            batch_perc_scores = _get_perc_scores(batch_scores, lengths)
            perc_scores.extend(batch_perc_scores)

            if eval_detail_log:
                assert len(batch_queries) == len(batch_perc_scores)
                for i, prec in enumerate(batch_perc_scores):
                    query = batch_queries[i]
                    assert query.query_graph is not None
                    q_s = query.serialize()
                    fm2query_prec[formula_key].append([q_s, prec])

    if eval_detail_log:
        return np.mean(perc_scores), fm2query_prec
    else:
        return np.mean(perc_scores)

def eval_perc_queries_spa_sem_lift(test_queries, enc_dec, batch_size=1000, 
            hard_negatives=False, eval_detail_log = False, do_spa_sem_lift = False):
    '''
    Given a list of queries, run enc_dec, compute average percentiel rank (APR) score with the negative samples and ground truth labels
    Args:
        test_queries: a dict()
            key: formula template
            value: the query object
    Return:
        perc_scores: average percentiel rank (APR) score for all test_queries
        the average percentiel rank (APR)

        fm2query_prec: a dict()
            key: (formula.query_type, formula.rels)
            value: a list, each item is [query.serialize(), prec]
                query.serialize(): (query_graph, neg_samples, hard_neg_samples)
                prec: prec score for current query
    '''
    if eval_detail_log:
        fm2query_prec = {}
    perc_scores = []
    for formula in test_queries:
        formula_queries = test_queries[formula]

        if eval_detail_log:
            # save the prec score for each query in each formula
            formula_key = (formula.query_type, formula.rels)
            fm2query_prec[formula_key] = []

        offset = 0
        while offset < len(formula_queries):
            max_index = min(offset+batch_size, len(formula_queries))
            batch_queries = formula_queries[offset:max_index]
            if hard_negatives:
                lengths = [len(formula_queries[j].hard_neg_samples) for j in range(offset, max_index)]
                negatives = [n for j in range(offset, max_index) for n in formula_queries[j].hard_neg_samples]
            else:
                lengths = [len(formula_queries[j].neg_samples) for j in range(offset, max_index)]
                negatives = [n for j in range(offset, max_index) for n in formula_queries[j].neg_samples]
            offset += batch_size

            # the 1st batch_queries is the positive query-target
            # the 2nd               is the negative query-target
            batch_scores = enc_dec.forward(formula, 
                    batch_queries+[b for i, b in enumerate(batch_queries) for _ in range(lengths[i])], 
                    [q.target_node for q in batch_queries] + negatives,
                    do_spa_sem_lift = do_spa_sem_lift)
            batch_scores = batch_scores.data.tolist()
            # batch_perc_scores: 
            #       a list of percentile rank score per query, APR are the average of all these score
            batch_perc_scores = _get_perc_scores(batch_scores, lengths)
            perc_scores.extend(batch_perc_scores)

            if eval_detail_log:
                assert len(batch_queries) == len(batch_perc_scores)
                for i, prec in enumerate(batch_perc_scores):
                    query = batch_queries[i]
                    assert query.query_graph is not None
                    q_s = query.serialize()
                    fm2query_prec[formula_key].append([q_s, prec])

    if eval_detail_log:
        return np.mean(perc_scores), fm2query_prec
    else:
        return np.mean(perc_scores)

def get_pos_encoder(geo_info, 
            spa_enc_type, 
            id2geo, 
            id2extent, 
            spa_enc, 
            graph, 
            spa_enc_embed_norm = True, 
            device = "cpu"):
    if geo_info in ["geo", "proj"]:
        pos_enc = PositionEncoder(spa_enc_type, id2geo, spa_enc, graph, 
                                    spa_enc_embed_norm = spa_enc_embed_norm, device = device)
    elif geo_info in ["projbbox", "projbboxmerge"]:
        pos_enc = ExtentPositionEncoder(spa_enc_type, id2geo, id2extent, spa_enc, graph, 
                                    spa_enc_embed_norm = spa_enc_embed_norm, device = device)
    else:
        raise Exception("Unknown geo_info parameters!")
    return pos_enc

def get_encoder(depth, graph, out_dims, feature_modules, 
            geo_info,
            spa_enc_type = "no", 
            spa_enc_embed_norm = True,
            id2geo = None, 
            id2extent = None,
            spa_enc = None, 
            enc_agg_type = "add", 
            task = "qa",
            device = "cpu"):
    '''
    Construct the GraphSAGE style node embedding encoder
    Args:
        depth: the depth of the graph node embedding encoder, num of GraphSAGE aggregaters
        graph: a Graph() object
        out_dims: a dict() from node type to embed_dim
        feature_modules: a dict of embedding matrix by node type, each embed matrix shape: [num_ent_by_type + 2, embed_dim]
        spa_enc_type: the type of place encoding method
        spa_enc_embed_norm: whether to do position embedding normlization is pos_enc
        spa_enc: the space encoder

        device: cpu or cuda or cuda:0 or cuda:1
    Return:
        enc: a encoder whose forward(nodes, mode) will return node embedding metrix of shape [embed_dim, num_ent]
    '''
    if depth < 0 or depth > 3:
        raise Exception("Depth must be between 0 and 3 (inclusive)")

    if depth == 0:
        if graph.features is not None and feature_modules is not None:
            # 0 layer, directly embedding lookup
            feat_enc = DirectEncoder(graph.features, feature_modules)
        else:
            feat_enc = None

        if spa_enc_type == "no":
            pos_enc = None
        else:
            assert spa_enc is not None
            pos_enc = get_pos_encoder(geo_info = geo_info, 
                                    spa_enc_type = spa_enc_type, 
                                    id2geo = id2geo, 
                                    id2extent = id2extent, 
                                    spa_enc = spa_enc, 
                                    graph = graph, 
                                    spa_enc_embed_norm = spa_enc_embed_norm, 
                                    device = device)
            # pos_enc = PositionEncoder(spa_enc_type, id2geo, spa_enc, graph, 
            #                         spa_enc_embed_norm = spa_enc_embed_norm, device = device)
        if task == "qa":
            enc = NodeEncoder(feat_enc, pos_enc, agg_type = enc_agg_type)
        elif task == "spa_sem_lift":
            enc = NodeAndLocationEncoder(feat_enc, pos_enc, 
                                    out_dims = out_dims, agg_type = enc_agg_type)
        # elif spa_enc_type == "simple":
        #     enc = SimpleSpatialEncoder(graph.features, feature_modules, out_dims, id2geo)
    else:
        if spa_enc_type != "no":
            raise Exception("The place encoding is implemented for depth-0 encoder")
        # 1 GraphSAGE mean aggregator
        aggregator1 = MeanAggregator(graph.features)
        # enc1: a GraphSage Layer, forward() will output [embed_dim, num_ent]
        enc1 = Encoder(graph.features, 
                graph.feature_dims, 
                out_dims, 
                graph.relations, 
                graph.adj_lists, feature_modules=feature_modules, 
                aggregator=aggregator1,
                device = device)
        enc = enc1
        if depth >= 2:
            # 2 GraphSAGE mean aggregator
            aggregator2 = MeanAggregator(lambda nodes, mode : enc1(nodes, mode).t().squeeze())
            enc2 = Encoder(lambda nodes, mode : enc1(nodes, mode).t().squeeze(),
                    enc1.out_dims, 
                    out_dims, 
                    graph.relations, 
                    graph.adj_lists, base_model=enc1,
                    aggregator=aggregator2,
                    device = device)
            enc = enc2
            if depth >= 3:
                # 3 GraphSAGE mean aggregator
                aggregator3 = MeanAggregator(lambda nodes, mode : enc2(nodes, mode).t().squeeze())
                enc3 = Encoder(lambda nodes, mode : enc1(nodes, mode).t().squeeze(),
                        enc2.out_dims, 
                        out_dims, 
                        graph.relations, 
                        graph.adj_lists, base_model=enc2,
                        aggregator=aggregator3,
                        device = device)
                enc = enc3
    return enc

def get_metapath_decoder(graph, out_dims, decoder, feat_dims, spa_embed_dim, enc_agg_type):
    '''
    The metapath decoder just define the geometric project operator
    Args:
        graph: a Graph() object
        out_dims: a dict() mapping node type -> embed_dim
        decoder: a flag for decoder's geometric project operator type
        feat_dims: a dict() mapping node type -> feat embed dim
        enc_agg_type:
    '''
    if decoder == "bilinear":
        dec = BilinearMetapathDecoder(graph.relations, out_dims)
    elif decoder == "transe":
        dec = TransEMetapathDecoder(graph.relations, out_dims)
    elif decoder == "bilinear-diag":
        dec = BilinearDiagMetapathDecoder(graph.relations, out_dims)
    elif decoder == "bilinear_blockdiag":
        assert enc_agg_type == "concat"
        assert feat_dims[list(feat_dims.keys())[0]] > 0 and spa_embed_dim > 0
        dec = BilinearBlockDiagMetapathDecoder(graph.relations, 
                    dims = out_dims, 
                    feat_dims = feat_dims, 
                    spa_embed_dim = spa_embed_dim)
    elif decoder == "blockdiag_p2fmat":
        assert enc_agg_type == "concat"
        assert feat_dims[list(feat_dims.keys())[0]] > 0 and spa_embed_dim > 0
        dec = BilinearBlockDiagPos2FeatMatMetapathDecoder(graph.relations, 
                    dims = out_dims, 
                    feat_dims = feat_dims, 
                    spa_embed_dim = spa_embed_dim)
    else:
        raise Exception("Metapath decoder not recognized.")
    return dec

def get_intersection_decoder(graph, out_dims, decoder, use_relu = True):
    '''
    The intersection decoder define the geometric intersection operator
    Args:
        graph: a Graph() object
        out_dims: a dict() mapping node type -> embed_dim
        decoder: a flag for decoder's geometric intersection operator type
    '''
    if decoder == "mean":
        dec = SetIntersection(out_dims, out_dims, use_relu = use_relu, use_post_mat = True, agg_func=torch.mean)
    elif decoder == "mean_nopostm":
        dec = SetIntersection(out_dims, out_dims, use_relu = use_relu, use_post_mat = False, agg_func=torch.mean)
    elif decoder == "mean_simple":
        dec = SimpleSetIntersection(agg_func=torch.mean)
    elif decoder == "min":
        dec = SetIntersection(out_dims, out_dims, use_relu = use_relu, use_post_mat = True, agg_func=torch.min)
    elif decoder == "min_nopostm":
        dec = SetIntersection(out_dims, out_dims, use_relu = use_relu, use_post_mat = False, agg_func=torch.min)
    elif decoder == "min_simple":
        dec = SimpleSetIntersection(agg_func=torch.min)
    else:
        raise Exception("Intersection decoder not recognized.")
    return dec

def get_intersection_attention(out_dims, inter_decoder_atten_type, inter_decoder_atten_num=0, inter_decoder_atten_act="leakyrelu", inter_decoder_atten_f_act='sigmoid'):
    '''
    The attention mechinism sit on top of intersection operator
    '''
    if inter_decoder_atten_num == 0:
        return None
    else:
        if inter_decoder_atten_type == "concat":
            attn = IntersectConcatAttention(out_dims, out_dims, inter_decoder_atten_num, activation = inter_decoder_atten_act, f_activation = inter_decoder_atten_f_act, layernorm = False, use_post_mat = False)
        elif inter_decoder_atten_type == "concat_norm":
            attn = IntersectConcatAttention(out_dims, out_dims, inter_decoder_atten_num, activation = inter_decoder_atten_act, f_activation = inter_decoder_atten_f_act, layernorm = True, use_post_mat = False)
        elif inter_decoder_atten_type == "concat_postm":
            attn = IntersectConcatAttention(out_dims, out_dims, inter_decoder_atten_num, activation = inter_decoder_atten_act, f_activation = inter_decoder_atten_f_act, layernorm = False, use_post_mat = True)
        elif inter_decoder_atten_type == "concat_norm_postm":
            attn = IntersectConcatAttention(out_dims, out_dims, inter_decoder_atten_num, activation = inter_decoder_atten_act, f_activation = inter_decoder_atten_f_act, layernorm = True, use_post_mat = True)
        elif inter_decoder_atten_type == "dotproduct_scaled":
            attn = IntersectDotProductAttention(out_dims, out_dims, inter_decoder_atten_num, dotproduct_scaled = True, layernorm = False, use_post_mat = False)
        elif inter_decoder_atten_type == "dotproduct":
            attn = IntersectDotProductAttention(out_dims, out_dims, inter_decoder_atten_num, dotproduct_scaled = False, layernorm = False, use_post_mat = False)
        elif inter_decoder_atten_type == "dotproduct_scaled_norm":
            attn = IntersectDotProductAttention(out_dims, out_dims, inter_decoder_atten_num, dotproduct_scaled = True, layernorm = True, use_post_mat = False)
        elif inter_decoder_atten_type == "dotproduct_norm":
            attn = IntersectDotProductAttention(out_dims, out_dims, inter_decoder_atten_num, dotproduct_scaled = False, layernorm = True, use_post_mat = False)
        elif inter_decoder_atten_type == "dotproduct_scaled_postm":
            attn = IntersectDotProductAttention(out_dims, out_dims, inter_decoder_atten_num, dotproduct_scaled = True, layernorm = False, use_post_mat = True)
        elif inter_decoder_atten_type == "dotproduct_postm":
            attn = IntersectDotProductAttention(out_dims, out_dims, inter_decoder_atten_num, dotproduct_scaled = False, layernorm = False, use_post_mat = True)
        elif inter_decoder_atten_type == "dotproduct_scaled_norm_postm":
            attn = IntersectDotProductAttention(out_dims, out_dims, inter_decoder_atten_num, dotproduct_scaled = True, layernorm = True, use_post_mat = True)
        elif inter_decoder_atten_type == "dotproduct_norm_postm":
            attn = IntersectDotProductAttention(out_dims, out_dims, inter_decoder_atten_num, dotproduct_scaled = False, layernorm = True, use_post_mat = True)
        else: 
            raise Exception("intersection attention type not recognized.")

    return attn


def setup_logging(log_file, console=True, filemode='w'):
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        filename=log_file,
                        filemode=filemode)
    if console:
        console = logging.StreamHandler()
        # optional, set the logging level
        console.setLevel(logging.INFO)
        # set a format which is the same for console use
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        # tell the handler to use this format
        console.setFormatter(formatter)
        # add the handler to the root logger
        logging.getLogger('').addHandler(console)
    return logging

def sample_entity_by_metapath(graph, batch_size, neighbor_size, iterator):
    '''
    Args:
        graph: Graph() object
        batch_size: the maximum number of entities for each mini-batch
        neighbor_size: the number of triple templates need to be sampled whose head type is the sampled entity type
        iterator:
    Return:
        mode: a node type
        nodes: a set of node ids with the node type mode
        neg_nodes: a list of node ids as the negative samples
        neighbor_templates: a list of triple templates whose domain type is mode

    '''
    start = time.time()
    # 1. randomly sample a node type
    mode = random.choice(graph.flat_adj_lists.keys())
    nodes = set()
    while len(nodes) == 0:
        # 2. sample K=neighbor_size triple template for the given node type
        templates = graph.relations[mode]
        if len(templates) < neighbor_size:
            neighbor_templates = templates
        else:
            neighbor_templates = random.sample(templates, neighbor_size)
        neighbor_templates = [(mode, to_r[1], to_r[0]) for to_r in neighbor_templates]
        # 3. get all nodes whose satisfy all these sampled triple templates
        nodes_union = set()
        for i, rel in enumerate(neighbor_templates):
            if i == 0:
                nodes = set(graph.adj_lists[rel].keys())
                nodes_union = set(graph.adj_lists[rel].keys())
            else:
                nodes = nodes.intersection(set(graph.adj_lists[rel].keys()))
                nodes_union = nodes_union.union(set(graph.adj_lists[rel].keys()))
            if len(nodes) == 0:
                break

    hard_neg_nodes = list(nodes_union - nodes)
    # 4. get negative nodes
    if len(nodes) > batch_size:
        nodes = set(random.sample(list(nodes), batch_size))
    neg_nodes = list(graph.full_sets[mode] - nodes)
    nodes = list(nodes)
    if len(neg_nodes) > len(nodes):
        neg_nodes = list(np.random.choice(neg_nodes, size=len(nodes), replace=False))
    else:
        neg_nodes = list(np.random.choice(neg_nodes, size=len(nodes), replace=True))
    # 5. random sample tail node for each triple template
    # tail_nodes: [len(neighbor_templates), len(nodes)], ideally [neighbor_size, batch_size]
    tail_nodes = []
    for i, rel in enumerate(neighbor_templates):
        t_nodes = []
        for n in nodes:
            t_nodes.append(random.choice(list(graph.adj_lists[rel][n])))
        tail_nodes.append(t_nodes)

    # 6. FOR NOW, get a fake hard negative sampling
    if len(hard_neg_nodes) > len(nodes):
        hard_neg_nodes = list(np.random.choice(hard_neg_nodes, size=len(nodes), replace=False))
    elif len(hard_neg_nodes) == 0:
        hard_neg_nodes = neg_nodes
    else:
        hard_neg_nodes = list(np.random.choice(hard_neg_nodes, size=len(nodes), replace=True))

    # 7. reverse the relation in neighbor_templates
    neighbor_templates = [graph._reverse_relation(rel) for rel in neighbor_templates]

    assert len(nodes) == len(neg_nodes) == len(hard_neg_nodes)
    assert len(neighbor_templates) == len(tail_nodes)
    
    print("mode: {}".format(mode))
    print(nodes)
    print(neg_nodes)
    print(hard_neg_nodes)
    print(neighbor_templates)
    print(tail_nodes)
    print("The total time: {}".format(time.time()-start))

    return mode, nodes, neg_nodes, hard_neg_nodes, neighbor_templates, tail_nodes


########################
'''
This is for space encoding
'''

def get_ffn(args, input_dim, f_act, context_str = ""):
    # print("Create 3 FeedForward NN!!!!!!!!!!")
    if args.use_layn == "T":
        use_layn = True
    else:
        use_layn = False
    if args.skip_connection == "T":
        skip_connection = True
    else:
        skip_connection = False
#     if args.use_post_mat == "T":
#         use_post_mat = True
#     else:
#         use_post_mat = False
    return MultiLayerFeedForwardNN(
            input_dim=input_dim,
            output_dim=args.spa_embed_dim,
            num_hidden_layers=args.num_hidden_layer,
            dropout_rate=args.dropout,
            hidden_dim=args.hidden_dim,
            activation=f_act,
            use_layernormalize=use_layn,
            skip_connection = skip_connection,
            context_str = context_str)

# def get_spatial_context():
#     extent = (-180, 180, -90, 90)
#     return extent

def get_spatial_context(id2geo, geo_info = "geo", percision = 100):
        '''
        get extent of the input geo-entities
        percision: the number we want to get for the extent, 0 means no change
        '''
        if geo_info == "geo":
            return (-180, 180, -90, 90)
        elif geo_info == "proj" or geo_info == "projbbox" or geo_info == "projbboxmerge":
            iri = list(id2geo.keys())[0]
            
            x_min = id2geo[iri][0]
            x_max = id2geo[iri][0]
            y_min = id2geo[iri][1]
            y_max = id2geo[iri][1]
            
            for iri in id2geo:
                if id2geo[iri][0] < x_min:
                    x_min = id2geo[iri][0]
                    
                if id2geo[iri][0] > x_max:
                    x_max = id2geo[iri][0]
                    
                if id2geo[iri][1] < y_min:
                    y_min = id2geo[iri][1]
                    
                if id2geo[iri][1] > y_max:
                    y_max = id2geo[iri][1]
            
            if percision > 0:
                x_min = math.floor(x_min/percision)*percision
                x_max = math.ceil(x_max/percision)*percision
                y_min = math.floor(y_min/percision)*percision
                y_max = math.ceil(y_max/percision)*percision
            return (x_min, x_max, y_min, y_max)
        else:
            raise Exception("geo_info Unknown!")

def get_spa_encoder(args, geo_info, spa_enc_type, id2geo, spa_embed_dim, coord_dim = 2,
                    anchor_sample_method = "fromid2geo", 
                    num_rbf_anchor_pts = 100, 
                    rbf_kernal_size = 10e2, 
                    frequency_num = 16, 
                    max_radius = 10000, 
                    min_radius = 1, 
                    f_act = "sigmoid", 
                    freq_init = "geometric", 
                    use_postmat = "T",
                    device = "cpu"):
    '''
    Args:
        args: the argparser Object, the attribute we use
            use_layn
            skip_connection
            spa_embed_dim
            num_hidden_layer
            dropout
            hidden_dim
        spa_enc_type: the type of space encoder
        id2geo: a dict(): node id -> [longitude, latitude]
        spa_embed_dim: the output space embedding
        coord_dim:
        
    '''

    if args.use_layn == "T":
        use_layn = True
    else:
        use_layn = False
    
    if use_postmat == "T":
        use_post_mat = True
    else:
        use_post_mat = False
    if spa_enc_type == "gridcell":
        ffn = get_ffn(args,
            input_dim=int(4 * frequency_num),
            f_act = f_act,
            context_str = "GridCellSpatialRelationEncoder")
        spa_enc = GridCellSpatialRelationEncoder(
            spa_embed_dim, 
            coord_dim = coord_dim, 
            frequency_num = frequency_num, 
            max_radius = max_radius,
            min_radius = min_radius,
            freq_init = freq_init,
            ffn=ffn,
            device=device)
    elif spa_enc_type == "gridcellnonorm":
        ffn = get_ffn(args,
            input_dim=int(4 * frequency_num),
            f_act = f_act,
            context_str = "GridNoNormCellSpatialRelationEncoder")
        spa_enc = GridNoNormCellSpatialRelationEncoder(
            spa_embed_dim, 
            coord_dim = coord_dim, 
            frequency_num = frequency_num, 
            max_radius = max_radius,
            min_radius = min_radius,
            freq_init = freq_init,
            ffn=ffn,
            device=device)
    elif spa_enc_type == "hexagridcell":
        spa_enc = HexagonGridCellSpatialRelationEncoder(
            spa_embed_dim, 
            coord_dim = coord_dim, 
            frequency_num = frequency_num, 
            max_radius = max_radius,
            dropout = args.dropout, 
            f_act= f_act,
            device=device)
    elif spa_enc_type == "theory":
        ffn = get_ffn(args,
            input_dim=int(6 * frequency_num),
            f_act = f_act,
            context_str = "TheoryGridCellSpatialRelationEncoder")
        spa_enc = TheoryGridCellSpatialRelationEncoder(
            spa_embed_dim,
            coord_dim = coord_dim,
            frequency_num = frequency_num,
            max_radius = max_radius,
            min_radius = min_radius,
            freq_init = freq_init,
            ffn=ffn,
            device=device)
    elif spa_enc_type == "theorydiag":
        spa_enc = TheoryDiagGridCellSpatialRelationEncoder(
            spa_embed_dim, 
            coord_dim = coord_dim, 
            frequency_num = frequency_num, 
            max_radius = max_radius, 
            min_radius = min_radius,
            dropout = args.dropout, 
            f_act= f_act, 
            freq_init = freq_init, 
            use_layn = use_layn, 
            use_post_mat = use_post_mat,
            device=device)
    elif spa_enc_type == "naive":
        extent = get_spatial_context(id2geo, geo_info = geo_info)
        ffn = get_ffn(args,
            input_dim=2,
            f_act = f_act,
            context_str = "NaiveSpatialRelationEncoder")
        spa_enc = NaiveSpatialRelationEncoder(
            spa_embed_dim, 
            extent = extent, 
            coord_dim = coord_dim, 
            ffn = ffn,
            device=device)
    # elif spa_enc_type == "polar":
    #     ffn = get_ffn(args,
    #         input_dim=2,
    #         f_act = f_act,
    #         context_str = "PolarCoordSpatialRelationEncoder")
    #     spa_enc = PolarCoordSpatialRelationEncoder(spa_embed_dim, coord_dim = coord_dim, ffn = ffn)
    # elif spa_enc_type == "polardist":
    #     ffn = get_ffn(args,
    #         input_dim=1,
    #         f_act = f_act,
    #         context_str = "PolarDistCoordSpatialRelationEncoder")
    #     spa_enc = PolarDistCoordSpatialRelationEncoder(spa_embed_dim, coord_dim = coord_dim, ffn = ffn)
    # elif spa_enc_type == "polargrid":
    #     ffn = get_ffn(args,
    #         input_dim=int(2 * frequency_num),
    #         f_act = f_act,
    #         context_str = "PolarGridCoordSpatialRelationEncoder")
    #     spa_enc = PolarGridCoordSpatialRelationEncoder(
    #         spa_embed_dim, 
    #         coord_dim = coord_dim, 
    #         frequency_num = frequency_num,
    #         max_radius = max_radius,
    #         min_radius = min_radius,
    #         freq_init = freq_init,
    #         ffn=ffn)
    elif spa_enc_type == "rbf":
        extent = get_spatial_context(id2geo, geo_info = geo_info)
        ffn = get_ffn(args,
            input_dim=num_rbf_anchor_pts,
            f_act = f_act,
            context_str = "RBFSpatialRelationEncoder")
        spa_enc = RBFSpatialRelationEncoder(
            id2geo = id2geo,
            spa_embed_dim = spa_embed_dim,
            coord_dim = coord_dim, 
            anchor_sample_method = anchor_sample_method,
            num_rbf_anchor_pts = num_rbf_anchor_pts,
            rbf_kernal_size = rbf_kernal_size,
            rbf_kernal_size_ratio = 0,  # we just use 0, because this is only used for global pos enc
            extent = extent,
            ffn=ffn,
            device=device)
    # elif spa_enc_type == "distrbf":
    #     spa_enc = DistRBFSpatialRelationEncoder(
    #         spa_embed_dim, coord_dim = coord_dim,
    #         num_rbf_anchor_pts = num_rbf_anchor_pts, rbf_kernal_size = rbf_kernal_size, max_radius = max_radius,
    #         dropout = dropout, f_act = f_act)
    elif spa_enc_type == "gridlookup":
        ffn = get_ffn(args,
            input_dim=spa_embed_dim,
            f_act = f_act,
            context_str = "GridLookupSpatialRelationEncoder")

        extent = get_spatial_context(id2geo, geo_info = geo_info)
        
        spa_enc = GridLookupSpatialRelationEncoder(
            spa_embed_dim, 
            coord_dim = coord_dim, 
            interval = min_radius, 
            extent = extent, 
            ffn = ffn,
            device=device)
    elif spa_enc_type == "gridlookupnoffn":
        extent = get_spatial_context(id2geo, geo_info = geo_info)

        spa_enc = GridLookupSpatialRelationEncoder(
            spa_embed_dim, 
            coord_dim = coord_dim, 
            interval = min_radius, 
            extent = extent, 
            ffn = None,
            device=device)
    # elif spa_enc_type == "polargridlookup":
    #     assert model_type == "relative"
    #     ffn = get_ffn(args,
    #         input_dim=args.spa_embed_dim,
    #         f_act = f_act,
    #         context_str = "PolarGridLookupSpatialRelationEncoder")
    #     spa_enc = PolarGridLookupSpatialRelationEncoder(
    #         spa_embed_dim, 
    #         coord_dim = coord_dim, 
    #         max_radius = max_radius, 
    #         frequency_num = frequency_num, 
    #         ffn = ffn)
    elif spa_enc_type == "aodha":
        extent = get_spatial_context(id2geo, geo_info = geo_info)
        spa_enc = AodhaSpatialRelationEncoder(
            spa_embed_dim, 
            extent = extent, 
            coord_dim = coord_dim,
            num_hidden_layers = args.num_hidden_layer,
            hidden_dim = args.hidden_dim,
            use_post_mat=use_post_mat,
            f_act=f_act,
            device=device)
    elif spa_enc_type == "none":
        assert spa_embed_dim == 0
        spa_enc = None
    else:
        raise Exception("Space encoder function no support!")
    return spa_enc