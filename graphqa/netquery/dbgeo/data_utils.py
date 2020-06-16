import cPickle as pickle
import torch
from collections import OrderedDict, defaultdict
from multiprocessing import Process
import random
import json
from netquery.data_utils import parallel_sample, load_queries_by_type, sample_clean_test, parallel_inter_query_sample

from netquery.graph import Graph, Query#, _reverse_edge

def load_graph(data_dir, embed_dim, graph_data_path = "/graph_data.pkl"):
    '''
    Given embed_dim, load graph data from file and construc Graph() object

    Return:
        graph: a Graph() object
        feature_modules: a dict of embedding matrix by node type, each embed matrix shape: [num_ent_by_type + 2, embed_dim]
        node_maps: a dict()
            key: type, 5 types: function, sideeffects, protein, disease, drug
            value: dict():
                key: global node id
                value: local node id for this type
    '''

    '''
    rels: a dict() of all triple templates
        key:    domain entity type
        value:  a list of tuples (range entity type, predicate)
    adj_lists: a dict about the edges in KG
        key: triple template, i.e. ('drug', 'psoriatic_arthritis', 'drug')
        value: a defaultdict about all the edges instance of thos metapath
            key: the head entity id
            value: a set of tail entity ids
    node_maps: a dict () about node types
        key: type, 5 types: function, sideeffects, protein, disease, drug
        value: a list of node id
    '''
    rels, adj_lists, node_maps, rid2inverse = pickle.load(open(data_dir+graph_data_path, "rb"))
    node_maps = {m : {n : i for i, n in enumerate(id_list)} for m, id_list in node_maps.iteritems()}
    '''
    node_maps: a dict()
        key: type, 5 types: function, sideeffects, protein, disease, drug
        value: dict():
            key: global node id
            value: local node id for this type
    '''
    for m in node_maps:
        node_maps[m][-1] = -1
    feature_dims = {m : embed_dim for m in rels}
    if embed_dim > 0:
        # initialze embedding matrix for each node type [num_ent_by_type + 2, embed_dim]
        feature_modules = {m : torch.nn.Embedding(len(node_maps[m])+1, embed_dim) for m in rels}
        for mode in rels:
            # define embedding initialization method: normal dist
            feature_modules[mode].weight.data.normal_(0, 1./embed_dim)
        '''
        features(nodes, mode): a embedding lookup function to make a dict() from node type to embeddingbag
            nodes: a lists of global node id which are in type (mode)
            mode: node type
            return: embedding vectors, shape [num_node, embed_dim]
        '''
        features = lambda nodes, mode : feature_modules[mode](
                torch.autograd.Variable(torch.LongTensor([node_maps[mode][n] for n in nodes])+1))
    else:
        feature_modules = None
        features = None
    graph = Graph(features, feature_dims, rels, adj_lists, rid2inverse = rid2inverse)
    return graph, feature_modules, node_maps

def sample_new_clean(data_dir, id2geo = None):
    graph_loader = lambda : load_graph(data_dir, 10)[0]
    sample_clean_test(graph_loader, data_dir, id2geo = id2geo)

def json_load(json_filepath):
    with open(json_filepath, "r") as f:
            json_obj = json.load(f)

    return json_obj


def clean_test():
    '''
    Check testing/validation 2/3 edge split data, make sure each query have one edge which are in test/validate edge set
    Then make 1000/10000 for validate/testing queries per query type
    '''
    test_edges = pickle.load(open("/dfs/scratch0/nqe-bio/test_edges.pkl", "rb"))
    val_edges = pickle.load(open("/dfs/scratch0/nqe-bio/val_edges.pkl", "rb"))  
    deleted_edges = set([q[0][1] for q in test_edges] + [_reverse_edge(q[0][1]) for q in test_edges] + 
                [q[0][1] for q in val_edges] + [_reverse_edge(q[0][1]) for q in val_edges])

    for i in range(2,4):
        for kind in ["val", "test"]:
            if kind == "val":
                to_keep = 1000
            else:
                to_keep = 10000
            test_queries = load_queries_by_type("/dfs/scratch0/nqe-bio/{:s}_queries_{:d}-split.pkl".format(kind, i), keep_graph=True)
            print "Loaded", i, kind
            for query_type in test_queries:
                test_queries[query_type] = [q for q in test_queries[query_type] if len(q.get_edges().intersection(deleted_edges)) > 0]
                test_queries[query_type] = test_queries[query_type][:to_keep]
            test_queries = [q.serialize() for queries in test_queries.values() for q in queries]
            pickle.dump(test_queries, open("/dfs/scratch0/nqe-bio/{:s}_queries_{:d}-clean.pkl".format(kind, i), "wb"), protocol=pickle.HIGHEST_PROTOCOL)
            print "Finished", i, kind
        

def pickle_load(pickle_filepath):
    with open(pickle_filepath, "rb") as f:
        obj = pickle.load(f)
    return obj

def make_valid_test_edge_from_triple(graph, edge, neg_sample_size):
    neg_samples = graph.get_negative_edge_samples(edge, neg_sample_size)
    return Query(("1-chain", edge), neg_samples, None, neg_sample_size, keep_graph=True)

def make_train_test_edge_data(data_dir, neg_sample_size):
    '''
    1. Load graph-data.pkl for the same format
    2. Load training/valid/testing triples, a list of edge (head id, (domain type, predicate, range type), tail id)
    '''
    print("Loading graph...")
    graph, _, _ = load_graph(data_dir, 10)
    
    print("Load training/valid/testing triples...")
    train_triples = pickle_load(data_dir + "/train_triples.pkl")
    valid_triples = pickle_load(data_dir + "/valid_triples.pkl")
    test_triples = pickle_load(data_dir + "/test_triples.pkl")

    print("Getting full negative samples (for APR evaluation) and make queries...")
    valid_queries = [make_valid_test_edge_from_triple(graph, edge, neg_sample_size) for edge in valid_triples]
    test_queries = [make_valid_test_edge_from_triple(graph, edge, neg_sample_size) for edge in test_triples]

    print("Getting one negative samples (for AUC evaluation) and make queries...")
    valid_queries += [make_valid_test_edge_from_triple(graph, edge, 1) for edge in valid_triples]
    test_queries += [make_valid_test_edge_from_triple(graph, edge, 1) for edge in test_triples]

    print("Dumping valid/test 1-chain queries")
    pickle.dump([q.serialize() for q in valid_queries], open(data_dir+"/val_edges.pkl", "w"), protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump([q.serialize() for q in test_queries], open(data_dir+"/test_edges.pkl", "w"), protocol=pickle.HIGHEST_PROTOCOL)

    print("Dumping train 1-chain queries")
    train_queries = [Query(("1-chain", e), None, None, keep_graph=True) for e in train_triples]
    pickle.dump([q.serialize() for q in train_queries], open(data_dir+"/train_edges.pkl", "w"), protocol=pickle.HIGHEST_PROTOCOL)

    print("Finish making training/valid/testing 1-chain queries")

    

def _discard_negatives(file_name, small_prop=0.9):
    queries = pickle.load(open(file_name, "rb"))
#    queries = [q if random.random() > small_prop else (q[0],[random.choice(tuple(q[1]))], None if q[2] is None else [random.choice(tuple(q[2]))]) for q in queries]
    queries = [q if random.random() > small_prop else (q[0],[random.choice(list(q[1]))], None if q[2] is None else [random.choice(list(q[2]))]) for q in queries] 
    pickle.dump(queries, open(file_name.split(".")[0] + "-split.pkl", "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    print "Finished", file_name

def discard_negatives(data_dir):
    _discard_negatives(data_dir + "/val_edges.pkl")
    _discard_negatives(data_dir + "/test_edges.pkl")
    for i in range(2,4):
        _discard_negatives(data_dir + "/val_queries_{:d}.pkl".format(i))
        _discard_negatives(data_dir + "/test_queries_{:d}.pkl".format(i))


# def make_train_test_query_data(data_dir):
#     graph, _, _ = load_graph(data_dir, 10)
#     queries_2, queries_3 = parallel_sample(graph, 20, 50000, data_dir, test=False)
#     t_queries_2, t_queries_3 = parallel_sample(graph, 20, 5000, data_dir, test=True)
#     t_queries_2 = list(set(t_queries_2) - set(queries_2))
#     t_queries_3 = list(set(t_queries_3) - set(queries_3))
#     pickle.dump([q.serialize() for q in queries_2], open(data_dir + "/train_queries_2.pkl", "wb"), protocol=pickle.HIGHEST_PROTOCOL)
#     pickle.dump([q.serialize() for q in queries_3], open(data_dir + "/train_queries_3.pkl", "wb"), protocol=pickle.HIGHEST_PROTOCOL)
#     pickle.dump([q.serialize() for q in t_queries_2[10000:]], open(data_dir + "/test_queries_2.pkl", "wb"), protocol=pickle.HIGHEST_PROTOCOL)
#     pickle.dump([q.serialize() for q in t_queries_3[10000:]], open(data_dir + "/test_queries_3.pkl", "wb"), protocol=pickle.HIGHEST_PROTOCOL)
#     pickle.dump([q.serialize() for q in t_queries_2[:10000]], open(data_dir + "/val_queries_2.pkl", "wb"), protocol=pickle.HIGHEST_PROTOCOL)
#     pickle.dump([q.serialize() for q in t_queries_3[:10000]], open(data_dir + "/val_queries_3.pkl", "wb"), protocol=pickle.HIGHEST_PROTOCOL)

def make_train_multiedge_query_data(data_dir, num_workers, samples_per_worker, mp_result_dir = None, id2geo = None):
    '''
    Args:
        id2geo: node id => [longitude, latitude] 
                if not None, we sample geographic query with target node as geographic entity
    '''
    graph, _, _ = load_graph(data_dir, 10)
    queries_2, queries_3 = parallel_sample(graph, 
            num_workers, 
            samples_per_worker, 
            data_dir, 
            test=False, 
            mp_result_dir = mp_result_dir, 
            id2geo = id2geo)
    
    if id2geo is not None:
        file_postfix = "-geo"
    else:
        file_postfix = ""
    pickle.dump([q.serialize() for q in queries_2], open(data_dir + "/train_queries_2{}.pkl".format(file_postfix), "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump([q.serialize() for q in queries_3], open(data_dir + "/train_queries_3{}.pkl".format(file_postfix), "wb"), protocol=pickle.HIGHEST_PROTOCOL)

def make_train_inter_query_data(data_dir, num_workers, samples_per_worker, max_inter_size=5, mp_result_dir = None, id2geo = None):
    '''
    This just like sample x-inter query from KG, Learning projection and intersection operator from the KG directly
    Args:
        id2geo: node id => [longitude, latitude] 
                if not None, we sample geographic query with target node as geographic entity
    '''
    graph, _, _ = load_graph(data_dir, 10)
    queries_dict = parallel_inter_query_sample(graph, num_workers, samples_per_worker, data_dir, 
            max_inter_size = max_inter_size, 
            test=False, 
            mp_result_dir = mp_result_dir,
            id2geo=id2geo)
    
    if id2geo is not None:
        file_postfix = "-geo"
    else:
        file_postfix = ""
    for arity in queries_dict:
        pickle.dump([q.serialize() for q in queries_dict[arity]], open(data_dir + "/train_inter_queries_{:d}{:s}.pkl".format(arity, file_postfix), "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    
    
if __name__ == "__main__":
    #make_train_test_query_data("/dfs/scratch0/nqe-bio/")
    #make_train_test_edge_data("/dfs/scratch0/nqe-bio/")
    # sample_new_clean("/dfs/scratch0/nqe-bio/")
    #clean_test()



    data_dir = "/home/gengchen/Attention_GraphQA/graphqembed/dbgeo/"
    # 1. make training/valid/testing 1-chain queries
    # make_train_test_edge_data(data_dir, 100)

    # 2. make valid/testing 2/3 edges queries
    # sample_new_clean(data_dir)


    # 3. make train 2/3 edges queries
    mp_result_dir = data_dir + "/train_queries_mp/"
    # make_train_multiedge_query_data(data_dir, 50, 20000,  mp_result_dir = mp_result_dir)


    # 4. make train x-inter queries
    mp_result_dir = data_dir + "/train_inter_queries_mp/"
    # make_train_inter_query_data(data_dir, 50, 10000,  max_inter_size=7, mp_result_dir = mp_result_dir)
    


    id2geo = pickle_load(data_dir + "/id2geo.pkl")

    # 5. make valid/testing 2/3 edges geographic queries, negative samples are geo-entities
    # sample_new_clean(data_dir, id2geo = id2geo)

    
    # 6. make train x-inter queries, negative samples are geo-entities
    print("Do geo contect sample!!!!!!")
    mp_result_geo_dir = data_dir + "/train_inter_queries_geo_mp/"
    # make_train_inter_query_data(data_dir, 50, 10000,  max_inter_size=7, mp_result_dir = mp_result_geo_dir, id2geo = id2geo)















    # # NOUSE. make train 2/3 edges queries
    # mp_result_geo_dir = data_dir + "/train_queries_geo_mp/"
    # # make_train_multiedge_query_data(data_dir, 50, 20000,  mp_result_dir = mp_result_geo_dir, id2geo = id2geo)
