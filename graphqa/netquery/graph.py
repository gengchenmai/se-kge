from collections import OrderedDict, defaultdict
import random
import re

from sets import Set

# def _reverse_relation(relation):
#     '''
#     reverse the triple template
#         relation: ('drug', 'hematopoietic_system_disease', 'drug')
#     '''
#     return (relation[-1], relation[1], relation[0])

# def _reverse_edge(edge):
#     '''
#     reverse the edge structure
#     edge: (359, ('drug', 'hematopoietic_system_disease', 'drug'), 273)
#     '''
#     return (edge[-1], _reverse_relation(edge[1]), edge[0])


class Formula():
    # query structure type: t: target node
    # 1. 1-chain: o->t
    # 2. 2-chain: o->o->t
    # 3. 3-chain: o->o->o->t
    # 4. 2-inter: o->t<-o
    # 5. 3-inter: o->t<-o
    #                ^
    #                |
    #                o
    # 
    # 6. 3-inter_chain: t<-o<-o
    #                   ^
    #                   |
    #                   o
    # 
    # 7. 3-chain_inter: t
    #                   ^
    #                   |
    #                o->o<-o
    def __init__(self, query_type, rels):
        '''
        query_type: a flag for query type
        rels: Basically, Just a tuple/list or a dict-like structure, 
            each item is a triple template (head entity domain, predicate, tail entity domain)
            It is converted from query_graph in Query() such that the structure like the original query_graph
            t: target node type; pi: predicate; ai: anchor node type; ei: bounded variable
                1-chain: ((t, p1, a1))
                2-chain: ((t, p1, e1),(e1, p2, a1))
                3-chain: ((t, p1, e1),(e1, p2, e2),(e2, p3, a1))
                2-inter: ((t, p1, a1),(t, p2, a2))
                3-inter: ((t, p1, a1),(t, p2, a2),(t, p3, a3))
                3-inter_chain:  (
                                    (t, p1, a1),
                                    (
                                        (t, p2, e1),
                                        (e1, p3, a2)
                                    )
                                )
                3-chain_inter:  (
                                    (t, p1, e1),
                                    (
                                        (e1, p2, a1),
                                        (e1, p3, a2)
                                    )
                                )
                x-inter: ((t, p1, e1),(e1, p2, e2),(e2, p3, a1), ...)
        '''
        self.query_type = query_type
        # Get the target node type
        self.target_mode = rels[0][0]
        self.rels = rels
        pattern = re.compile("[\d]+-inter$")
        # query structure type: t: target node
        # o->t   o->o->t   o->o->o->t
        if query_type == "1-chain" or query_type == "2-chain" or query_type == "3-chain":
            # get the anchor node types
            self.anchor_modes = (rels[-1][-1],)
        # o->t<-o      o->t<-o
        #                 ^
        #                 |
        #                 o
        # match x-inter
        # elif query_type == "2-inter" or query_type == "3-inter":
        elif pattern.match(query_type) is not None:
            self.anchor_modes = tuple([rel[-1] for rel in rels])
        #        t<-o<-o
        #        ^
        #        |
        #        o 
        elif query_type == "3-inter_chain":
            self.anchor_modes = (rels[0][-1], rels[1][-1][-1])
        #        t
        #        ^
        #        |
        #     o->o<-o
        elif query_type == "3-chain_inter":
            self.anchor_modes = (rels[1][0][-1], rels[1][1][-1])

    def __hash__(self):
         return hash((self.query_type, self.rels))

    def __eq__(self, other):
        return ((self.query_type, self.rels)) == ((other.query_type, other.rels))

    def __neq__(self, other):
        return ((self.query_type, self.rels)) != ((other.query_type, other.rels))

    def __str__(self):
        return self.query_type + ": " + str(self.rels)

class Query():

    def __init__(self, query_graph, neg_samples, hard_neg_samples, neg_sample_max=100, keep_graph=False):
        '''
        query_graph: Just the 1st item in each entry of train/val/test_edges and train/val/test_queries_2/3 
            ('1-chain', (1326, ('protein', 'catalysis', 'protein'), 8451))
        neg_samples: the negative sample node ids
            [105888, 108201, 101614, ...]
        hard_neg_samples: the hard negative sample node ids
            None
        neg_sample_max: the max negative sample size and hard negative sample size


        Return:
            self.anchor_nodes: a tuple: a list of anchor nodes id
            self.target_node: the target node id
            self.formula: a Formula() object
            self.query_graph: query_graph if keep_graph else None
            self.neg_samples: a list of negative node ids, sample from neg_samples
            self.hard_neg_samples: a list of hard negative node ids, sample from hard_neg_samples
        '''
        query_type = query_graph[0]
        pattern = re.compile("[\d]+-inter$")
        if query_type == "1-chain" or query_type == "2-chain" or query_type == "3-chain":
            self.formula = Formula(query_type, tuple([query_graph[i][1] for i in range(1, len(query_graph))]))
            self.anchor_nodes = (query_graph[-1][-1],)
        # elif query_type == "2-inter" or query_type == "3-inter":
        # if query_type is x-inter
        elif pattern.match(query_type) is not None:
            self.formula = Formula(query_type, tuple([query_graph[i][1] for i in range(1, len(query_graph))]))
            self.anchor_nodes = tuple([query_graph[i][-1] for i in range(1, len(query_graph))])
        elif query_type == "3-inter_chain":
            self.formula = Formula(query_type, (query_graph[1][1], (query_graph[2][0][1], query_graph[2][1][1])))
            self.anchor_nodes = (query_graph[1][-1], query_graph[2][-1][-1])
        elif query_type == "3-chain_inter":
            self.formula = Formula(query_type, (query_graph[1][1], (query_graph[2][0][1], query_graph[2][1][1])))
            self.anchor_nodes = (query_graph[2][0][-1], query_graph[2][1][-1])
        self.target_node = query_graph[1][0]
        if keep_graph:
            self.query_graph = query_graph
        else:
            self.query_graph = None
        if not neg_samples is None:
            self.neg_samples = list(neg_samples) if len(neg_samples) < neg_sample_max else random.sample(neg_samples, neg_sample_max)
        else:
            self.neg_samples = None
        if not hard_neg_samples is None:
            self.hard_neg_samples = list(hard_neg_samples) if len(hard_neg_samples) <= neg_sample_max else random.sample(hard_neg_samples, neg_sample_max)
        else:
            self.hard_neg_samples =  None

    def contains_edge(self, edge):
        '''
        Given a edge structure, decide where it is in the current query_graph
        edge: (359, ('drug', 'hematopoietic_system_disease', 'drug'), 273)
        '''
        if self.query_graph is None:
            raise Exception("Can only test edge contain if graph is kept. Reinit with keep_graph=True")
        edges =  self.query_graph[1:]
        if "inter_chain" in self.query_graph[0] or "chain_inter" in self.query_graph[0]:
            edges = (edges[0], edges[1][0], edges[1][1])
        # return edge in edges or (edge[1], _reverse_relation(edge[1]), edge[0]) in edges
        return edge in edges or self._reverse_edge(edge) in edges

    def get_edges(self):
        '''
        Given the current query_graph, get a list of edge structures and their reverse edge
        return a set of these edge strcuture
        '''
        if self.query_graph is None:
            raise Exception("Can only test edge contain if graph is kept. Reinit with keep_graph=True")
        edges =  self.query_graph[1:]
        if "inter_chain" in self.query_graph[0] or "chain_inter" in self.query_graph[0]:
            edges = (edges[0], edges[1][0], edges[1][1])
        # return set(edges).union(set([(e[-1], _reverse_relation(e[1]), e[0]) for e in edges]))
        return set(edges).union(set([self._reverse_edge(e) for e in edges]))

    def __hash__(self):
         return hash((self.formula, self.target_node, self.anchor_nodes))

    def __eq__(self, other):
        '''
        The euqavalence between two queries depend on:
            1. the query formula
            2. the target node id
            3. the list of anchor node ids
        '''
        return (self.formula, self.target_node, self.anchor_nodes) == (other.formula, other.target_node, other.anchor_nodes)

    def __neq__(self, other):
        return self.__hash__() != other.__hash__()

    def serialize(self):
        '''
        Serialize the current Query() object as an entry for train/val/test_edges and train/val/test_queries_2/3
        '''
        if self.query_graph is None:
            raise Exception("Cannot serialize query loaded with query graph!")
        return (self.query_graph, self.neg_samples, self.hard_neg_samples)

    @staticmethod
    def deserialize(serial_info, keep_graph=False):
        '''
        Given a entry (serial_info) in train/val/test_edges and train/val/test_queries_2/3
        parse it as Query() object
        '''
        return Query(serial_info[0], serial_info[1], serial_info[2], None if serial_info[1] is None else len(serial_info[1]), keep_graph=keep_graph)



class Graph():
    """
    Simple container for heteregeneous graph data.
    """
    def __init__(self, features, feature_dims, relations, adj_lists, rid2inverse = None):
        '''
        Args:
            features(nodes, mode): a embedding lookup function to make a dict() from node type to embeddingbag
                nodes: a lists of global node id which are in type (mode)
                mode: node type
                return: embedding vectors, shape [num_node, embed_dim]
            feature_dims: a dict() from node type to embed_dim for the previous GraphSAGE layer or the original embed_dim
            relations: a dict() of all triple templates
                key:    domain entity type
                value:  a list of tuples (range entity type, predicate)
            adj_lists: a dict about the edges in KG (note that, they already add all reverse edges)
                key: triple template, i.e. ('drug', 'psoriatic_arthritis', 'drug')
                value: a defaultdict about all the edges instance of this metapath
                    key: the head entity id
                    value: a set of tail entity ids
            rid2inverse: relation id => inverse relation id, used to reverse the relation
        Return:
            self.full_sets: a dict
                key: node type
                value: a set of all node ids with this type 
            self.full_lists: a dict, similar to self.full_sets, but change value from a set to a list, 
                this used for negative sampling for '1-chain', sample from the node set will the same type
        '''
        self.features = features
        self.feature_dims = feature_dims
        self.relations = relations
        self.adj_lists = adj_lists
        self.rid2inverse = rid2inverse
        # self.full_sets: a dict from node type to a set of all node ids with this type and appear as the head of a triple
        self.full_sets = defaultdict(set)
        self.full_lists = {}
        self.meta_neighs = defaultdict(dict)
        for rel, adjs in self.adj_lists.iteritems():
            full_set = set(self.adj_lists[rel].keys())
            self.full_sets[rel[0]] = self.full_sets[rel[0]].union(full_set)
        for mode, full_set in self.full_sets.iteritems():
            self.full_lists[mode] = list(full_set)
        self.make_node2type()
        self._cache_edge_counts()
        self._make_flat_adj_lists()

    

    def _reverse_relation(self, relation):
        '''
        reverse the triple template
            relation: ('drug', 'hematopoietic_system_disease', 'drug')
        '''
        if self.rid2inverse is None:
            return (relation[-1], relation[1], relation[0])
        else:
            return (relation[-1], str(self.rid2inverse[int(relation[1])]), relation[0])

    def _reverse_edge(self, edge):
        '''
        reverse the edge structure
        edge: (359, ('drug', 'hematopoietic_system_disease', 'drug'), 273)
        '''
        return (edge[-1], self._reverse_relation(edge[1]), edge[0])

    def _make_flat_adj_lists(self):
        '''
        self.flat_adj_lists: a dict
            key: node type A
            value: a dict
                key: head node id with type A
                value: a list of tuple (triple template, global tail node id)
        '''
        self.flat_adj_lists = defaultdict(lambda : defaultdict(list))
        for rel, adjs in self.adj_lists.iteritems():
            for node, neighs in adjs.iteritems():
                self.flat_adj_lists[rel[0]][node].extend([(rel, neigh) for neigh in neighs])

    def _cache_edge_counts(self):
        '''
        Compute the number of edges per triple template, and the weighted for each triple template and node type

        self.rel_edges: a dict
            key: triple template, i.e. ('drug', 'psoriatic_arthritis', 'drug')
            value: num of triples match this triple template
        self.edges: number of triple template * number of unique head entity 
        self.rel_weights: a dict
            key: triple template, i.e. ('drug', 'psoriatic_arthritis', 'drug')
            value: average number of triple per unique entity
        self.mode_edges: a dict()
            key: node type
            value: number of triples whose head entity is the node type
        self.mode_weights: a dict()
            key: node type
            value: edge_count / self.edges

        '''
        self.edges = 0.
        self.rel_edges = {}
        for r1 in self.relations:
            for r2 in self.relations[r1]:
                rel = (r1,r2[1], r2[0])
                self.rel_edges[rel] = 0.
                for adj_list in self.adj_lists[rel].values():
                    self.rel_edges[rel] += len(adj_list)
                    self.edges += 1.
        self.rel_weights = OrderedDict()
        self.mode_edges = defaultdict(float)
        self.mode_weights = OrderedDict()
        for rel, edge_count in self.rel_edges.iteritems():
            self.rel_weights[rel] = edge_count / self.edges
            self.mode_edges[rel[0]] += edge_count
        for mode, edge_count in self.mode_edges.iteritems():
            self.mode_weights[mode] = edge_count / self.edges

    def remove_edges(self, edge_list):
        '''
        Given a list of edges, remove it and its reverse edge from self.adj_lists
        Args:
            edge_list: a list of edges, like (122939, ('disease', '0', 'protein'), 107943)
        '''
        for edge in edge_list:
            try:
                self.adj_lists[edge[1]][edge[0]].remove(edge[-1])
            except Exception:
                continue

            try:
                self.adj_lists[self._reverse_relation(edge[1])][edge[-1]].remove(edge[0])
            except Exception:
                continue
        self.meta_neighs = defaultdict(dict)
        self._cache_edge_counts()
        self._make_flat_adj_lists()

    def get_all_edges(self, seed=0, exclude_rels=set([])):
        """
        Returns all edges in the form (node1, relation, node2), exclude edges whose match any of the triple templates in exclude_rels
        Args:
            seed: random seed
            exclude_rels: a set of triple templates need to be excluded from the final result
        """
        edges = []
        random.seed(seed)
        for rel, adjs in self.adj_lists.iteritems():
            if rel in exclude_rels:
                continue
            for node, neighs in adjs.iteritems():
                edges.extend([(node, rel, neigh) for neigh in neighs if neigh != -1])
        random.shuffle(edges)
        return edges

    def get_all_edges_byrel(self, seed=0, exclude_rels=set([])):
        """
        Returns a dict of all edge, exclude edges whose match any of the triple templates in exclude_rels
        Args:
            seed: random seed
            exclude_rels: a set of triple templates need to be excluded from the final result
        Return:
            edges: a dict
                key: triple template
                value: a set of unique tuple (head id, tail id)
        """
        random.seed(seed)
        edges = defaultdict(list)
        for rel, adjs in self.adj_lists.iteritems():
            if rel in exclude_rels:
                continue
            for node, neighs in adjs.iteritems():
                edges[(rel,)].extend([(node, neigh) for neigh in neighs if neigh != -1])
        random.shuffle(edges)
        return edges

    def get_negative_edge_samples(self, edge, num, rejection_sample=True):
        '''
        Given one edge, get N (N=num) negative samples for the head id such that, 
        the negative nodes has the same node type as head id but doe snot satify the edge
        Args:
            edge: an edge in the form (node1, relation, node2), like (122939, ('disease', '0', 'protein'), 107943)
            num: the number of negative samples
            rejection_sample: whether to do rejection sampling
        '''
        if rejection_sample:
            neg_nodes = set([])
            counter = 0
            while len(neg_nodes) < num:
                neg_node = random.choice(self.full_lists[edge[1][0]])
                if not neg_node in self.adj_lists[self._reverse_relation(edge[1])][edge[2]]:
                    neg_nodes.add(neg_node)
                counter += 1
                if counter > 100*num:
                    return self.get_negative_edge_samples(edge, num, rejection_sample=False)
        else:
            neg_nodes = self.full_sets[edge[1][0]] - self.adj_lists[self._reverse_relation(edge[1])][edge[2]]
        neg_nodes = list(neg_nodes) if len(neg_nodes) <= num else random.sample(list(neg_nodes), num)
        return neg_nodes

    def sample_test_queries(self, train_graph, q_types, samples_per_type, neg_sample_max, verbose=True, id2geo = None):
        '''
        Sample the testing/validation queries for different query type, the negative sampling is operating on the whole graph
        NOTE: make sure the sampled query is not directly answerable based on training graph
        Args:
            train_graph: a Graph() which represent the training graph
            q_types: a list of query types
            samples_per_type: number of query sampled per query type
            neg_sample_max: the maximum negative samples
            verbose: whether to print the query sampling number
            id2geo: node id => [longitude, latitude], if not None, we simple query with target node has coordinate
        Return:
            queries: a list of Query() which is the sampled query
        '''
        queries = []
        if id2geo is not None:
            geoid_list = list(id2geo.keys())
        for q_type in q_types:
            sampled = 0
            while sampled < samples_per_type:
                # sample a query from the whole graph
                if id2geo is None:
                    q = self.sample_query_subgraph_bytype(q_type)
                else:
                    # sample a geographic node as the target node for the sampled query
                    geoid = random.choice(geoid_list)
                    geomode = self.node2type[geoid]
                    q = self.sample_query_subgraph_bytype(q_type, start_node = (geoid, geomode))
                # if the query is None or if the target node in the training graph, reject this query
                # This control the test query to be unanswerable based on training graph
                if q is None or not train_graph._is_negative(q, q[1][0], False):
                    continue
                # We can do negetive sampling in the whole graph
                negs, hard_negs = self.get_negative_samples(q, id2geo = id2geo)
                if negs is None or ("inter" in q[0] and hard_negs is None):
                    continue
                query = Query(q, negs, hard_negs, neg_sample_max=neg_sample_max, keep_graph=True)
                queries.append(query)
                sampled += 1
                if sampled % 1000 == 0 and verbose:
                    print("Sampled {}".format(sampled))
        return queries

    def sample_queries(self, arity, num_samples, neg_sample_max, verbose=True, id2geo = None):
        '''
        Sample the training queries given arity
        Args:
            arity: the number of edge in the query to be sampled
            num_samples: number of sampled query for thsi arity
            neg_sample_max: the maximum negative samples
            verbose: whether to print the query sampling number
            id2geo: node id => [longitude, latitude] 
                    if not None, we sample geographic query with target node as geographic entity
        Return:
            queries: a list of Query() which is the sampled query
        '''
        sampled = 0
        queries = []
        if id2geo is not None:
            geoid_list = list(id2geo.keys())
        while sampled < num_samples:
            if id2geo is None:
                q = self.sample_query_subgraph(arity)
            else:
                # sample a geographic node as the target node for the sampled query
                geoid = random.choice(geoid_list)
                geomode = self.node2type[geoid]
                q = self.sample_query_subgraph(arity, start_node = (geoid, geomode))
            if q is None:
                continue
            negs, hard_negs = self.get_negative_samples(q)
            if negs is None or ("inter" in q[0] and hard_negs is None):
                continue
            query = Query(q, negs, hard_negs, neg_sample_max=neg_sample_max, keep_graph=True)
            queries.append(query)
            sampled += 1
            if sampled % 1000 == 0 and verbose:
                print "Sampled", sampled
        return queries


    def get_negative_samples(self, query, id2geo = None):
        '''
        Given a query, get the negative samples and hard negative samples for the target node
        if id2geo is not None, both neg_samples and hard_neg_samples should be geo-entities
        Args:
            query: a tuple, (query_type, edge1, edge2, ...), for 3-inter_chain and 3-chain_inter, the 3rd item is a tuple of two edges
        Return:
            neg_samples: a set of nodes whose with the target node type, but do not satify the current query
            hard_neg_samples: a set of nodes whose with the target node type, also satify one or more edge, but do not satify the current whole query
                        only available for "inter" query
        '''
        if id2geo is not None:
            geoset = Set(id2geo.keys())
        if query[0] == "3-chain" or query[0] == "2-chain":
            edges = query[1:]
            # rels: [(a1, p3, e2), (e2, p2, e1), (e1, p1, t)]
            rels = [self._reverse_relation(edge[1]) for edge in edges[::-1]]
            # meta_neighs: a set of node ids which all satisfy metapath (rels) from the ancor node (query[-1][-1])
            meta_neighs = self.get_metapath_neighs(query[-1][-1], tuple(rels))
            # query[1][1][0]: target node type
            # negative_samples: all node with target node type - meta_neighs
            if id2geo is None:
                negative_samples = self.full_sets[query[1][1][0]] - meta_neighs
            else:
                negative_samples = self.full_sets[query[1][1][0]].intersection(geoset) - meta_neighs
            if len(negative_samples) == 0:
                return None, None
            else:
                return negative_samples, None
        elif query[0] == "2-inter" or query[0] == "3-inter":
            # (a1, p1, t)
            rel_1 = self._reverse_relation(query[1][1])
            # all node in target node position which satisfy the first edge (reverse) (a1, p1, t)
            # union_neighs: the union of all nodes in target node position which satisfy all 2/3 edges
            # inter_neighs: the intersection of all nodes in target node position which satisfy all 2/3 edges (This is the real answer set for the whole query)
            union_neighs = self.adj_lists[rel_1][query[1][-1]]
            inter_neighs = self.adj_lists[rel_1][query[1][-1]]
            for i in range(2,len(query)):
                rel = self._reverse_relation(query[i][1])
                union_neighs = union_neighs.union(self.adj_lists[rel][query[i][-1]])
                inter_neighs = inter_neighs.intersection(self.adj_lists[rel][query[i][-1]])
            if id2geo is None:
                neg_samples = self.full_sets[query[1][1][0]] - inter_neighs
                hard_neg_samples = union_neighs - inter_neighs
            else:
                neg_samples = self.full_sets[query[1][1][0]].intersection(geoset) - inter_neighs
                hard_neg_samples = union_neighs.intersection(geoset) - inter_neighs
            if len(neg_samples) == 0 or len(hard_neg_samples) == 0:
                return None, None
            return neg_samples, hard_neg_samples
        elif query[0] == "3-inter_chain":
            rel_1 = self._reverse_relation(query[1][1])
            # all node in target node position which satisfy the first edge (reverse) (a1, p1, t)
            union_neighs = self.adj_lists[rel_1][query[1][-1]]
            inter_neighs = self.adj_lists[rel_1][query[1][-1]]
            # chain_rels: [(a2, p3, e1) (e1, p2, t)]                          
            chain_rels  = [self._reverse_relation(edge[1]) for edge in query[2][::-1]]
            # chain_neighs: all nodes in target node position which satisfy the 2nd chain (reverse) [(a2, p3, e1) (e1, p2, t)]
            chain_neighs = self.get_metapath_neighs(query[2][-1][-1], tuple(chain_rels))
            union_neighs = union_neighs.union(chain_neighs)
            inter_neighs = inter_neighs.intersection(chain_neighs)
            if id2geo is None:
                neg_samples = self.full_sets[query[1][1][0]] - inter_neighs
                hard_neg_samples = union_neighs - inter_neighs
            else:
                neg_samples = self.full_sets[query[1][1][0]].intersection(geoset) - inter_neighs
                hard_neg_samples = union_neighs.intersection(geoset) - inter_neighs
            if len(neg_samples) == 0 or len(hard_neg_samples) == 0:
                return None, None
            return neg_samples, hard_neg_samples
        elif query[0] == "3-chain_inter":
            # here t, ex, ax represent node id, px represent triple template, not the same as before
            # 3-chain_inter:  (
            #                     query type
            #                     (t, p1, e1),
            #                     (
            #                         (e1, p2, a1),
            #                         (e1, p3, a2)
            #                     )
            #                 )
            inter_rel_1 = self._reverse_relation(query[-1][0][1])
            inter_neighs_1 = self.adj_lists[inter_rel_1][query[-1][0][-1]]
            inter_rel_2 = self._reverse_relation(query[-1][1][1])
            inter_neighs_2 = self.adj_lists[inter_rel_2][query[-1][1][-1]]
            
            inter_neighs = inter_neighs_1.intersection(inter_neighs_2)
            union_neighs = inter_neighs_1.union(inter_neighs_2)
            rel = self._reverse_relation(query[1][1])
            pos_nodes = set([n for neigh in inter_neighs for n in self.adj_lists[rel][neigh]]) 
            union_pos_nodes = set([n for neigh in union_neighs for n in self.adj_lists[rel][neigh]]) 
            if id2geo is None:
                neg_samples = self.full_sets[query[1][1][0]] - pos_nodes
                hard_neg_samples = union_pos_nodes - pos_nodes
            else:
                neg_samples = self.full_sets[query[1][1][0]].intersection(geoset) - pos_nodes
                hard_neg_samples = union_pos_nodes.intersection(geoset) - pos_nodes
            if len(neg_samples) == 0 or len(hard_neg_samples) == 0:
                return None, None
            return neg_samples, hard_neg_samples

    

    def sample_edge(self, node, mode):
        '''
        Randomly sample an edge from graph, based on the head node id and type
        '''
        rel, neigh = random.choice(self.flat_adj_lists[mode][node])
        edge = (node, rel, neigh)
        return edge

    def sample_query_subgraph_bytype(self, q_type, start_node=None):
        '''
        Given a query type, and a start_node (target node id, target node type), sample a query from the adj_lists
        Args:
            q_type: query type
            start_node: a tupe, (target node id, target node type)
        Return:
            query: a tuple, (query_type, edge1, edge2, ...), for 3-inter_chain and 3-chain_inter, the 3rd item is a tuple of two edges
        '''
        if start_node is None:
            start_rel = random.choice(self.adj_lists.keys())
            node = random.choice(self.adj_lists[start_rel].keys())
            mode = start_rel[0]
        else:
            node, mode = start_node

        if q_type[0] == "3":
            # num_edges: the number of edge connect target node
            if q_type == "3-chain" or q_type == "3-chain_inter":
                num_edges = 1
            elif q_type == "3-inter_chain":
                num_edges = 2
            elif q_type == "3-inter":
                num_edges = 3
            if num_edges > len(self.flat_adj_lists[mode][node]):
                return None
            if num_edges == 1:
                # randomly select rel (triple template) and neigh (tail node id) from node
                rel, neigh = random.choice(self.flat_adj_lists[mode][node])
                edge = (node, rel, neigh)
                # Something wrong with their code?!!!!, not rel[0], should be rel[-1], for the tail node type
                next_query = self.sample_query_subgraph_bytype(
                    "2-chain" if q_type == "3-chain" else "2-inter", start_node=(neigh, rel[-1]))
                # next_query = self.sample_query_subgraph_bytype(
                #     "2-chain" if q_type == "3-chain" else "2-inter", start_node=(neigh, rel[0]))
                if next_query is None:
                    return None
                if next_query[0] == "2-chain":
                    return ("3-chain", edge, next_query[1], next_query[2])
                else:
                    return ("3-chain_inter", edge, (next_query[1], next_query[2]))
            elif num_edges == 2:
                # make sure the randomly sampled 2 edges are not the same
                rel_1, neigh_1 = random.choice(self.flat_adj_lists[mode][node])
                edge_1 = (node, rel_1, neigh_1)
                neigh_2 = neigh_1
                rel_2 = rel_1
                while (neigh_1, rel_1) == (neigh_2, rel_2):
                    rel_2, neigh_2 = random.choice(self.flat_adj_lists[mode][node])
                edge_2 = (node, rel_2, neigh_2)
                return ("3-inter_chain", edge_1, (edge_2, self.sample_edge(neigh_2, rel_2[-1])))
            elif num_edges == 3:
                # make sure the randomly sampled 3 edges are not the same
                rel_1, neigh_1 = random.choice(self.flat_adj_lists[mode][node])
                edge_1 = (node, rel_1, neigh_1)
                neigh_2 = neigh_1
                rel_2 = rel_1
                while (rel_1, neigh_1) == (rel_2, neigh_2):
                    rel_2, neigh_2 = random.choice(self.flat_adj_lists[mode][node])
                edge_2 = (node, rel_2, neigh_2)
                neigh_3 = neigh_1
                rel_3 = rel_1
                while ((rel_1, neigh_1) == (rel_3, neigh_3)) or ((rel_2, neigh_2) == (rel_3, neigh_3)):
                    rel_3, neigh_3 = random.choice(self.flat_adj_lists[mode][node])
                edge_3 = (node, rel_3, neigh_3)
                return ("3-inter", edge_1, edge_2, edge_3)

        if q_type[0] == "2":
            num_edges = 1 if q_type == "2-chain" else 2
            # See whether there are enough edges for sampling
            if num_edges > len(self.flat_adj_lists[mode][node]):
                return None
            if num_edges == 1:
                rel, neigh = random.choice(self.flat_adj_lists[mode][node])
                edge = (node, rel, neigh)
                return ("2-chain", edge, self.sample_edge(neigh, rel[-1]))
            elif num_edges == 2:
                rel_1, neigh_1 = random.choice(self.flat_adj_lists[mode][node])
                edge_1 = (node, rel_1, neigh_1)
                neigh_2 = neigh_1
                rel_2 = rel_1
                while (neigh_1, rel_1) == (neigh_2, rel_2):
                    rel_2, neigh_2 = random.choice(self.flat_adj_lists[mode][node])
                edge_2 = (node, rel_2, neigh_2)
                return ("2-inter", edge_1, edge_2)


    def sample_query_subgraph(self, arity, start_node=None):
        '''
        Given arity, and a start_node (target node id, target node type), sample a query from the adj_lists
        Args:
            arity: the number of edge in the query to be sampled
            start_node: a tupe, (target node id, target node type)
        Return:
            query: a tuple, (query_type, edge1, edge2, ...), for 3-inter_chain and 3-chain_inter, the 3rd item is a tuple of two edges
        '''
        if start_node is None:
            start_rel = random.choice(list(self.adj_lists.keys()))
            node = random.choice(list(self.adj_lists[start_rel].keys()))
            mode = start_rel[0]
        else:
            node, mode = start_node
        if arity > 3 or arity < 2:
            raise Exception("Only arity of at most 3 is supported for queries")

        if arity == 3:
            '''
            num_edges: the number of edge connect target node
            1/2 prob of 1 edge, 1/4 prob of 2, 1/4 prob of 3
            1: 3-chain, 3-chain_inter
            2: 3-inter_chain
            3: 3-inter
            '''
            num_edges = random.choice([1,1,2,3])
            if num_edges > len(self.flat_adj_lists[mode][node]):
                return None
            if num_edges == 1:
                rel, neigh = random.choice(self.flat_adj_lists[mode][node])
                edge = (node, rel, neigh)
                # Something wrong with their code?!!!!, not rel[0], should be rel[-1], for the tail node type
                next_query = self.sample_query_subgraph(2, start_node=(neigh, rel[-1]))
                # next_query = self.sample_query_subgraph(2, start_node=(neigh, rel[0]))
                if next_query is None:
                    return None
                if next_query[0] == "2-chain":
                    return ("3-chain", edge, next_query[1], next_query[2])
                else:
                    # 2-inter
                    return ("3-chain_inter", edge, (next_query[1], next_query[2]))
            elif num_edges == 2:
                # 3-inter_chain
                rel_1, neigh_1 = random.choice(self.flat_adj_lists[mode][node])
                edge_1 = (node, rel_1, neigh_1)
                neigh_2 = neigh_1
                rel_2 = rel_1
                while (neigh_1, rel_1) == (neigh_2, rel_2):
                    rel_2, neigh_2 = random.choice(self.flat_adj_lists[mode][node])
                edge_2 = (node, rel_2, neigh_2)
                return ("3-inter_chain", edge_1, (edge_2, self.sample_edge(neigh_2, rel_2[-1])))
            elif num_edges == 3:
                # 3-inter
                rel_1, neigh_1 = random.choice(self.flat_adj_lists[mode][node])
                edge_1 = (node, rel_1, neigh_1)
                neigh_2 = neigh_1
                rel_2 = rel_1
                while (rel_1, neigh_1) == (rel_2, neigh_2):
                    rel_2, neigh_2 = random.choice(self.flat_adj_lists[mode][node])
                edge_2 = (node, rel_2, neigh_2)
                neigh_3 = neigh_1
                rel_3 = rel_1
                while ((rel_1, neigh_1) == (rel_3, neigh_3)) or ((rel_2, neigh_2) == (rel_3, neigh_3)):
                    rel_3, neigh_3 = random.choice(self.flat_adj_lists[mode][node])
                edge_3 = (node, rel_3, neigh_3)
                return ("3-inter", edge_1, edge_2, edge_3)

        if arity == 2:
            '''
            num_edges: the number of edge connect target node
            1: 2-chain
            2: 2-inter
            '''
            num_edges = random.choice([1,2])
            if num_edges > len(self.flat_adj_lists[mode][node]):
                return None
            if num_edges == 1:
                # 2-chain
                rel, neigh = random.choice(self.flat_adj_lists[mode][node])
                edge = (node, rel, neigh)
                return ("2-chain", edge, self.sample_edge(neigh, rel[-1]))
            elif num_edges == 2:
                # 2-inter
                rel_1, neigh_1 = random.choice(self.flat_adj_lists[mode][node])
                edge_1 = (node, rel_1, neigh_1)
                neigh_2 = neigh_1
                rel_2 = rel_1
                while (neigh_1, rel_1) == (neigh_2, rel_2):
                    rel_2, neigh_2 = random.choice(self.flat_adj_lists[mode][node])
                edge_2 = (node, rel_2, neigh_2)
                return ("2-inter", edge_1, edge_2)

    def sample_inter_queries_by_arity(self, arity, num_samples, neg_sample_max, verbose=True, id2geo = None):
        '''
        Sample the training x-inter queries given arity, equal to sample the node neighborhood with different neighborhood sample size
        Args:
            arity: the number of edge in the query to be sampled
            num_samples: number of sampled query for thsi arity
            neg_sample_max: the maximum negative samples
            verbose: whether to print the query sampling number
            id2geo: node id => [longitude, latitude] 
                if not None, we sample geographic query with target node as geographic entity
        Return:
            queries: a list of Query() which is the sampled query
        '''
        # get the nodes whose neighborhood size >= arity
        # If id2geo is not None, get a list of (geo-entity, mode), 
        #     who have >= arity number of geo-triple
        node_list = self.get_nodes_by_arity(arity, id2geo = id2geo)
        # if id2geo is not None:
        #     geo_node_list = [(geoid, self.node2type[geoid]) for geoid in id2geo]
        #     node_list = list(Set(possible_node_list).intersection(Set(geo_node_list)))
        # else:
        #     node_list = possible_node_list
        if len(node_list) == 0:
            raise Exception("There is no entities with node degree >= {}".format(arity))
        sampled = 0
        queries = []
        while sampled < num_samples:
            q = self.sample_inter_query_subgraph(arity, possible_node_list=node_list, id2geo = id2geo)
            if q is None:
                continue
            negs, hard_negs = self.get_inter_query_negative_samples(q, id2geo = id2geo)
            if negs is None or hard_negs is None:
                continue
            query = Query(q, negs, hard_negs, neg_sample_max=neg_sample_max, keep_graph=True)
            queries.append(query)
            sampled += 1
            if sampled % 1000 == 0 and verbose:
                print "Sampled", sampled
        return queries

    def make_node2type(self):
        self.node2type = dict()
        for rel in self.adj_lists:
            for h in self.adj_lists[rel]:
                self.node2type[h] = rel[0]
                for t in self.adj_lists[rel][h]:
                    self.node2type[t] = rel[-1]
        return

    def get_nodes_by_arity(self, arity, id2geo = None):
        '''
        Get a list of (node, mode) whose degree is larger or equal to arity
        If id2geo is not None, get a list of geo-entity, 
        who have >= arity number of geo-triple
        '''
        node_list = []
        for mode in self.flat_adj_lists:
            for node in self.flat_adj_lists[mode]:
                if id2geo is None:
                    if len(self.flat_adj_lists[mode][node]) >= arity:
                        node_list.append((node, mode))
                else:
                    if node in id2geo:
                        geo_context = [(rel, tail) for rel, tail in self.flat_adj_lists[mode][node] if tail in id2geo]
                        if len(geo_context) >= arity:
                            node_list.append((node, mode))
        return node_list


    def sample_inter_query_subgraph(self, arity, start_node=None, possible_node_list = [], id2geo = None):
        '''
        Given arity, and a start_node (target node id, target node type), sample a query from the adj_lists
        Similar to sample_query_subgraph(), but here, we only sample inter query which is equalvalent to sample a node's neighborhood
        Args:
            arity: the number of edge in the query to be sampled
            start_node: a tupe, (target node id, target node type)
            possible_node_list: a list of (node, mode) whose degree is larger or equal to arity
            id2geo: dict(), node id => [longitude, latitude] 
        Return:
            query: a tuple, (query_type, edge1, edge2, ...), for 3-inter_chain and 3-chain_inter, the 3rd item is a tuple of two edges
        '''
        if start_node is not None:
            node, mode = start_node
        elif start_node is None and len(possible_node_list) != 0:
            node, mode = random.choice(possible_node_list)
        else:
            start_rel = random.choice(self.adj_lists.keys())
            node = random.choice(self.adj_lists[start_rel].keys())
            mode = start_rel[0]
            while len(self.flat_adj_lists[mode][node]) < arity:
                start_rel = random.choice(self.adj_lists.keys())
                node = random.choice(self.adj_lists[start_rel].keys())
                mode = start_rel[0]
        
            
        if arity < 2:
            raise Exception("Arity should be larger than or equal to 2")

        # x-inter
        
        if id2geo is not None:
            # if id2geo is not None, we alreay make sure, 
            # node is a geo-entity, 
            # it has >=arity number of geo-triple
            assert node in id2geo
            geo_context = [(rel, tail) for rel, tail in self.flat_adj_lists[mode][node] if tail in id2geo]
            assert len(geo_context) >= arity
            rel_tail_list = random.sample(geo_context, arity)
        else:
            # We already make sure the neighborhood size of the current node is larger than arity
            assert len(self.flat_adj_lists[mode][node]) >= arity
            rel_tail_list = random.sample(self.flat_adj_lists[mode][node], arity)
        query_graph = ["{}-inter".format(arity)]
        for rel, neigh in rel_tail_list:
            query_graph.append((node, rel, neigh))
        return tuple(query_graph)

    def get_inter_query_negative_samples(self, query, id2geo = None):
        '''
        Given a inter query, get the negative samples and hard negative samples for the target node
        if id2geo is not None:
        then both neg_samples and hard_neg_samples should be geo-entities
        Args:
            query: a tuple, (query_type, edge1, edge2, ...), for 3-inter_chain and 3-chain_inter, the 3rd item is a tuple of two edges
        Return:
            neg_samples: a set of nodes whose with the target node type, but do not satify the current query
            hard_neg_samples: a set of nodes whose with the target node type, also satify one or more edge, but do not satify the current whole query
                        only available for "inter" query
        '''
        num_edges = int(query[0].replace("-inter", ""))
        assert num_edges == len(query)-1
        # (a1, p1, t)
        rel_1 = self._reverse_relation(query[1][1])
        # all node in target node position which satisfy the first edge (reverse) (a1, p1, t)
        # union_neighs: the union of all nodes in target node position which satisfy all 2/3 edges
        # inter_neighs: the intersection of all nodes in target node position which satisfy all 2/3 edges (This is the real answer set for the whole query)
        union_neighs = self.adj_lists[rel_1][query[1][-1]]
        inter_neighs = self.adj_lists[rel_1][query[1][-1]]
        for i in range(2,len(query)):
            rel = self._reverse_relation(query[i][1])
            union_neighs = union_neighs.union(self.adj_lists[rel][query[i][-1]])
            inter_neighs = inter_neighs.intersection(self.adj_lists[rel][query[i][-1]])
        if id2geo is None:
            neg_samples = self.full_sets[query[1][1][0]] - inter_neighs
            hard_neg_samples = union_neighs - inter_neighs
        else:
            geoset = Set(id2geo.keys())
            neg_samples = self.full_sets[query[1][1][0]].intersection(geoset) - inter_neighs
            hard_neg_samples = union_neighs.intersection(geoset) - inter_neighs
        if len(neg_samples) == 0 or len(hard_neg_samples) == 0:
            return None, None
        return neg_samples, hard_neg_samples
        
            

    def get_metapath_neighs(self, node, rels):
        '''
        Given a center node and a metapath, return a set of node ids which are the end by following the metapath from this center node
        Args:
            node: a center node id (ancor node) a
            rels: a type of metapath, from the center node, a tuple of triple templates, ((a, p1, t1), (t1, p2, t2), ...)
        Return:
            current_set: a set of node ids which are the end by following the metapath from this center node
            self.meta_neighs: a dict()
                key: a type of metapath, from the center node, a tuple of triple templates, ((a, p1, t1), (t1, p2, t2), ...)
                value: a dict()
                    key: the center node if
                    value: a set of nodes which are end nodes from the center node and follow the metapath (rels)
        '''
        if node in self.meta_neighs[rels]:
            return self.meta_neighs[rels][node]
        current_set = [node]
        for rel in rels:
            # for each step, get 1-d neighborhood node set of current_set
            current_set = set([neigh for n in current_set for neigh in self.adj_lists[rel][n]])
        # after n step (n=length of metapath), we get a set of nodes who the n-degree neighbors by following the metapath from center node
        self.meta_neighs[rels][node] = current_set
        return current_set

    ## TESTING CODE

    def _check_edge(self, query, i):
        '''
        Check the ith edge in query in the graph
        True: ith edge is correct
        False: ith edge is not in the graph
        '''
        return query[i][-1] in self.adj_lists[query[i][1]][query[i][0]]

    def _is_subgraph(self, query, verbose):
        '''
        Check the query quality, raise exception when the query structure does not match the query type
        Args:
            query: a tuple, (query_type, edge1, edge2, ...), for 3-inter_chain and 3-chain_inter, the 3rd item is a tuple of two edges
        Return:
            raise exception when the query structure does not match the query type
        '''
        if query[0] == "3-chain":
            for i in range(3):
                if not self._check_edge(query, i+1):
                    raise Exception(str(query))
            if not (query[1][-1] == query[2][0] and query[2][-1] == query[3][0]):
                raise Exception(str(query))
        if query[0] == "2-chain":
            for i in range(2):
                if not self._check_edge(query, i+1):
                    raise Exception(str(query))
            if not query[1][-1] == query[2][0]:
                raise Exception(str(query))
        if query[0] == "2-inter":
            for i in range(2):
                if not self._check_edge(query, i+1):
                    raise Exception(str(query))
            if not query[1][0] == query[2][0]:
                raise Exception(str(query))
        if query[0] == "3-inter":
            for i in range(3):
                if not self._check_edge(query, i+1):
                    raise Exception(str(query))
            if not (query[1][0] == query[2][0] and query[2][0] == query[3][0]):
                raise Exception(str(query))
        if query[0] == "3-inter_chain":
            if not (self._check_edge(query, 1) and self._check_edge(query[2], 0) and self._check_edge(query[2], 1)):
                raise Exception(str(query))
            if not (query[1][0] == query[2][0][0] and query[2][0][-1] == query[2][1][0]):
                raise Exception(str(query))
        if query[0] == "3-chain_inter":
            if not (self._check_edge(query, 1) and self._check_edge(query[2], 0) and self._check_edge(query[2], 1)):
                raise Exception(str(query))
            if not (query[1][-1] == query[2][0][0] and query[2][0][0] == query[2][1][0]):
                raise Exception(str(query))
        return True

    def _is_negative(self, query, neg_node, is_hard):
        '''
        Given a query and a neg_node in the target node position, decide whether neg_node is the (hard) negative sample for this query
        Args:
            query: a tuple, (query_type, edge1, edge2, ...), for 3-inter_chain and 3-chain_inter, the 3rd item is a tuple of two edges
            neg_node: node id
            is_hard: True/False, do hard negative sample
        Return:
            is_hard == True:
                True: neg_node is a hard negative sample
                False: neg_node is not a hard negative sample
            is_hard == False:
                True: neg_node is a negative sample
                False: neg_node is not a negative sample
        '''
        if query[0] == "2-chain":
            query = (query[0], (neg_node, query[1][1], query[1][2]), query[2])
            if query[2][-1] in self.get_metapath_neighs(query[1][0], (query[1][1], query[2][1])):
                return False
        if query[0] == "3-chain":
            query = (query[0], (neg_node, query[1][1], query[1][2]), query[2], query[3])
            if query[3][-1] in self.get_metapath_neighs(query[1][0], (query[1][1], query[2][1], query[3][1])):
                return False
        if query[0] == "2-inter":
            query = (query[0], (neg_node, query[1][1], query[1][2]), (neg_node, query[2][1], query[2][2]))
            if not is_hard:
                # if the 1st and 2nd edge are all in the graph, this is not a negative sample
                if self._check_edge(query, 1) and self._check_edge(query, 2):
                    return False
            else:
                # (self._check_edge(query, 1) and self._check_edge(query, 2)): satisfy both edge
                # not (self._check_edge(query, 1) or self._check_edge(query, 2)): whole - the union of 1st and 2nd edge
                # Basic, if neg_node is not a hard negative sample
                if (self._check_edge(query, 1) and self._check_edge(query, 2)) or not (self._check_edge(query, 1) or self._check_edge(query, 2)):
                    return False
        if query[0] == "3-inter":
            query = (query[0], (neg_node, query[1][1], query[1][2]), (neg_node, query[2][1], query[2][2]), (neg_node, query[3][1], query[3][2]))
            if not is_hard:
                if self._check_edge(query, 1) and self._check_edge(query, 2) and self._check_edge(query, 3):
                    return False
            else:
                if (self._check_edge(query, 1) and self._check_edge(query, 2) and self._check_edge(query, 3))\
                        or not (self._check_edge(query, 1) or self._check_edge(query, 2) or self._check_edge(query, 3)):
                    return False
        if query[0] == "3-inter_chain":
            query = (query[0], (neg_node, query[1][1], query[1][2]), ((neg_node, query[2][0][1], query[2][0][2]), query[2][1]))
            # check whether neg_node satisfy 2nd chain
            meta_check = lambda : query[2][-1][-1] in self.get_metapath_neighs(query[1][0], (query[2][0][1], query[2][1][1]))
            # check whether neg_node satisfy 1st edge 
            neigh_check = lambda : self._check_edge(query, 1)
            if not is_hard:
                if meta_check() and neigh_check():
                    return False
            else:
                if (meta_check() and neigh_check()) or not (meta_check() or neigh_check()):
                    return False
        if query[0] == "3-chain_inter":
            query = (query[0], (neg_node, query[1][1], query[1][2]), query[2])
            target_neigh = self.adj_lists[query[1][1]][neg_node]
            neigh_1 = self.adj_lists[self._reverse_relation(query[2][0][1])][query[2][0][-1]]
            neigh_2 = self.adj_lists[self._reverse_relation(query[2][1][1])][query[2][1][-1]]
            if not is_hard:
                if target_neigh in neigh_1.intersection(neigh_2):
                    return False
            else:
                # Something wrong?!!! should be or, not and,
                if target_neigh in neigh_1.intersection(neigh_2) or not target_neigh in neigh_1.union(neigh_2):
                # if target_neigh in neigh_1.intersection(neigh_2) and not target_neigh in neigh_1.union(neigh_2):
                    return False
        return True

            

    def _run_test(self, num_samples=1000):
        '''
        This is a test function to test to robustness of sample_query_subgraph() and get_negative_samples()
        '''
        for i in range(num_samples):
            q = self.sample_query_subgraph(2)
            if q is None:
                continue
            self._is_subgraph(q, True)
            negs, hard_negs = self.get_negative_samples(q)
            if not negs is None:
                for n in negs:
                    self._is_negative(q, n, False)
            if not hard_negs is None:
                for n in hard_negs:
                    self._is_negative(q, n, True)
            q = self.sample_query_subgraph(3)
            if q is None:
                continue
            self._is_subgraph(q, True)
            negs, hard_negs = self.get_negative_samples(q)
            if not negs is None:
                for n in negs:
                    self._is_negative(q, n, False)
            if not hard_negs is None:
                for n in hard_negs:
                    self._is_negative(q, n, True)
        return True


    """
    TO DELETE? 
    def sample_chain_from_node(self, length, node, rel):
        rels = [rel]
        for cur_len in range(length-1):
            next_rel = random.choice(self.relations[rels[-1][-1]])
            rels.append((rels[-1][-1], next_rel[-1], next_rel[0]))

        rels = tuple(rels)
        meta_neighs = self.get_metapath_neighs(node, rels)
        rev_rel = _reverse_relation(rels[-1])
        full_set = self.full_sets[rev_rel]
        diff_set = full_set - meta_neighs
        if len(meta_neighs) == 0 or len(diff_set) == 0:
            return None, None, None
        chain = (node, random.choice(list(meta_neighs)))
        neg_chain = (node, random.choice(list(diff_set)))
        return chain, neg_chain, rels

    def sample_chain(self, length, start_mode):
        rel = random.choice(self.relations[start_mode])
        rel = (start_mode, rel[-1], rel[0])
        if len(self.adj_lists[rel]) == 0:
            return None, None, None
        node = random.choice(self.adj_lists[rel].keys())
        return self.sample_chain_from_node(length, node, rel)
                
    def sample_chains(self, length, anchor_weights, num_samples):
        sampled = 0
        graph_chains = defaultdict(list)
        neg_chains = defaultdict(list)
        while sampled < num_samples: 
            anchor_mode = anchor_weights.keys()[np.argmax(np.random.multinomial(1, anchor_weights.values()))]
            chain, neg_chain, rels = self.sample_chain(length, anchor_mode)
            if chain is None:
                continue
            graph_chains[rels].append(chain)
            neg_chains[rels].append(neg_chain)
            sampled += 1
        return graph_chains, neg_chains


    def sample_polytree_rootinter(self, length, target_mode, try_out=100):
        num_chains = random.randint(2,length)
        added = 0
        nodes = []
        rels_list = []

        for i in range(num_chains):
            remaining = length-added-num_chains
            if i != num_chains - 1:
                remaining = remaining if remaining == 0 else random.randint(0, remaining)
            added += remaining
            chain_len = 1 + remaining
            if i == 0:
                chain, _, rels = self.sample_chain(chain_len, target_mode)
                try_count = 0 
                while chain is None and try_count <= try_out:
                    chain, _, rels = self.sample_chain(chain_len, target_mode)
                    try_count += 1

                if chain is None:
                    return None, None, None, None, None
                target_node = chain[0]
                nodes.append(chain[-1])
                rels_list.append(tuple([_reverse_relation(rel) for rel in rels[::-1]]))
            else:
                rel = random.choice([r for r in self.relations[target_mode] 
                    if len(self.adj_lists[(target_mode, r[-1], r[0])][target_node]) > 0])
                rel = (target_mode, rel[-1], rel[0])
                chain, _, rels = self.sample_chain_from_node(chain_len, target_node, rel)
                try_count = 0
                while chain is None and try_count <= try_out:
                    chain, _, rels = self.sample_chain_from_node(chain_len, target_node, rel)
                    if chain is None:
                        try_count += 1
                    elif chain[-1] in nodes:
                        chain = None
                if chain is None:
                    return None, None, None, None, None
                nodes.append(chain[-1])
                rels_list.append(tuple([_reverse_relation(rel) for rel in rels[::-1]]))

        for i in range(len(nodes)):
            meta_neighs = self.get_metapath_neighs(nodes[i], rels_list[i])
            if i == 0:
                meta_neighs_inter = meta_neighs
                meta_neighs_union = meta_neighs
            else:
                meta_neighs_inter = meta_neighs_inter.intersection(meta_neighs)
                meta_neighs_union = meta_neighs_union.union(meta_neighs)
        hard_neg_nodes = list(meta_neighs_union-meta_neighs_inter)
        neg_nodes = list(self.full_sets[rels[0]]-meta_neighs_inter)
        if len(neg_nodes) == 0:
            return None, None, None, None, None
        if len(hard_neg_nodes) == 0:
            return None, None, None, None, None

        return target_node, neg_nodes, hard_neg_nodes, tuple(nodes), tuple(rels_list)


    def sample_polytrees_parallel(self, length, thread_samples, threads, try_out=100):
        pool = Pool(threads)
        sample_func = partial(self.sample_polytree, length)
        sizes = [thread_samples for _ in range(threads)]
        results = pool.map(sample_func, sizes)
        polytrees = {}
        neg_polytrees = {}
        hard_neg_polytrees = {}
        for p, n, h in results: 
            polytrees.update(p)
            neg_polytrees.update(n)
            hard_neg_polytrees.updarte(h)
        return polytrees, neg_polytrees, hard_neg_polytrees
        
    def sample_polytrees(self, length, num_samples, try_out=1):
        samples = 0
        polytrees = defaultdict(list)
        neg_polytrees = defaultdict(list)
        hard_neg_polytrees = defaultdict(list)
        while samples < num_samples:
            t, n, h_n, nodes, rels = self.sample_polytree(length, random.choice(self.relations.keys()))
            if t is None:
                continue
            samples += 1
            polytrees[rels].append((t, nodes))
            neg_polytrees[rels].append((n, nodes))
            hard_neg_polytrees[rels].append((h_n, nodes))
        return polytrees, neg_polytrees, hard_neg_polytrees

    """
