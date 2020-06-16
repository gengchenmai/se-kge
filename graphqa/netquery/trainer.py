from argparse import ArgumentParser

from netquery.utils import *
# from netquery.data_utils import load_graph, json_load
from netquery.data_utils import load_queries_by_formula, load_test_queries_by_formula, load_queries, pickle_load, pickle_dump
from netquery.model import QueryEncoderDecoder, SpatialSemanticLiftingEncoderDecoder
from netquery.train_helpers import run_train, run_eval, run_train_spa_sem_lift, run_eval_spa_sem_lift

import torch
from torch import optim
import numpy as np
from collections import defaultdict
from os import path

def make_args_parser():
    parser = ArgumentParser()
    # dir
    parser.add_argument("--data_dir", type=str, default="./bio_data/")
    parser.add_argument("--model_dir", type=str, default="./")
    parser.add_argument("--log_dir", type=str, default="./")

    parser.add_argument("--task", type=str, default="qa")
    parser.add_argument("--spa_sem_lift_loss_weight", type=float, default=1.0,
        help='the weight assigned to semantic lifting object in loss')
    

    # model
    parser.add_argument("--embed_dim", type=int, default=128,
        help='node embedding dim')
    parser.add_argument("--depth", type=int, default=0,
        help='the depth of node embedding encoder')
    parser.add_argument("--decoder", type=str, default="bilinear",
        help='the metapath projection operator type')
    parser.add_argument("--inter_decoder", type=str, default="mean",
        help='the intersection operator type')
    parser.add_argument("--use_relu", action='store_true',
        help='whether we use the RelU to compute the embedding stack for attention, default')

    # data
    parser.add_argument("--kg_train", action='store_true',
        help='Whether we use the full KG (x-inter) to train the model')
    parser.add_argument("--geo_train", action='store_true',
        help='Whether we use the geographic queries to train and evaluate the model')
    parser.add_argument("--max_arity", type=int, default=0,
        help='the maximum arity of x-inter query we use to train')

    # attention
    parser.add_argument("--inter_decoder_atten_type", type=str, default='concat',
        help='the type of the intersection operator attention')
    parser.add_argument("--inter_decoder_atten_act", type=str, default='leakyrelu',
        help='the activation function of the intersection operator attention, see GAT paper Equ 3')
    parser.add_argument("--inter_decoder_atten_f_act", type=str, default='sigmoid',
        help='the final activation function of the intersection operator attention, see GAT paper Equ 6')
    parser.add_argument("--inter_decoder_atten_num", type=int, default=0,
        help='the number of the intersection operator attention')
    parser.add_argument("--use_inter_node", action='store_true',
        help='Whether we use the True nodes in the intersection attention as the query embedding to train the QueryEncoderDecoder, without the flag mean False')

    # space encoder
    parser.add_argument("--geo_info", type=str, default='geo',
        help='the type of geographic information, geo (geographic coordinate), or proj (projection coordinate)')
    parser.add_argument("--spa_enc_type", type=str, default='no',
        help='the type of space encoding method')
    parser.add_argument("--enc_agg_type", type=str, default='add',
        help='the method to integrate space embedding with entity embedding, e.g. add, min, max, mean')
    parser.add_argument("--spa_embed_dim", type=int, default=64,
        help='Point Spatial relation embedding dim')
    parser.add_argument("--freq", type=int, default=16,
        help='The number of frequency used in the space encoder')
    parser.add_argument("--max_radius", type=float, default=360,
        help='The maximum frequency in the space encoder')
    parser.add_argument("--min_radius", type=float, default=0.0001,
        help='The minimum frequency in the space encoder')
    parser.add_argument("--spa_f_act", type=str, default='sigmoid',
        help='The final activation function used by spatial relation encoder')
    parser.add_argument("--freq_init", type=str, default='geometric',
        help='The frequency list initialization method')
    parser.add_argument("--spa_enc_use_postmat", type=str, default='T',
        help='whether to use post matrix in spa_enc')
    parser.add_argument("--spa_enc_embed_norm", type=str, default='F',
        help='whether to do position embedding normalization in spa_enc')

    # rbf param
    parser.add_argument("--anchor_sample_method", type=str, default='fromid2geo',
        help='the type of RBF anchor pts sampling method, e.g., fromid2geo, random')
    parser.add_argument("--num_rbf_anchor_pts", type=int, default=100,
        help='The number of RBF anchor points used in the "rbf" space encoder')
    parser.add_argument("--rbf_kernal_size", type=float, default=10e2,
        help='The RBF kernal size in the "rbf" space encoder')
    # parser.add_argument("--rbf_kernal_size_ratio", type=float, default=0,
    #     help='The RBF kernal size ratio in the relative "rbf" space encoder')


    # ffn
    parser.add_argument("--num_hidden_layer", type=int, default=3,
        help='The number of hidden layer in feedforward NN in the (global) space encoder')
    parser.add_argument("--hidden_dim", type=int, default=128,
        help='The hidden dimention in feedforward NN in the (global) space encoder')
    parser.add_argument("--use_layn", type=str, default="F",
        help='use layer normalization or not in feedforward NN in the (global) space encoder')
    parser.add_argument("--skip_connection", type=str, default="F",
        help='skip connection or not in feedforward NN in the (global) space encoder')
    parser.add_argument("--dropout", type=float, default=0.5,
        help='The dropout rate used in all fully connected layer')

    # parser.add_argument("--place_decoder_type", type=str, default='no',
    #     help='the type of place decoder method')
    # parser.add_argument("--sc_radius", type=float, default=100,
    #     help='the buffer radius of spatial context')
    # parser.add_argument("--sc_topn", type=int, default=10,
    #     help='the maximum number of entities in the spatial context')
    # parser.add_argument("--sc_num_neg", type=int, default=10,
    #     help='number of negative samples')

    # train
    parser.add_argument("--opt", type=str, default="adam")
    parser.add_argument("--lr", type=float, default=0.01,
        help='learning rate')
    parser.add_argument("--max_iter", type=int, default=100000000,
        help='the maximum iterator for model converge')
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--cuda", action='store_true')
    parser.add_argument("--device", type=str, default="cpu",
        help='cpu or cuda')
    parser.add_argument("--max_burn_in", type=int, default=100000,
        help='the maximum iterator for 1-chain edge traing for edge converge')
    parser.add_argument("--tol", type=float, default=0.0001)
    parser.add_argument("--inter_weight", type=float, default=0.005,
        help='the weight assigned to inter type query in loss')
    parser.add_argument("--path_weight", type=float, default=0.01,
        help='the weight assigned to path type query except 1-chain in loss')

    # eval
    parser.add_argument("--val_every", type=int, default=5000)

    # load old model
    parser.add_argument("--load_model", action='store_true')

    parser.add_argument("--edge_conv", action='store_true')

    parser.add_argument("--eval_general", action='store_true',
        help='eval the model against general queries')

    parser.add_argument("--eval_log", action='store_true',
        help='save the pkl file about the AUC per formula in general queries')

    parser.add_argument("--eval_geo_log", action='store_true',
        help='save the pkl file about the AUC per formula in geo queries')

    

    return parser



def make_args_combine(args):
    if args.use_relu:
        use_relu_flag = "T"
    else:
        use_relu_flag = "F"

    if args.use_inter_node:
        use_inter_node_flag = "T"
    else:
        use_inter_node_flag = "F"

    if args.kg_train:
        kg_train_flag = "T"
    else:
        kg_train_flag = "F"

    if args.geo_train:
        geo_train_flag = "T"
    else:
        geo_train_flag = "F"

    if args.task == "qa":
        args_task = ""
        args_inter = "-{inter_decoder:s}-{use_relu_flag:s}-{inter_decoder_atten_type:s}-{inter_decoder_atten_act:s}-{inter_decoder_atten_f_act:s}-{inter_decoder_atten_num:d}-{use_inter_node_flag:s}-{kg_train_flag:s}-{max_arity:d}".format(
            inter_decoder=args.inter_decoder,
            use_relu_flag=use_relu_flag,
            inter_decoder_atten_type=args.inter_decoder_atten_type,
            inter_decoder_atten_act=args.inter_decoder_atten_act,
            inter_decoder_atten_f_act=args.inter_decoder_atten_f_act,
            inter_decoder_atten_num=args.inter_decoder_atten_num,
            use_inter_node_flag = use_inter_node_flag,
            kg_train_flag=kg_train_flag,
            max_arity=args.max_arity)
    else:
        args_task = "-{task:s}-{spa_sem_lift_loss_weight:f}".format(
            task = args.task,
            spa_sem_lift_loss_weight = args.spa_sem_lift_loss_weight)
        args_inter = ""

    if args.spa_enc_type == "no":
        args_spa_enc = "{spa_enc_type:s}".format(
            spa_enc_type = args.spa_enc_type
            )
        args_rbf = ""
        args_ffn = ""
    else:
        args_spa_enc = "{geo_info:s}-{spa_enc_type:s}-{enc_agg_type:s}-{spa_embed_dim:d}-{freq:d}-{max_radius:f}-{min_radius:f}-{spa_f_act:s}-{freq_init:s}-{spa_enc_use_postmat:s}-{spa_enc_embed_norm:s}".format(
                geo_info = args.geo_info,
                spa_enc_type = args.spa_enc_type,
                enc_agg_type = args.enc_agg_type,
                spa_embed_dim = args.spa_embed_dim,
                freq = args.freq,
                max_radius = args.max_radius,
                min_radius = args.min_radius,
                spa_f_act = args.spa_f_act,
                freq_init = args.freq_init,
                spa_enc_use_postmat = args.spa_enc_use_postmat,
                spa_enc_embed_norm = args.spa_enc_embed_norm
                )

        args_rbf = "{anchor_sample_method:s}-{num_rbf_anchor_pts:d}-{rbf_kernal_size:.1f}".format(
                anchor_sample_method=args.anchor_sample_method,
                num_rbf_anchor_pts=args.num_rbf_anchor_pts,
                rbf_kernal_size=args.rbf_kernal_size
                )

        args_ffn = "-{num_hidden_layer:d}-{hidden_dim:d}-{use_layn:s}-{skip_connection:s}-{dropout:.2f}".format(
                num_hidden_layer = args.num_hidden_layer, 
                hidden_dim = args.hidden_dim, 
                use_layn = args.use_layn, 
                skip_connection = args.skip_connection,
                dropout = args.dropout)

    # args_combine = "/{data:s}{args_task:s}-{depth:d}-{args_spa_enc:s}-{args_rbf:s}-{args_ffn:s}-{embed_dim:d}-{lr:f}-{batch_size:d}-{inter_weight:f}-{path_weight:f}-{decoder:s}{args_inter:s}".format(
    #         data=args.data_dir.strip().split("/")[-2],
    #         args_task = args_task,
    #         depth=args.depth,

    #         args_spa_enc=args_spa_enc,
    #         args_rbf=args_rbf,
    #         args_ffn=args_ffn,

    #         embed_dim=args.embed_dim,
    #         lr=args.lr,
    #         batch_size=args.batch_size,
    #         inter_weight=args.inter_weight,
    #         path_weight=args.path_weight,
    #         decoder=args.decoder,
    #         args_inter = args_inter,
    #         geo_train_flag=geo_train_flag)
    args_combine = "/{data:s}{args_task:s}-{depth:d}-{args_spa_enc:s}-{args_rbf:s}-{args_ffn:s}-{embed_dim:d}-{batch_size:d}-{inter_weight:f}-{path_weight:f}-{decoder:s}{args_inter:s}".format(
            data=args.data_dir.strip().split("/")[-2],
            args_task = args_task,
            depth=args.depth,

            args_spa_enc=args_spa_enc,
            args_rbf=args_rbf,
            args_ffn=args_ffn,

            embed_dim=args.embed_dim,
            # lr=args.lr,
            batch_size=args.batch_size,
            inter_weight=args.inter_weight,
            path_weight=args.path_weight,
            decoder=args.decoder,
            args_inter = args_inter,
            geo_train_flag=geo_train_flag)

    return args_combine

class Trainer():
    '''
    Trainer
    '''
    def __init__(self, args, graph, feature_modules, node_maps, out_dims, 
        console = True):
        '''
        Args:
            graph: a Graph() object
            feature_modules: a dict of embedding matrix by node type, each embed matrix shape: [num_ent_by_type + 2, embed_dim]
            node_maps: a dict()
                key: type, 5 types: function, sideeffects, protein, disease, drug
                value: dict():
                    key: global node id
                    value: local node id for this type
            out_dims: 
            
        '''
        self.args = args
        self.graph = graph
        self.feature_modules = feature_modules
        self.node_maps = node_maps
        self.out_dims = out_dims

        self.train_queries = None
        self.val_queries = {"full_neg" : defaultdict(lambda : defaultdict(list)), 
                            "one_neg" : defaultdict(lambda : defaultdict(list))}
        self.test_queries = {"full_neg" : defaultdict(lambda : defaultdict(list)), 
                            "one_neg" : defaultdict(lambda : defaultdict(list))}

        # geographic val/test query
        # val_queries_geo and test_queries_geo DO NOT have 1-chain query
        self.train_queries_geo = None
        self.val_queries_geo = {"full_neg" : defaultdict(lambda : defaultdict(list)), 
                            "one_neg" : defaultdict(lambda : defaultdict(list))}
        self.test_queries_geo = {"full_neg" : defaultdict(lambda : defaultdict(list)), 
                            "one_neg" : defaultdict(lambda : defaultdict(list))}


        if args.spa_enc_type == "no" and args.embed_dim == 0:
            raise Exception("You can not set embed_dim = 0 when you do not want to use spa_enc!")
        if args.enc_agg_type in ["add", "min", "max", "mean"] and args.spa_enc_type != "no" and args.embed_dim > 0:
            # here we assume spa_embed_dim == embed_dim 
            assert args.spa_embed_dim == args.embed_dim
        '''
        id2geo: a dict()
                key: entity/node id
                value: [longitude, lantitude]
        '''
        self.id2geo, self.id2extent = self.load_id2geo()
        id2geo = self.id2geo

        self.args_combine = make_args_combine(args) #+ ".L2"

        self.log_file = args.log_dir + self.args_combine + ".log"
        self.model_file = args.model_dir + self.args_combine + ".pth"

        self.set_evel_file_name()


        self.logger = setup_logging(self.log_file, console = console, filemode='a')

        # self.logger.info("All argusment:")
        # for arg in vars(args):
        #     self.logger.info("{}: {}".format(arg, getattr(args, arg)))


        # construct the GraphSAGE style node embedding encoder
        # enc: its forward(nodes, mode) will return node embedding metrix of shape [embed_dim, num_ent]
        # construct position encoder
        if args.spa_enc_type != "no":
            self.spa_enc = get_spa_encoder(args = args, 
                    geo_info = args.geo_info,
                    spa_enc_type = args.spa_enc_type, 
                    id2geo = id2geo, 
                    spa_embed_dim = args.spa_embed_dim, 
                    coord_dim = 2,
                    anchor_sample_method = args.anchor_sample_method, 
                    num_rbf_anchor_pts = args.num_rbf_anchor_pts, 
                    rbf_kernal_size = args.rbf_kernal_size, 
                    frequency_num = args.freq, 
                    max_radius = args.max_radius, 
                    min_radius = args.min_radius, 
                    f_act = args.spa_f_act, 
                    freq_init = args.freq_init, 
                    use_postmat = args.spa_enc_use_postmat,
                    device = args.device)
        else:
            self.spa_enc = None

        if args.spa_enc_embed_norm == "T":
            spa_enc_embed_norm = True
        else:
            spa_enc_embed_norm = False

        self.enc = get_encoder(depth = args.depth, 
                    graph = graph, 
                    out_dims = out_dims, 
                    feature_modules = feature_modules, 
                    geo_info = args.geo_info,
                    spa_enc_type = args.spa_enc_type, 
                    spa_enc_embed_norm = spa_enc_embed_norm,
                    id2geo = self.id2geo, 
                    id2extent = self.id2extent,
                    spa_enc = self.spa_enc, 
                    enc_agg_type = args.enc_agg_type,
                    task = args.task,
                    device = args.device)

        if args.enc_agg_type == "concat":
            # assert args.spa_enc_type != "no" and args.embed_dim > 0
            model_out_dims = {mode: out_dims[mode]+args.spa_embed_dim for mode in out_dims}
        elif args.enc_agg_type in ["add", "min", "max", "mean"]:
            if args.embed_dim > 0:
                # if we have feature encoder, 
                model_out_dims = out_dims
            else:
                # if we do not have feature encoder, we must have position encoder
                model_out_dims = {mode:args.spa_embed_dim for mode in out_dims}

        else:
            raise Exception("enc_agg_type not support!")

        self.model_out_dims = model_out_dims

        self.dec = get_metapath_decoder(graph, 
                    out_dims = model_out_dims, 
                    decoder = args.decoder, 
                    feat_dims = out_dims, 
                    spa_embed_dim = args.spa_embed_dim,
                    enc_agg_type = args.enc_agg_type)
        if args.task == "qa":
            self.inter_dec = get_intersection_decoder(graph, model_out_dims, args.inter_decoder, args.use_relu)
            if args.inter_decoder_atten_num > 0:
                self.inter_attn = get_intersection_attention(model_out_dims, 
                                                        inter_decoder_atten_type=args.inter_decoder_atten_type, 
                                                        inter_decoder_atten_num=args.inter_decoder_atten_num, 
                                                        inter_decoder_atten_act = args.inter_decoder_atten_act, 
                                                        inter_decoder_atten_f_act = args.inter_decoder_atten_f_act)
            else:
                self.inter_attn = None

            self.enc_dec = QueryEncoderDecoder(graph, self.enc, self.dec, self.inter_dec, self.inter_attn, args.use_inter_node)
        elif args.task == "spa_sem_lift":
            self.inter_dec = None
            self.inter_attn = None

            self.enc_dec = SpatialSemanticLiftingEncoderDecoder(graph, self.enc, self.dec, self.inter_dec, self.inter_attn, args.use_inter_node)


        
        self.enc_dec.to(args.device)

        if args.opt == "sgd":
            self.optimizer = optim.SGD(filter(lambda p : p.requires_grad, self.enc_dec.parameters()), lr=args.lr, momentum=0)
        elif args.opt == "adam":
            self.optimizer = optim.Adam(filter(lambda p : p.requires_grad, self.enc_dec.parameters()), lr=args.lr)

        print("create model from {}".format(self.args_combine + ".pth"))
        self.logger.info("Save file at {}".format(self.args_combine + ".pth"))

    def load_id2geo(self):
        # load place geographic coordinates
        if self.args.spa_enc_type == 'no':
            id2geo = None
            id2extent = None
        else:
            if self.args.geo_info == "geo":
                id2geo = pickle_load(self.args.data_dir + "/id2geo.pkl")
                id2extent = None
            elif self.args.geo_info == "proj":
                id2geo = pickle_load(self.args.data_dir + "/id2geo_proj.pkl")
                id2extent = None
            elif self.args.geo_info == "projbbox":
                id2geo = pickle_load(self.args.data_dir + "/id2geo_proj.pkl")
                id2extent = pickle_load(self.args.data_dir + "/id2extent_proj.pkl")
            elif self.args.geo_info == "projbboxmerge":
                id2geo = pickle_load(self.args.data_dir + "/id2geo_proj.pkl")
                id2extent = pickle_load(self.args.data_dir + "/id2extent_proj_merge.pkl")
            else:
                raise Exception("Unknown geo_info parameters!")
        return id2geo, id2extent



    def load_edge_data(self, load_geo_query = False, test_query_keep_graph=False):
        '''
        just load 1-d query for train/val/test
            train_queries:     train_queries[query_type][formula] = list of query
            val_queries:       val_queries[one_neg/full_neg][query_type][formula] = list of query
            test_queries:      test_queries[one_neg/full_neg][query_type][formula] = list of query
        '''
        if load_geo_query:
            file_postfix = "-geo"
        else:
            file_postfix = ""

        print("Loading edge data..")
        
        print("Loading training edge data..")
        train_queries = load_queries_by_formula(self.args.data_dir + "/train_edges{:s}.pkl".format(file_postfix))
        print("Loading validation edge  data..")
        val_queries = load_test_queries_by_formula(self.args.data_dir + "/val_edges{:s}.pkl".format(file_postfix), keep_graph = test_query_keep_graph)
        print("Loading testing edge data..")
        test_queries = load_test_queries_by_formula(self.args.data_dir + "/test_edges{:s}.pkl".format(file_postfix), keep_graph = test_query_keep_graph)

        

        if load_geo_query:
            self.train_queries_geo = train_queries
            self.val_queries_geo = val_queries
            self.test_queries_geo = test_queries
        else:
            self.train_queries = train_queries
            self.val_queries = val_queries
            self.test_queries = test_queries

    def load_multi_edge_query_data(self, load_geo_query = False, test_query_keep_graph=False):
        '''
        Load multi edge query for train/val/test
        '''
        if load_geo_query:
            file_postfix = "-geo"
            train_queries = self.train_queries
            val_queries = self.val_queries_geo
            test_queries = self.test_queries_geo
        else:
            file_postfix = ""
            train_queries = self.train_queries
            val_queries = self.val_queries
            test_queries = self.test_queries

        
        print("Loading {:s} query data..".format(file_postfix))
        for i in range(2,4):
            # if not args.kg_train:
            print("Loading training {:s} {:d} triple data..".format(file_postfix, i))
            train_queries_file = self.args.data_dir + "/train_queries_{:d}{:s}.pkl".format(i,file_postfix)
            if path.exists(train_queries_file):
                train_queries.update(load_queries_by_formula(train_queries_file))
            else:
                print("{} no exist!".format(train_queries_file))

            print("Loading validate {:s} {:d} triple data..".format(file_postfix, i))
            val_queries_file = self.args.data_dir + "/val_queries_{:d}{:s}.pkl".format(i,file_postfix)
            if path.exists(val_queries_file):
                i_val_queries = load_test_queries_by_formula(val_queries_file, keep_graph = test_query_keep_graph)
                val_queries["one_neg"].update(i_val_queries["one_neg"])
                val_queries["full_neg"].update(i_val_queries["full_neg"])
            else:
                print("{} no exist!".format(val_queries_file))

            print("Loading testing {:s} {:d} triple data..".format(file_postfix, i))
            test_queries_file = self.args.data_dir + "/test_queries_{:d}{:s}.pkl".format(i,file_postfix)
            if path.exists(test_queries_file):
                i_test_queries = load_test_queries_by_formula(test_queries_file, keep_graph = test_query_keep_graph)
                test_queries["one_neg"].update(i_test_queries["one_neg"])
                test_queries["full_neg"].update(i_test_queries["full_neg"])
            else:
                print("{} no exist!".format(test_queries_file))


        if self.args.kg_train:
            print("Loading x-inter train {:s} query data..".format(file_postfix))
            if self.args.max_arity < 4:
                raise Exception("for full KG train, arity should be >= 4") 
            for arity in range(4, self.args.max_arity+1):
                print("Loading training {:s} {:d}-inter query data..".format(file_postfix, arity))
                train_queries.update(load_queries_by_formula(self.args.data_dir + "/train_inter_queries_{:d}{:s}.pkl".format(arity,file_postfix)))


    def load_model(self):
        self.logger.info("Load model from {}".format(self.model_file))
        self.enc_dec.load_state_dict(torch.load(self.model_file))

    def set_evel_file_name(self):

        self.test_auc_detail_log_file = self.args.log_dir + self.args_combine + "--fm_auc_Test.pkl"
        self.test_geo_auc_detail_log_file = self.args.log_dir + self.args_combine + "--fm_auc_geo_Test.pkl"
        self.val_auc_detail_log_file = self.args.log_dir + self.args_combine + "--fm_auc_Valid.pkl"
        self.val_geo_auc_detail_log_file = self.args.log_dir + self.args_combine + "--fm_auc_geo_Valid.pkl"

        self.test_prec_detail_log_file = self.args.log_dir + self.args_combine + "--fm_prec_Test.pkl"
        self.test_geo_prec_detail_log_file = self.args.log_dir + self.args_combine + "--fm_prec_geo_Test.pkl"
        self.val_prec_detail_log_file = self.args.log_dir + self.args_combine + "--fm_prec_Valid.pkl"
        self.val_geo_prec_detail_log_file = self.args.log_dir + self.args_combine + "--fm_prec_geo_Valid.pkl"



    def eval_model(self, flag="TEST"):
        self.test_auc_detail_log_file = self.args.log_dir + self.args_combine + "--fm_auc_Test.pkl"
        self.test_geo_auc_detail_log_file = self.args.log_dir + self.args_combine + "--fm_auc_geo_Test.pkl"
        self.val_auc_detail_log_file = self.args.log_dir + self.args_combine + "--fm_auc_Valid.pkl"
        self.val_geo_auc_detail_log_file = self.args.log_dir + self.args_combine + "--fm_auc_geo_Valid.pkl"

        self.test_prec_detail_log_file = self.args.log_dir + self.args_combine + "--fm_prec_Test.pkl"
        self.test_geo_prec_detail_log_file = self.args.log_dir + self.args_combine + "--fm_prec_geo_Test.pkl"
        self.val_prec_detail_log_file = self.args.log_dir + self.args_combine + "--fm_prec_Valid.pkl"
        self.val_geo_prec_detail_log_file = self.args.log_dir + self.args_combine + "--fm_prec_geo_Valid.pkl"


        if flag == "TEST":
            queries = self.test_queries
            queries_geo = self.test_queries_geo

            auc_detail_log_file = self.test_auc_detail_log_file
            geo_auc_detail_log_file = self.test_geo_auc_detail_log_file

            prec_detail_log_file = self.test_prec_detail_log_file
            geo_prec_detail_log_file = self.test_geo_prec_detail_log_file

            tag = "Test"
        elif flag == "VALID":
            queries = self.val_queries
            queries_geo = self.val_queries_geo

            auc_detail_log_file = self.val_auc_detail_log_file
            geo_auc_detail_log_file = self.val_geo_auc_detail_log_file

            prec_detail_log_file = self.val_prec_detail_log_file
            geo_prec_detail_log_file = self.val_geo_prec_detail_log_file

            tag = "Valid"
        
        if self.args.eval_general:
            if self.args.eval_log:
                v, aprs, qtype2fm_auc, qtype2fm_q_prec = run_eval(self.enc_dec, queries, 0, self.logger, eval_detail_log = True)
                # detail_log_file = self.args.log_dir + self.args_combine + "--fmauc_{}.pkl".format(tag)
                pickle_dump(qtype2fm_auc, auc_detail_log_file)
                pickle_dump(qtype2fm_q_prec, prec_detail_log_file)
            else:
                v, aprs = run_eval(self.enc_dec, queries, 0, self.logger)
            self.logger.info("{} macro-averaged AUC: {:f}, APR: {:f}".format(tag, np.mean(v.values()), np.mean(aprs.values())))
        if self.args.geo_train:
            if self.args.eval_geo_log:
                v_geo, aprs_geo, qtype2fm_auc_geo, qtype2fm_q_prec_geo = run_eval(self.enc_dec, queries_geo, 0, self.logger, geo_train = True, eval_detail_log = True)
                # geo_detail_log_file = self.args.log_dir + self.args_combine + "--fmauc_geo_{}.pkl".format(tag)
                pickle_dump(qtype2fm_auc_geo, geo_auc_detail_log_file)
                pickle_dump(qtype2fm_q_prec_geo, geo_prec_detail_log_file)
            else:
                v_geo, aprs_geo = run_eval(self.enc_dec, queries_geo, 0, self.logger, geo_train = True)
            self.logger.info("GEO {} macro-averaged AUC: {:f}, APR: {:f}".format(tag, np.mean(v_geo.values()), np.mean(aprs_geo.values())))

        

        # if flag == "TEST":
        #     if self.args.eval_log:
        #         v, aprs, qtype2fm_auc = run_eval(self.enc_dec, self.test_queries, 0, self.logger, eval_detail_log = True)
        #         self.detail_log_file = self.args.log_dir + self.args_combine + "--fmauc.pkl"
        #         pickle_dump(qtype2fm_auc, self.detail_log_file)
        #     else:
        #         v, aprs = run_eval(self.enc_dec, self.test_queries, 0, self.logger)
        #     self.logger.info("Test macro-averaged AUC: {:f}, APR: {:f}".format(np.mean(v.values()), np.mean(aprs.values())))
        #     if self.args.geo_train:
        #         if self.args.eval_geo_log:
        #             v_geo, aprs_geo, qtype2fm_auc_geo = run_eval(self.enc_dec, self.test_queries_geo, 0, self.logger, geo_train = True, eval_detail_log = True)
        #             self.detail_log_file = self.args.log_dir + self.args_combine + "--fmauc_geo.pkl"
        #             pickle_dump(qtype2fm_auc_geo, self.detail_log_file)
        #         else:
        #             v_geo, aprs_geo = run_eval(self.enc_dec, self.test_queries_geo, 0, self.logger, geo_train = True)
        #         self.logger.info("GEO Test macro-averaged AUC: {:f}, APR: {:f}".format(np.mean(v_geo.values()), np.mean(aprs_geo.values())))
        # elif flag == "VALID":
        #     v, aprs = run_eval(self.enc_dec, self.val_queries, 0, self.logger)
        #     self.logger.info("Valid macro-averaged AUC: {:f}, APR: {:f}".format(np.mean(v.values()), np.mean(aprs.values())))
        #     if self.args.geo_train:
        #         v_geo, aprs_geo = run_eval(self.enc_dec, self.val_queries_geo, 0, self.logger, geo_train = True)
        #         self.logger.info("GEO Valid macro-averaged AUC: {:f}, APR: {:f}".format(np.mean(v_geo.values()), np.mean(aprs_geo.values())))
    
    def train(self):
        run_train(model = self.enc_dec, 
            optimizer = self.optimizer, 
            train_queries = self.train_queries, 
            val_queries = self.val_queries, 
            test_queries = self.test_queries, 
            logger = self.logger,
            max_burn_in = self.args.max_burn_in, 
            batch_size = self.args.batch_size, 
            log_every=100, 
            val_every= self.args.val_every, 
            tol = self.args.tol,
            max_iter = self.args.max_iter, 
            inter_weight = self.args.inter_weight, 
            path_weight = self.args.path_weight, 
            model_file = self.model_file, 
            edge_conv = self.args.edge_conv,
            geo_train = self.args.geo_train, 
            val_queries_geo = self.val_queries_geo, 
            test_queries_geo = self.test_queries_geo)
        torch.save(self.enc_dec.state_dict(), self.model_file)


    def train_spa_sem_lift(self):
        run_train_spa_sem_lift(model = self.enc_dec, 
            optimizer = self.optimizer, 
            train_queries = self.train_queries, 
            val_queries = self.val_queries, 
            test_queries = self.test_queries, 
            logger = self.logger,
            max_burn_in = self.args.max_burn_in, 
            batch_size = self.args.batch_size, 
            log_every=100, 
            val_every= self.args.val_every, 
            tol = self.args.tol,
            max_iter = self.args.max_iter, 
            inter_weight = self.args.inter_weight, 
            path_weight = self.args.path_weight, 
            model_file = self.model_file, 
            edge_conv = self.args.edge_conv,
            geo_train = self.args.geo_train, 
            spa_sem_lift_loss_weight = self.args.spa_sem_lift_loss_weight,
            train_queries_geo = self.train_queries_geo,
            val_queries_geo = self.val_queries_geo, 
            test_queries_geo = self.test_queries_geo)
        torch.save(self.enc_dec.state_dict(), self.model_file)

    def eval_spa_sem_lift_model(self, flag="TEST"):
        self.test_auc_detail_log_file = self.args.log_dir + self.args_combine + "--fm_auc_Test.pkl"
        self.test_geo_auc_detail_log_file = self.args.log_dir + self.args_combine + "--fm_auc_geo_Test.pkl"
        self.val_auc_detail_log_file = self.args.log_dir + self.args_combine + "--fm_auc_Valid.pkl"
        self.val_geo_auc_detail_log_file = self.args.log_dir + self.args_combine + "--fm_auc_geo_Valid.pkl"

        self.test_prec_detail_log_file = self.args.log_dir + self.args_combine + "--fm_prec_Test.pkl"
        self.test_geo_prec_detail_log_file = self.args.log_dir + self.args_combine + "--fm_prec_geo_Test.pkl"
        self.val_prec_detail_log_file = self.args.log_dir + self.args_combine + "--fm_prec_Valid.pkl"
        self.val_geo_prec_detail_log_file = self.args.log_dir + self.args_combine + "--fm_prec_geo_Valid.pkl"


        if flag == "TEST":
            queries = self.test_queries
            queries_geo = self.test_queries_geo

            auc_detail_log_file = self.test_auc_detail_log_file
            geo_auc_detail_log_file = self.test_geo_auc_detail_log_file

            prec_detail_log_file = self.test_prec_detail_log_file
            geo_prec_detail_log_file = self.test_geo_prec_detail_log_file

            tag = "Test"
        elif flag == "VALID":
            queries = self.val_queries
            queries_geo = self.val_queries_geo

            auc_detail_log_file = self.val_auc_detail_log_file
            geo_auc_detail_log_file = self.val_geo_auc_detail_log_file

            prec_detail_log_file = self.val_prec_detail_log_file
            geo_prec_detail_log_file = self.val_geo_prec_detail_log_file

            tag = "Valid"
        
        if self.args.eval_general:
            if self.args.eval_log:
                v, aprs, qtype2fm_auc, qtype2fm_q_prec = run_eval_spa_sem_lift(self.enc_dec, queries, 0, self.logger, 
                                        eval_detail_log = True, do_spa_sem_lift = False)
                # detail_log_file = self.args.log_dir + self.args_combine + "--fmauc_{}.pkl".format(tag)
                pickle_dump(qtype2fm_auc, auc_detail_log_file)
                pickle_dump(qtype2fm_q_prec, prec_detail_log_file)
            else:
                v, aprs = run_eval_spa_sem_lift(self.enc_dec, queries, 0, self.logger, do_spa_sem_lift = False)
            self.logger.info("{} macro-averaged AUC: {:f}, APR: {:f}".format(tag, np.mean(v.values()), np.mean(aprs.values())))
        if self.args.geo_train:
            if self.args.eval_geo_log:
                v_geo, aprs_geo, qtype2fm_auc_geo, qtype2fm_q_prec_geo = run_eval_spa_sem_lift(self.enc_dec, queries_geo, 0, self.logger, 
                                        geo_train = True, eval_detail_log = True, do_spa_sem_lift = True)
                # geo_detail_log_file = self.args.log_dir + self.args_combine + "--fmauc_geo_{}.pkl".format(tag)
                pickle_dump(qtype2fm_auc_geo, geo_auc_detail_log_file)
                pickle_dump(qtype2fm_q_prec_geo, geo_prec_detail_log_file)
            else:
                v_geo, aprs_geo = run_eval_spa_sem_lift(self.enc_dec, queries_geo, 0, self.logger, 
                                        geo_train = True, do_spa_sem_lift = True)
            self.logger.info("GEO {} macro-averaged AUC: {:f}, APR: {:f}".format(tag, np.mean(v_geo.values()), np.mean(aprs_geo.values())))





