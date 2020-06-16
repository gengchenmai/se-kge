from argparse import ArgumentParser

from netquery.utils import *
from netquery.dbgeo.data_utils import load_graph, json_load
from netquery.data_utils import load_queries_by_formula, load_test_queries_by_formula, load_queries, pickle_load
from netquery.model import QueryEncoderDecoder
from netquery.train_helpers import run_train, run_eval
from netquery.trainer import *

from torch import optim
import numpy as np


# define arguments
parser = make_args_parser()
args = parser.parse_args()

# cuda device
args.device = detect_cuda_device(args.device)

print("Loading graph data..")
'''
graph: a Graph() object
feature_modules: a dict of embedding matrix by node type, each embed matrix shape: [num_ent_by_type + 2, embed_dim]
node_maps: a dict()
    key: type, 5 types: function, sideeffects, protein, disease, drug
    value: dict():
        key: global node id
        value: local node id for this type
'''
graph, feature_modules, node_maps = load_graph(args.data_dir, args.embed_dim)
if feature_modules is not None:
	graph.features = cudify(feature_modules, node_maps, device = args.device)
out_dims = {mode:args.embed_dim for mode in graph.relations}




trainer = Trainer(args, graph, feature_modules, node_maps, out_dims, 
        console = True)

trainer.logger.info("All argusment:")
for arg in vars(args):
    trainer.logger.info("{}: {}".format(arg, getattr(args, arg)))

# load 1-d edge query
trainer.load_edge_data()
# load multi edge query
trainer.load_multi_edge_query_data(load_geo_query = False)

if args.geo_train:
	# load multi edge geographic query
	trainer.load_multi_edge_query_data(load_geo_query = True)



# load model
if args.load_model:
	trainer.load_model()

# train NN model
trainer.train()

# trainer.args.eval_general = True
# trainer.args.eval_log = False
# if trainer.args.geo_train:
#     trainer.args.eval_geo_log = False
# trainer.eval_model(flag="VALID")
# trainer.eval_model(flag="TEST")

trainer.logger.info("geo_info: {}".format(trainer.args.geo_info))
trainer.logger.info("lr: {:f}".format(trainer.args.lr))
trainer.logger.info("freq: {:d}".format(trainer.args.freq))
trainer.logger.info("max_radius: {:f}".format(trainer.args.max_radius))
trainer.logger.info("min_radius: {:f}".format(trainer.args.min_radius))
trainer.logger.info("num_hidden_layer: {:d}".format(trainer.args.num_hidden_layer))
trainer.logger.info("hidden_dim: {:d}".format(trainer.args.hidden_dim))
trainer.logger.info("embed_dim: {}".format(trainer.args.embed_dim))