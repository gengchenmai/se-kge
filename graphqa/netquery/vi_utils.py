import matplotlib
import matplotlib.pyplot as plt
import torch
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as plab
from matplotlib import rcParams
from mpl_toolkits.axes_grid1 import make_axes_locatable

from sklearn.manifold import TSNE
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter

import geopy
from geopy.distance import geodesic

import geopandas as gpd
# from shapely.geometry.polygon import Polygon
# from shapely.geometry.multipolygon import MultiPolygon
from shapely.geometry import Point, Polygon, MultiPolygon
from shapely.ops import transform
import pyproj
from functools import partial

import re
import torch

def get_coords(extent, interval=100000):
    coords = []
#     interval = 100000

    # latitude
    for y in range(int(extent[2]), int(extent[3])+interval, interval):
        coord = []
    #     longitude
        for x in range(int(extent[0]), int(extent[1])+interval, interval):
            coord.append([float(x),float(y)])
        coords.append(coord)
    return coords

def map_id2geo(place2geo):
    x_list = []
    y_list = []
    # iri_list = []
    for iri in place2geo:
        x_list.append(place2geo[iri][0])
        y_list.append(place2geo[iri][1])
        # iri_list.append(iri.replace("http://dbpedia.org/resource/", ""))

    plt.scatter(x_list, y_list, s=1)
    
def visualize_encoder(module, layername, coords, extent, num_ch = 8, img_path=None):
    if layername == "input_emb":
        res = module.make_input_embeds(coords)
        if type(res) == torch.Tensor:
            res = res.data.numpy()
        elif type(res) == np.ndarray:
            res = res
        print res.shape
        res_np = res
    elif layername == "output_emb":       
        res = module.forward(coords)
        embed_dim = res.size()[2]
        res_data = res.data.tolist()
        res_np = np.asarray(res_data)

    num_rows = num_ch/4
 
    plt.figure(figsize=(28, 50))
#     for i in range(embed_dim):
    for i in range(num_ch):
        if num_ch <= 4:
            ax= plt.subplot(1,num_ch ,i+1)
        else:
            ax= plt.subplot(num_rows,4 ,i+1)
        ax.imshow(res_np[:,:,i][::-1, :], extent=extent)
        ax.set_yticklabels([])
        ax.set_xticklabels([])
#         plt.tight_layout()
#         plt.title(i, fontsize=160)
        plt.title(i, fontsize=40)
    fig = plt.gcf()
    plt.show()
    plt.draw()
    if img_path:
        fig.savefig(img_path, dpi=300, bbox_inches='tight')
        
def visualize_embed_cosine(embed, module, layername, coords, extent, centerpt = None, xy_list = None, pt_size = 5, polygon = None, img_path=None):
    '''
    Args:
        embed: (spa_embed_dim, 1)
    '''
    if layername == "input_emb":
        res = module.make_input_embeds(coords)
        if type(res) == torch.Tensor:
            res = res.data.numpy()
        elif type(res) == np.ndarray:
            res = res
        print res.shape
        res_np = res
    elif layername == "output_emb":       
        res = module.forward(coords)
        embed_dim = res.size()[2]
        res_data = res.data.tolist()
        # res_np: (y_list, x_list, spa_embed_dim)
        res_np = np.asarray(res_data)
    
    # mat_cos: (y_list, x_list, 1)
    mat_cos = np.matmul(res_np, embed)
    
    # res_norm: (y_list, x_list, 1)
    res_norm = np.linalg.norm(res_np, ord = 2, axis = -1, keepdims = True)
    
    embed_norm = np.linalg.norm(embed, ord = 2)
    
    mat_cos_norm = mat_cos/res_norm
    mat_cos_norm = mat_cos_norm/embed_norm
    
    # mat_cos_norm: (y_list, x_list)
    mat_cos_norm = np.squeeze(mat_cos_norm, axis=-1)
    
    
    plt.figure(figsize=(15, 10))
    im = plt.imshow(mat_cos_norm, extent=extent)
    
    if polygon is not None:
        x,y = polygon.exterior.xy
        plt.plot(x,y, color='red')
        
    if xy_list is not None:
        plt.scatter(xy_list[0], xy_list[1], s=pt_size, c="red")
        
    if centerpt is not None:
        plt.scatter(centerpt[0], centerpt[1], s=pt_size*10, marker='^', c="white")
    
    
    
    fig = plt.gcf()
    
    fig.colorbar(im)
    
#     ax.set_yticklabels([])
#     ax.set_xticklabels([])
#     plt.tight_layout()
#     plt.title(i, fontsize=160)
#     plt.title(i, fontsize=40)
    
    
    plt.show()
    plt.draw()
    if img_path:
        fig.savefig(img_path, dpi=300, bbox_inches='tight')

def spa_enc_embed_clustering(module, num_cluster, extent, interval, coords, tsne_comp = 4):

    
    # res: (y_list, x_list, spa_embed_dim)
    res = module.forward(coords)

    res_data = res.data.tolist()
    res_np = np.asarray(res_data)

    num_y,num_x,embed_dim = res_np.shape
    embeds = np.reshape(res_np, (num_y*num_x, -1))
    
#     embeds = TSNE(n_components=tsne_comp).fit_transform(embeds)
    
    
    embed_clusters = AgglomerativeClustering(n_clusters=num_cluster, 
                                             affinity="cosine", linkage="complete").fit(embeds)
    cluster_labels = np.reshape(embed_clusters.labels_, (num_y, num_x))
#     plt.matshow(cluster_labels, extent=extent, cmap=plt.get_cmap("terrain"))
#     plt.xticks(np.arange(extent[0], extent[1]+10000, 10000))
#     plt.colorbar()
#     plt.scatter(x, y, s=0.5, c=types,alpha=0.5)
#     fig = plt.gcf()

#     plt.show()
#     plt.draw()
#     img_path = "/home/gengchen/Position_Encoding/spacegraph/img/{}/{}/".format(dataset, model_type) + "/{}_g_spa_enc.png".format(spa_enc)
#     fig.savefig(img_path, dpi=300)
    return embeds, cluster_labels

# def make_enc_map(cluster_labels, num_cluster, extent, margin,
#                  xy_list = None, polygon = None, coords_color = "red", 
#                  colorbar=False, img_path=None, xlabel = None, ylabel = None):
#     cmap = plt.cm.terrain
#     bounds = np.arange(-0.5,num_cluster + 0.5,1)
#     norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
#     plt.figure(figsize=(32, 32))

# #     pt_x_list, pt_y_list = plot_poi_by_type(enc_dec, type2pts, tid)
#     plt.matshow(cluster_labels[::-1, :], extent=extent, cmap=cmap, norm = norm);
#     # plt.colorbar()

#     # We must be sure to specify the ticks matching our target names
#     if colorbar:
#         plt.colorbar(ticks=bounds-0.5)
    
#     if polygon is not None:
#         x,y = polygon.exterior.xy
#         plt.plot(x,y, color='red')
        
#     if xy_list is not None:
#         plt.scatter(xy_list[0], xy_list[1], s=1.5, c=coords_color, alpha=0.5)

# #     plt.scatter(pt_x_list, pt_y_list, s=1.5, c="red", alpha=0.5)
#     plt.xlim(extent[0]-margin, extent[1]+margin)
#     plt.ylim(extent[2]-margin, extent[3]+margin)
#     if xlabel is not None:
#         plt.xlabel(xlabel)
#     if ylabel is not None:
#         plt.ylabel(ylabel)
     
#     plt.xticks(np.arange(extent[0]-margin, extent[1]+margin, 10000))
#     fig = plt.gcf()
#     # fig.suptitle(tid2type[tid])
# #     plt.xlabel(poi_type, fontsize=10)
#     plt.show()
#     if img_path:
#         fig.savefig(img_path, dpi=300)    

def make_enc_map(cluster_labels, num_cluster, extent, margin,
                 xy_list = None, polygon = None, usa_gdf = None, coords_color = "red", 
                 colorbar=False, img_path=None, xlabel = None, ylabel = None):
    cmap = plt.cm.terrain
    bounds = np.arange(-0.5,num_cluster + 0.5,1)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
#     plt.figure(figsize=(32, 32))
    fig, ax = plt.subplots(figsize = (32,32)) 

#     pt_x_list, pt_y_list = plot_poi_by_type(enc_dec, type2pts, tid)
    ax.matshow(cluster_labels[::-1, :], extent=extent, cmap=cmap, norm = norm);
    # plt.colorbar()
    if usa_gdf is not None:

        usa_gdf.geometry.boundary.plot(color=None,edgecolor='k',linewidth = 2,ax=ax)
        # usa_gdf.plot()
        usa_gdf['coords'] = usa_gdf['geometry'].apply(lambda x: x.representative_point().coords[:])
        usa_gdf['coords'] = [coords[0] for coords in usa_gdf['coords']]
        for idx, row in usa_gdf.iterrows():
            plt.annotate(s=row['NAME'], xy=row['coords'],
                         horizontalalignment='center', fontsize = 20, color = "red")

    # We must be sure to specify the ticks matching our target names
    if colorbar:
        ax.colorbar(ticks=bounds-0.5)
    
    if polygon is not None:
        x,y = polygon.exterior.xy
        ax.plot(x,y, color='red')
        
    if xy_list is not None:
        ax.scatter(xy_list[0], xy_list[1], s=3, c=coords_color, alpha=0.5)

#     plt.scatter(pt_x_list, pt_y_list, s=1.5, c="red", alpha=0.5)
    plt.xlim(extent[0]-margin, extent[1]+margin)
    plt.ylim(extent[2]-margin, extent[3]+margin)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
     
    plt.xticks(np.arange(extent[0]-margin, extent[1]+margin, 10000))
    fig = plt.gcf()
    # fig.suptitle(tid2type[tid])
#     plt.xlabel(poi_type, fontsize=10)
    plt.show()
    if img_path:
        fig.savefig(img_path, dpi=300)  

def explode(indata):
    indf = gpd.GeoDataFrame.from_file(indata)
    outdf = gpd.GeoDataFrame(columns=indf.columns)
    for idx, row in indf.iterrows():
        if type(row.geometry) == Polygon:
            outdf = outdf.append(row,ignore_index=True)
        if type(row.geometry) == MultiPolygon:
            multdf = gpd.GeoDataFrame(columns=indf.columns)
            recs = len(row.geometry)
            multdf = multdf.append([row]*recs,ignore_index=True)
            for geom in range(recs):
                multdf.loc[geom,'geometry'] = row.geometry[geom]
            outdf = outdf.append(multdf,ignore_index=True)
    return outdf

def get_pts_in_box(place2geo, extent):
    # extent:(x_min, x_max, y_min, y_max)
    x_list = []
    y_list = []
    for iri in place2geo:
        x, y = place2geo[iri]
        if x > extent[0] and x < extent[1] and y > extent[2] and y < extent[3]:
            x_list.append(x)
            y_list.append(y)
    return (x_list, y_list)


def load_USA_geojson(us_geojson_file):
    '''
    Load USA main land geojson,
    and project it to epsg:2163 projection coordinate system
    '''
    # the data we get from https://raw.githubusercontent.com/johan/world.geo.json/master/countries/USA.geo.json
    # us_geojson_file = "../geokg_collect/dbgeo_code/input/USA.geojson"

    # us_gdf = gpd.read_file(us_geojson_file)
    us_gdf = explode(us_geojson_file)

    project = partial(
        pyproj.transform,
        pyproj.Proj(init='epsg:4326'), # source coordinate system
        pyproj.Proj(init='epsg:2163')) # destination coordinate system

    polygon = us_gdf.loc[5, 'geometry']
    polygon_proj_raw = transform(project, polygon)

    return project, polygon, polygon_proj_raw

def get_projected_mainland_USA_states(us_states_geojson_file):
    us_gdf = gpd.read_file(us_states_geojson_file)
    us_mainland_gdf = us_gdf[(us_gdf.NAME != "Hawaii") & (us_gdf.NAME != "Alaska") & (us_gdf.NAME != "Puerto Rico")]
    us_mainland_proj_gdf = us_mainland_gdf.to_crs({'init': 'epsg:2163'})
    return us_mainland_proj_gdf

def read2idIndexFile(Index2idFilePath):
    '''
    read the entity2id.txt or relation2id.txt
    '''
    indexDict = dict()
    with open(Index2idFilePath, "r") as file:
        count = 0
        for line in file:
            if count > 0:
                line = line.split("\n")[0]
                line_list = line.split("\t")
                indexDict[line_list[0]] = int(line_list[1])
            count += 1
#     print("Load 2id file {}".format(Index2idFilePath))

    return indexDict

def reverse_dict(iri2id):
    id2iri = dict()
    for iri in iri2id:
        id2iri[iri2id[iri]] = iri
    return id2iri


def get_node_mode(node_maps, node_id):
    n_type = None
    for node_type in node_maps:
        if node_id in node_maps[node_type]:
            n_type =  node_type
    
    return n_type




def path_embedding_compute(path_dec, embeds1, rels):
    '''
    embeds1, embeds2 shape: [embed_dim, batch_size]
    rels: a list of triple templates, a n-length metapath
    '''
    # act: [batch_size, embed_dim]
    act = embeds1.t()
    feat_act, pos_act = torch.split(act, 
        [path_dec.feat_dims[rels[0][0]],path_dec.spa_embed_dim], dim=1)
    for i_rel in rels:
        feat_act = feat_act.mm(path_dec.feat_mats[i_rel])
        pos_act = pos_act.mm(path_dec.pos_mats[i_rel])
    #  act: [batch_size, embed_dim]
    act = torch.cat([feat_act, pos_act], dim=1)
    # act = path_dec.cos(act.t(), embeds2)
    # act: [embed_dim, batch_size]
    return act.t()

def query_embedding_compute(enc_dec, formula, queries, source_nodes, do_modelTraining = False):
    '''
    Args:
        enc_dec:
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
        # return enc_dec.path_dec.forward(
        #         enc_dec.enc.forward(source_nodes, formula.target_mode), 
        #         enc_dec.enc.forward([query.anchor_nodes[0] for query in queries], formula.anchor_modes[0]),
        #         formula.rels)

        # I think this is the right way to do x-chain
        reverse_rels = tuple([enc_dec.graph._reverse_relation(formula.rels[i]) for i in range(len(formula.rels)-1, -1, -1)])
        return path_embedding_compute(enc_dec.path_dec,
                enc_dec.enc.forward([query.anchor_nodes[0] for query in queries], formula.anchor_modes[0]),
                reverse_rels)
    elif formula.query_type == "3-inter_chain":
        # target_embeds = enc_dec.enc(source_nodes, formula.target_mode)

        # project the 1st anchor node to target node in rels: (t, p1, a1)
        # embeds1: [embed_dim, batch_size]
        embeds1 = enc_dec.enc([query.anchor_nodes[0] for query in queries], formula.anchor_modes[0])
        # embeds1: [embed_dim, batch_size]
        embeds1 = enc_dec.path_dec.project(embeds1, enc_dec.graph._reverse_relation(formula.rels[0]))

        # project the 2nd anchor node to target node in rels: 
        embeds2 = enc_dec.enc([query.anchor_nodes[1] for query in queries], formula.anchor_modes[1])
        # '3-inter_chain': project a2 to t by following ((t, p2, e1),(e1, p3, a2))
        for i_rel in formula.rels[1][::-1]: # loop the formula.rels[1] in the reverse order
            embeds2 = enc_dec.path_dec.project(embeds2, enc_dec.graph._reverse_relation(i_rel))

        query_intersection, embeds_inter = enc_dec.inter_dec(formula.target_mode, [embeds1, embeds2])
        
        # this is where the attention take affect
        if enc_dec.inter_attn is not None:
            if enc_dec.use_inter_node and do_modelTraining:
                # for 2-inter, 3-inter, 3-inter_chain, the inter node is target node
                # we we can use source_nodes
                query_embeds = enc_dec.graph.features([query.target_node for query in queries], formula.target_mode).t()
                query_intersection = enc_dec.inter_attn(query_embeds, embeds_inter, formula.target_mode)
            else:
                # query_intersection: [embed_dim, batch_size]
                query_intersection = enc_dec.inter_attn(query_intersection, embeds_inter, formula.target_mode)


        # scores = enc_dec.cos(target_embeds, query_intersection)
        return query_intersection
    elif formula.query_type == "3-chain_inter":
        # target_embeds = enc_dec.enc(source_nodes, formula.target_mode)

        # project the 1st anchor node to inter node (e1, p2, a1)
        embeds1 = enc_dec.enc([query.anchor_nodes[0] for query in queries], formula.anchor_modes[0])
        embeds1 = enc_dec.path_dec.project(embeds1, enc_dec.graph._reverse_relation(formula.rels[1][0]))
        # project the 2nd anchor node to inter node (e1, p3, a2)
        embeds2 = enc_dec.enc([query.anchor_nodes[1] for query in queries], formula.anchor_modes[1])
        embeds2 = enc_dec.path_dec.project(embeds2, enc_dec.graph._reverse_relation(formula.rels[1][1]))
        # intersect different inter node (e1) embeddings
        # query_intersection, embeds_inter = enc_dec.inter_dec(embeds1, embeds2, formula.rels[0][-1])
        query_intersection, embeds_inter = enc_dec.inter_dec(formula.rels[0][-1], [embeds1, embeds2])

        # this is where the attention take affect
        if enc_dec.inter_attn is not None:
            if enc_dec.use_inter_node and do_modelTraining:
                # for 3-chain_inter, the inter node is in the query_graph
                inter_nodes = [query.query_graph[1][2] for query in queries]
                inter_node_mode = formula.rels[0][2]
                query_embeds = enc_dec.graph.features(inter_nodes, inter_node_mode).t()
                query_intersection = enc_dec.inter_attn(query_embeds, embeds_inter, formula.target_mode)
            else:
                query_intersection = enc_dec.inter_attn(query_intersection, embeds_inter, formula.rels[0][-1])

        query_intersection = enc_dec.path_dec.project(query_intersection, enc_dec.graph._reverse_relation(formula.rels[0]))
        # scores = enc_dec.cos(target_embeds, query_intersection)
        # return scores
        return query_intersection
    elif pattern.match(formula.query_type) is not None:
        # x-inter
        # target_embeds = enc_dec.enc(source_nodes, formula.target_mode)
        num_edges = int(formula.query_type.replace("-inter", ""))
        embeds_list = []
        for i in range(0, num_edges):
            # project the ith anchor node to target node in rels: (t, pi, ai)
            embeds = enc_dec.enc([query.anchor_nodes[i] for query in queries], formula.anchor_modes[i])
            embeds = enc_dec.path_dec.project(embeds, enc_dec.graph._reverse_relation(formula.rels[i]))
            embeds_list.append(embeds)

        query_intersection, embeds_inter = enc_dec.inter_dec(formula.target_mode, embeds_list)

        # this is where the attention take affect
        if enc_dec.inter_attn is not None:
            if enc_dec.use_inter_node and do_modelTraining:
                # for x-inter, the inter node is target node
                # so we can use the real target node
                query_embeds = enc_dec.graph.features([query.target_node for query in queries], formula.target_mode).t()
                query_intersection = enc_dec.inter_attn(query_embeds, embeds_inter, formula.target_mode)
            else:
                query_intersection = enc_dec.inter_attn(query_intersection, embeds_inter, formula.target_mode)


        # scores = enc_dec.cos(target_embeds, query_intersection)
        # return scores
        return query_intersection