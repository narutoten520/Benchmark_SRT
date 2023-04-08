import os
import torch
import pandas as pd
import numpy as np
import scanpy as sc
from sklearn import metrics
import multiprocessing as mp
from sklearn.metrics.cluster import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics.cluster import (
    v_measure_score, homogeneity_score, completeness_score)
import glob2
import matplotlib.pyplot as plt
import squidpy as sq
import time
from matplotlib.backends.backend_pdf import PdfPages


def eval_model(pred, labels=None):
    if labels is not None:
        label_df = pd.DataFrame({"True": labels, "Pred": pred}).dropna()
        # label_df = pd.DataFrame({"True": labels, "Pred": pred}).dropna()
        completeness = completeness_score(label_df["True"], label_df["Pred"])
        hm = homogeneity_score(label_df["True"], label_df["Pred"])
        nmi = v_measure_score(label_df["True"], label_df["Pred"])
        ari = adjusted_rand_score(label_df["True"], label_df["Pred"])

    return completeness, hm, nmi, ari


def read_data(dataset, data_path='/home/sda1/fangzy/data/st_data/raw_data/'):
    if dataset == "STARmap":
        file_fold = data_path + str(dataset)
        adata = sc.read(file_fold+"/STARmap_20180505_BY3_1k.h5ad")
        adata.var_names_make_unique()
        df_meta = pd.read_table(file_fold + '/Annotation_STARmap_20180505_BY3_1k.txt',
                                sep='\t', index_col=0)
        adata.obs['ground_truth'] = df_meta.loc[adata.obs_names,
                                                'Annotation'].values

    if dataset == "Mouse_brain":
        adata = sq.datasets.visium_hne_adata()
        adata.var_names_make_unique()
        adata.obs['ground_truth'] = adata.obs["cluster"]

    if dataset == "Breast_cancer":
        # please replace 'file_fold' with the download path
        file_fold = data_path + str(dataset)
        adata = sc.read_visium(file_fold, count_file='filtered_feature_bc_matrix.h5',
                               load_images=True)
        adata.var_names_make_unique()
        df_meta = pd.read_table(file_fold + '/metadata.tsv', sep='\t',
                                index_col=0)
        adata.obs['ground_truth'] = df_meta.loc[adata.obs_names,
                                                'ground_truth'].values

    if dataset == "Mouse_hippocampus":
        adata = sq.datasets.slideseqv2()
        adata.var_names_make_unique()

    if dataset in ["Mouse_olfactory", "Mouse_brain_section_2", "Mouse_brain_section_1"]:
        # please replace 'file_fold' with the download path
        file_fold = data_path + str(dataset)
        adata = sc.read(file_fold+'/filtered_feature_bc_matrix.h5ad')
        adata.var_names_make_unique()
    return adata


# set seed before every run
def seed_torch(seed):
    import random
    import numpy as np
    import os
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def refine_function(sample_id, pred, dis, shape="hexagon"):
    refined_pred = []
    pred = pd.DataFrame({"pred": pred}, index=sample_id)
    dis_df = pd.DataFrame(dis, index=sample_id, columns=sample_id)
    if shape == "hexagon":
        num_nbs = 6
    elif shape == "square":
        num_nbs = 4
    else:
        print(
            "Shape not recongized, shape='hexagon' for Visium data, 'square' for ST data.")
    for i in range(len(sample_id)):
        index = sample_id[i]
        dis_tmp = dis_df.loc[index, :].sort_values(ascending=False)
        nbs = dis_tmp[0:num_nbs+1]
        nbs_pred = pred.loc[nbs.index, "pred"]
        self_pred = pred.loc[index, "pred"]
        v_c = nbs_pred.value_counts()
        if (v_c.loc[self_pred] < num_nbs/2) and (np.max(v_c) > num_nbs/2):
            refined_pred.append(v_c.idxmax())
        else:
            refined_pred.append(self_pred)
    return refined_pred


def run_conST(adata_h5, dataset,
              device=torch.device(
                  'cuda:0' if torch.cuda.is_available() else 'cpu'),
              save_data_path="/home/sda1/fangzy/data/st_data/Benchmark/conST/",
              n_clusters=6):
    import sys
    sys.path.append("/home/fangzy/project/st_cluster/code/methods/conST-main")
    from src.graph_func import graph_construction
    from src.utils_func import mk_dir, adata_preprocess, load_ST_file, res_search_fixed_clus, plot_clustering
    from src.training import conST_training
    import argparse
    import numpy as np

    parser = argparse.ArgumentParser()
    parser.add_argument('--k', type=int, default=10,
                        help='parameter k in spatial graph')
    parser.add_argument('--knn_distanceType', type=str, default='euclidean',
                        help='graph distance type: euclidean/cosine/correlation')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of epochs to train.')
    parser.add_argument('--cell_feat_dim', type=int,
                        default=300, help='Dim of PCA')
    parser.add_argument('--feat_hidden1', type=int,
                        default=100, help='Dim of DNN hidden 1-layer.')
    parser.add_argument('--feat_hidden2', type=int,
                        default=20, help='Dim of DNN hidden 2-layer.')
    parser.add_argument('--gcn_hidden1', type=int, default=32,
                        help='Dim of GCN hidden 1-layer.')
    parser.add_argument('--gcn_hidden2', type=int, default=8,
                        help='Dim of GCN hidden 2-layer.')
    parser.add_argument('--p_drop', type=float,
                        default=0.2, help='Dropout rate.')
    parser.add_argument('--use_img', type=bool,
                        default=False, help='Use histology images.')
    parser.add_argument('--img_w', type=float, default=0.1,
                        help='Weight of image features.')
    parser.add_argument('--use_pretrained', type=bool,
                        default=True, help='Use pretrained weights.')
    parser.add_argument('--using_mask', type=bool,
                        default=False, help='Using mask for multi-dataset.')
    parser.add_argument('--feat_w', type=float, default=10,
                        help='Weight of DNN loss.')
    parser.add_argument('--gcn_w', type=float, default=0.1,
                        help='Weight of GCN loss.')
    parser.add_argument('--dec_kl_w', type=float,
                        default=10, help='Weight of DEC loss.')
    parser.add_argument('--gcn_lr', type=float, default=0.01,
                        help='Initial GNN learning rate.')
    parser.add_argument('--gcn_decay', type=float,
                        default=0.01, help='Initial decay rate.')
    parser.add_argument('--dec_cluster_n', type=int,
                        default=10, help='DEC cluster number.')
    parser.add_argument('--dec_interval', type=int,
                        default=20, help='DEC interval nnumber.')
    parser.add_argument('--dec_tol', type=float, default=0.00, help='DEC tol.')

    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--beta', type=float, default=100,
                        help='beta value for l2c')
    parser.add_argument('--cont_l2l', type=float, default=0.3,
                        help='Weight of local contrastive learning loss.')
    parser.add_argument('--cont_l2c', type=float, default=0.1,
                        help='Weight of context contrastive learning loss.')
    parser.add_argument('--cont_l2g', type=float, default=0.1,
                        help='Weight of global contrastive learning loss.')

    parser.add_argument('--edge_drop_p1', type=float, default=0.1,
                        help='drop rate of adjacent matrix of the first view')
    parser.add_argument('--edge_drop_p2', type=float, default=0.1,
                        help='drop rate of adjacent matrix of the second view')
    parser.add_argument('--node_drop_p1', type=float, default=0.2,
                        help='drop rate of node features of the first view')
    parser.add_argument('--node_drop_p2', type=float, default=0.3,
                        help='drop rate of node features of the second view')

    # ______________ Eval clustering Setting ______________
    parser.add_argument('--eval_resolution', type=int,
                        default=1, help='Eval cluster number.')
    parser.add_argument('--eval_graph_n', type=int,
                        default=20, help='Eval graph kN tol.')

    params = parser.parse_args(args=['--k', '10', '--knn_distanceType',
                               'euclidean', '--epochs', '200', '--use_pretrained', 'False'])

    # np.random.seed(params.seed)
    # torch.manual_seed(params.seed)
    # torch.cuda.manual_seed(params.seed)
    params.device = device

    # seed_torch(params.seed)

    params.save_path = mk_dir(f'{save_data_path}/{dataset}')
    start = time.time()
    adata_X = adata_preprocess(
        adata_h5, min_cells=5, pca_n_comps=params.cell_feat_dim)
    graph_dict = graph_construction(
        adata_h5.obsm['spatial'], adata_h5.shape[0], params)
    # np.save('./input/adatax.npy', adata_X)
    # np.save('./input/graphdict.npy', graph_dict, allow_pickle = True)
    # adata_X = np.load('./input/adatax.npy')
    # graph_dict = np.load('./input/graphdict.npy',  allow_pickle = True).item()
    params.cell_num = adata_h5.shape[0]
    if ("ground_truth" in adata_h5.obs.keys()):
        n_clusters = len(set(adata_h5.obs["ground_truth"].dropna()))
    else:
        n_clusters = n_clusters

    if params.use_img:
        img_transformed = np.load('./MAE-pytorch/extracted_feature.npy')
        img_transformed = (img_transformed - img_transformed.mean()) / \
            img_transformed.std() * adata_X.std() + adata_X.mean()
        conST_net = conST_training(
            adata_X, graph_dict, params, n_clusters, img_transformed)
    else:
        conST_net = conST_training(adata_X, graph_dict, params, n_clusters)

    conST_net.pretraining()
    conST_net.major_training()

    conST_embedding = conST_net.get_embedding()

    np.save(f'{params.save_path}/conST_result.npy', conST_embedding)
    # clustering
    adata_h5.obsm["embedding"] = conST_embedding
    sc.pp.neighbors(adata_h5, n_neighbors=params.eval_graph_n,
                    use_rep='embedding')
    eval_resolution = res_search_fixed_clus(adata_h5, n_clusters)
    print(eval_resolution)
    cluster_key = "conST_leiden"
    sc.tl.leiden(adata_h5, key_added=cluster_key, resolution=eval_resolution)

    index = np.arange(start=0, stop=adata_X.shape[0]).tolist()
    index = [str(x) for x in index]

    dis = graph_dict['adj_norm'].to_dense().numpy(
    ) + np.eye(graph_dict['adj_norm'].shape[0])
    refine_map = {"Breast_cancer": "hexagon", "Mouse_brain": "hexagon",
                  "STARmap": "square", "Mouse_olfactory": "hexagon", }
    refine = refine_function(sample_id=index, shape=refine_map[dataset],
                             pred=adata_h5.obs['leiden'].tolist(), dis=dis)
    end = time.time()

    res = {}
    adata_h5.obs['refine'] = refine
    if ("ground_truth" in adata_h5.obs.keys()):
        completeness_r, hm_r, nmi_r, ari_r = eval_model(adata_h5.obs['refine'],
                                                        adata_h5.obs['ground_truth'])
        completeness, hm, nmi, ari = eval_model(adata_h5.obs['conST_leiden'],
                                                adata_h5.obs['ground_truth'])
        res["nmi"] = nmi
        res["ari"] = ari
        res["completeness"] = completeness
        res["hm"] = hm
        res["dataset"] = dataset
        res["time"] = end-start
        res["nmi_1"] = nmi_r
        res["ari_1"] = ari_r
        res["completeness_1"] = completeness_r
        res["hm_1"] = hm_r

        for metric in ["completeness", "hm", "nmi", "ari", "time",
                   "completeness_1", "hm_1", "nmi_1", "ari_1"]:
            adata_h5.uns[metric] = res[metric]

    cluster_key = 'refine'
    savepath = f'{params.save_path}/conST_leiden_plot_refined.jpg'
    plot_clustering(adata_h5, cluster_key, savepath=savepath)
    # plotting
    savepath = f'{params.save_path}/conST_leiden_plot.jpg'
    plot_clustering(adata_h5, cluster_key, savepath=savepath)
    adata_h5.obs["pred_label"] = refine

    return res, adata_h5


save_path = "/home/fangzy/project/st_cluster/code/compare/conST/res"
save_data_path = "/home/sda1/fangzy/data/st_data/Benchmark/conST/"
results = pd.DataFrame()
"Breast_cancer", "Mouse_brain", "STARmap", "Mouse_hippocampus", "Mouse_olfactory"
for dataset in ["Breast_cancer", "Mouse_brain", "STARmap"]:
    best_ari = 0
    adata = read_data(dataset)
    for i in range(1):
        res, adata_h5 = run_conST(adata.copy(), dataset)
        res["round"] = i
        results = results.append(res, ignore_index=True)
        results.to_csv(save_path +
                       "/other.csv", header=True)
        adata_h5.write_h5ad(save_data_path+str(dataset)+".h5ad")
# res_dataset_mean = results.groupby(["dataset"]).mean()
# res_dataset_mean.to_csv(save_path+"/other_data_mean.csv", header=True)

# res_dataset_std = results.groupby(["dataset"]).std()
# res_dataset_mean.to_csv(save_path+"/other_data_std.csv", header=True)

# res_mean = res_dataset_mean.mean()
# res_mean.to_csv(save_path+"/other_mean.csv", header=True)
# res_mean = res_dataset_mean.median()
# res_mean.to_csv(save_path+"/other_median.csv", header=True)


# n_clusters_map = {"Mouse_hippocampus": 10, "Mouse_olfactory": 7}
# for dataset in ["Mouse_hippocampus", "Mouse_olfactory"]:
#     print("------------------  "+dataset+" -------------------")
#     adata = read_data(dataset)
#     res, adata_h5 = run_conST(adata.copy(), dataset,
#                               n_clusters=n_clusters_map[dataset],
#                               device=torch.device('cuda:1'))
#     adata_h5.write_h5ad(save_data_path+str(dataset)+"_" +
#                         str(n_clusters_map[dataset])+".h5ad")
