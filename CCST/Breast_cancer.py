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
import shutil


def eval_model(pred, labels=None):
    if labels is not None:
        label_df = pd.DataFrame({"True": labels, "Pred": pred}).dropna()
        # label_df = pd.DataFrame({"True": labels, "Pred": pred}).dropna()
        completeness = completeness_score(label_df["True"], label_df["Pred"])
        hm = homogeneity_score(label_df["True"], label_df["Pred"])
        vm = v_measure_score(label_df["True"], label_df["Pred"])
        ari = adjusted_rand_score(label_df["True"], label_df["Pred"])
        nmi = normalized_mutual_info_score(label_df["True"], label_df["Pred"])

    return completeness, vm, hm, ari, nmi


def run_CCST(data_name, n_clusters):
    import os
    import sys
    sys.path.append("/home/fangzy/project/st_cluster/code/methods/CCST-main")
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    from sklearn import metrics
    from scipy import sparse

    import numpy as np
    import pickle
    import pandas as pd
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import GCNConv, ChebConv, GATConv, DeepGraphInfomax, global_mean_pool, global_max_pool  # noqa
    from torch_geometric.data import Data, DataLoader
    from datetime import datetime

    rootPath = os.path.dirname(sys.path[0])
    os.chdir(rootPath+'/CCST')

    import argparse
    parser = argparse.ArgumentParser()
    # ================Specify data type firstly===============
    parser.add_argument('--data_type', default='nsc', help='"sc" or "nsc", \
        refers to single cell resolution datasets(e.g. MERFISH) and \
        non single cell resolution data(e.g. ST) respectively')
    # =========================== args ===============================
    parser.add_argument('--data_name', type=str, default='V1_Breast_Cancer_Block_A_Section_1',
                        help="'MERFISH' or 'V1_Breast_Cancer_Block_A_Section_1")
    # 0.8 on MERFISH, 0.3 on ST
    parser.add_argument('--lambda_I', type=float, default=0.3)
    parser.add_argument('--data_path', type=str,
                        default='generated_data/', help='data path')
    parser.add_argument('--model_path', type=str, default='model')
    parser.add_argument('--embedding_data_path', type=str,
                        default='Embedding_data')
    parser.add_argument('--result_path', type=str, default='results')
    parser.add_argument('--DGI', type=int, default=1,
                        help='run Deep Graph Infomax(DGI) model, otherwise direct load embeddings')
    parser.add_argument('--load', type=int, default=0,
                        help='Load pretrained DGI model')
    parser.add_argument('--num_epoch', type=int, default=5000,
                        help='numebr of epoch in training DGI')
    parser.add_argument('--hidden', type=int, default=256,
                        help='hidden channels in DGI')
    parser.add_argument('--PCA', type=int, default=1, help='run PCA or not')
    parser.add_argument('--cluster', type=int, default=1,
                        help='run cluster or not')
    parser.add_argument('--n_clusters', type=int, default=5,
                        help='number of clusters in Kmeans, when ground truth label is not avalible.')  # 5 on MERFISH, 20 on Breast
    parser.add_argument('--draw_map', type=int,
                        default=1, help='run drawing map')
    parser.add_argument('--diff_gene', type=int, default=0,
                        help='Run differential gene expression analysis')
    args = parser.parse_args(args=['--data_type', "nsc",
                                   '--data_name', data_name,
                                   '--data_path', '/home/sda1/fangzy/data/st_data/Benchmark/CCST/generate/',
                                   '--model_path', '/home/sda1/fangzy/data/st_data/Benchmark/CCST/model/',
                                   '--embedding_data_path', '/home/sda1/fangzy/data/st_data/Benchmark/CCST/embedding',
                                   '--result_path', '/home/sda1/fangzy/data/st_data/Benchmark/CCST/result',
                                   '--n_clusters', n_clusters])

    args.embedding_data_path = args.embedding_data_path + '/' + args.data_name + '/'
    args.model_path = args.model_path + '/' + args.data_name + '/'
    args.result_path = args.result_path + '/' + args.data_name + '/'
    if not os.path.exists(args.embedding_data_path):
        os.makedirs(args.embedding_data_path)
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    args.result_path = args.result_path+'lambdaI'+str(args.lambda_I) + '/'
    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)
    print('------------------------Model and Training Details--------------------------')
    print(args)

    if args.data_type == 'sc':  # should input a single cell resolution dataset, e.g. MERFISH
        from CCST_merfish_utils import CCST_on_MERFISH
        CCST_on_MERFISH(args)
    elif args.data_type == 'nsc':  # should input a non-single cell resolution dataset, e.g. V1_Breast_Cancer_Block_A_Section_1
        from CCST_ST_utils import CCST_on_ST
        CCST_on_ST(args)
    else:
        print('Data type not specified')
# dataset = "151669"
# run_CCST(dataset)
# pred = pd.read_csv("/home/sda1/fangzy/data/st_data/Benchmark/CCST/result/"+dataset+"/lambdaI0.3/types.txt",
#                         index_col=0, header=None)
# completeness, hm, nmi, ari = eval_model(pred.iloc[:,2],
#                                         pred.iloc[:,1])
# print(ari)

save_path = "/home/fangzy/project/st_cluster/code/compare/CCST/res/"
dataset="Breast_cancer"
results = pd.DataFrame()
best_ari = 0
for i in range(10):
    start = time.time()
    run_CCST(dataset, "20")
    end = time.time()
    pred = pd.read_csv("/home/sda1/fangzy/data/st_data/Benchmark/CCST/result/"+dataset+"/lambdaI0.3/types.txt",
                        index_col=0)
    completeness, vm, hm, ari, nmi = eval_model(pred.iloc[:,2],
                                            pred.iloc[:,1])
    res = {}
    res["nmi"] = nmi
    res["ari"] = ari
    res["completeness"] = completeness
    res["hm"] = hm
    res["vm"] = vm
    res["dataset"] = dataset
    res["time"] = end-start
    res["nmi_1"] = nmi
    res["ari_1"] = ari
    res["completeness_1"] = completeness
    res["hm_1"] = hm
    res["vm_1"] = vm
    res["round"] = i
    if res["ari"] > best_ari:
        if os.path.exists("/home/sda1/fangzy/data/st_data/Benchmark/CCST/best/result/"+dataset):
            shutil.rmtree("/home/sda1/fangzy/data/st_data/Benchmark/CCST/best/result/"+dataset)
        if os.path.exists("/home/sda1/fangzy/data/st_data/Benchmark/CCST/best/embedding/"+dataset):
            shutil.rmtree("/home/sda1/fangzy/data/st_data/Benchmark/CCST/best/embedding/"+dataset)
        shutil.move("/home/sda1/fangzy/data/st_data/Benchmark/CCST/result/"+dataset, "/home/sda1/fangzy/data/st_data/Benchmark/CCST/best/result")
        shutil.move("/home/sda1/fangzy/data/st_data/Benchmark/CCST/embedding/"+dataset, "/home/sda1/fangzy/data/st_data/Benchmark/CCST/best/embedding")

    results = results.append(res, ignore_index=True)
    results.to_csv(save_path+dataset+".csv", header=True)

res_mean = res_dataset_mean.mean()
res_mean.to_csv(save_path+datast+"_mean.csv", header=True)
res_mean = res_dataset_mean.median()
res_mean.to_csv(save_path+dataset+"_median.csv", header=True)





# save_path = "/home/fangzy/project/st_cluster/code/compare/CCST/res/"
# datasets = os.listdir(data_path)
# results = pd.DataFrame()
# for dataset in datasets:
#     start = time.time()
#     torch.cuda.empty_cache()
#     deepen = run(save_path = save_data_path+str(dataset),
#         platform="Visium",
#         pca_n_comps = 100,
#         pre_epochs = 1000, #### According to your own hardware, choose the number of training
#         linear_encoder_hidden=[64,16],
#         conv_hidden=[64,16],
#         epochs = 1000, #### According to your own hardware, choose the number of training
#         Conv_type="GCNConv", #### you can choose GNN types.
#         )
#     adata = deepen._get_adata(data_path, dataset)
#     adata = deepen._get_augment(adata, adjacent_weight = 0.3,
#                                 neighbour_k=4, weights="weights_matrix_all")
#     graph_dict = deepen._get_graph(adata.obsm["spatial"], distType="KDTree",
#                                    k=12)
#     adata = deepen._fit(adata, graph_dict, pretrain = False, dim_reduction=True)

#     # add ground_truth
#     df_meta = pd.read_table(data_path+str(dataset)+'/'+str(dataset)+'_truth.txt',
#                             sep='\t', header=None, index_col=0)
#     df_meta.columns = ['groud_truth']
#     adata.obs['ground_truth'] = df_meta.loc[adata.obs_names, 'groud_truth'].values
#     n_clusters = len(set(adata.obs["ground_truth"].dropna()))
#     adata = deepen._get_cluster_data(adata, n_domains = n_clusters, priori=True)
#     end = time.time()

#     completeness, hm, nmi, ari = eval_model(adata.obs['DeepST_refine_domain'],
#                                             adata.obs['ground_truth'])
#     res = {}
#     res["nmi"] = nmi
#     res["ari"] = ari
#     res["completeness"] = completeness
#     res["hm"] = hm
#     res["dataset"] = dataset
#     res["time"] = end-start

#     for metric in ["completeness", "hm", "nmi", "ari", "time"]:
#         adata.uns[metric] = res[metric]
#     adata.write_h5ad(save_data_path+str(dataset)+".h5ad")
#     results = results.append(res, ignore_index=True)
#     results.to_csv(save_path +
#                 "/res_new_4.csv", header=True)
# results = results.drop(["dataset"], axis=1)
# res_mean = results.mean()
# res_mean.to_csv(save_path+"/mean_res_new.csv", header=True)
