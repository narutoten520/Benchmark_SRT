# https://github.com/jianhuupenn/SpaGCN

import os
import torch
import pandas as pd
import scanpy as sc
import numpy as np
from sklearn import metrics
import multiprocessing as mp
from sklearn.metrics.cluster import adjusted_rand_score
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
        adata.obs['ground_truth'] = df_meta.loc[adata.obs_names, 'Annotation'].values


    if dataset == "Mouse_brain":
        adata = sq.datasets.visium_hne_adata()
        adata.var_names_make_unique()
        adata.obs['ground_truth'] = adata.obs["cluster"]

    if dataset == "Breast_cancer":
        file_fold = data_path + str(dataset)
        adata = sc.read_visium(file_fold, count_file='filtered_feature_bc_matrix.h5',
                               load_images=True)
        adata.var_names_make_unique()
        df_meta = pd.read_table(file_fold + '/metadata.tsv',sep='\t',
                                index_col=0)
        adata.obs['ground_truth'] = df_meta.loc[adata.obs_names, 'ground_truth'].values
    return adata


def run_SpaGCN(adata, dataset, random_seed = np.random.randint(100),
                device=torch.device('cuda:1' if torch.cuda.is_available() else 'cpu'),
                save_data_path="/home/sda1/fangzy/data/st_data/Benchmark/SpaGCN/"):
    # refine_map
    import sys
    import warnings
    import numpy as np
    import argparse
    import SpaGCN as spg
    import random, torch
    from sklearn import metrics
    import cv2
    import matplotlib.pyplot as plt
    from pathlib import Path

    start = time.time()

    ##### Spatial domain detection using SpaGCN
    spg.prefilter_genes(adata, min_cells=3) # avoiding all genes are zeros
    spg.prefilter_specialgenes(adata)
    #Normalize and take log for UMI
    sc.pp.normalize_per_cell(adata)
    sc.pp.log1p(adata)

    #Set coordinates
    adata.obs["x_array"]=adata.obs["array_row"]
    adata.obs["y_array"]=adata.obs["array_col"]
    x_array=adata.obs["x_array"].tolist()
    y_array=adata.obs["y_array"].tolist()

    adj=spg.calculate_adj_matrix(x=x_array,y=y_array, histology=False)
    p=0.5 
    l=spg.search_l(p, adj, start=0.01, end=1000, tol=0.01, max_run=100)
    
    n_clusters = len(set(adata.obs["ground_truth"].dropna()))
    r_seed=t_seed=n_seed=random_seed
    res=spg.search_res(adata, adj, l, n_clusters, start=0.7, step=0.1, 
                       tol=5e-3, lr=0.05, max_epochs=20, r_seed=r_seed, 
                       t_seed=t_seed, n_seed=n_seed)
    
    ### 4.3 Run SpaGCN
    clf=spg.SpaGCN()
    clf.set_l(l)
    # Set seed
    random.seed(r_seed)
    torch.manual_seed(t_seed)
    np.random.seed(n_seed)
    #Run
    clf.train(adata, adj, init_spa=True, init="louvain",
              res=res, tol=5e-3, lr=0.05, max_epochs=200)
    y_pred, prob=clf.predict()
    adata.obs["pred"]= y_pred
    adata.obs["pred"]=adata.obs["pred"].astype('category')

    refine_map = {"Breast_cancer": "hexagon", "Mouse_brain": "hexagon", 
                  "STARmap": "square"}
    #Do cluster refinement(optional) 
    # shape="hexagon" for Visium data, "square" for ST data.
    adj_2d=spg.calculate_adj_matrix(x=x_array, y=y_array, histology=False)
    refined_pred=spg.refine(sample_id=adata.obs.index.tolist(), 
                            pred=adata.obs["pred"].tolist(), dis=adj_2d,
                            shape=refine_map[dataset])
    adata.obs["refined_pred"]=refined_pred
    adata.obs["refined_pred"]=adata.obs["refined_pred"].astype('category')
    #Save results
    # adata.write_h5ad(f"{dir_output}/results.h5ad")
    # adata.obs.to_csv(f'{dir_output}/metadata.tsv', sep='\t')

    completeness, hm, nmi, ari = eval_model(adata.obs['pred'],
                                            adata.obs['ground_truth'])
    completeness_r, hm_r, nmi_r, ari_r = eval_model(adata.obs['refined_pred'],
                                                    adata.obs['ground_truth'])

    end = time.time()
    res = {}
    res["nmi"] = nmi
    res["ari"] = ari
    res["completeness"] = completeness
    res["hm"] = hm
    res["nmi_1"] = nmi_r
    res["ari_1"] = ari_r
    res["completeness_1"] = completeness_r
    res["hm_1"] = hm_r
    res["dataset"] = dataset
    res["time"] = end-start

    for metric in ["completeness", "hm", "nmi", "ari", "time",
                   "completeness_1", "hm_1", "nmi_1", "ari_1"]:
        adata.uns[metric] = res[metric]
    # adata.write_h5ad(save_data_path+str(dataset)+".h5ad")
    return res, adata


save_data_path="/home/sda1/fangzy/data/st_data/Benchmark/SpaGCN/"
save_path = "/home/fangzy/project/st_cluster/code/compare/SpaGCN/res"
results = pd.DataFrame()
for dataset in ["Breast_cancer", "Mouse_brain", "STARmap"]:
    print("---------start-----------"+dataset)
    adata = read_data(dataset)
    random_seed = np.random.randint(100)
    res, adata_h5 = run_SpaGCN(adata.copy(), dataset, random_seed=random_seed)
    results = results.append(res, ignore_index=True)
    results.to_csv(save_path +
                "/res_other.csv", header=True)
    adata_h5.write_h5ad(save_data_path+str(dataset)+".h5ad")
# res_dataset_mean = results.groupby(["dataset"]).mean()
# res_dataset_mean.to_csv(save_path+"/other_data_mean.csv", header=True)

# res_dataset_std = results.groupby(["dataset"]).std()
# res_dataset_mean.to_csv(save_path+"/other_data_std.csv", header=True)

# res_mean = res_dataset_mean.mean()
# res_mean.to_csv(save_path+"/other_mean.csv", header=True)
# res_mean = res_dataset_mean.median()
# res_mean.to_csv(save_path+"/other_median.csv", header=True)
