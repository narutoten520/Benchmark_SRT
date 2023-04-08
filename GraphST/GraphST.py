import os
import torch
import pandas as pd
import scanpy as sc
from sklearn import metrics
import multiprocessing as mp
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import (
    v_measure_score, homogeneity_score, completeness_score)
import glob2
import matplotlib.pyplot as plt
import squidpy as sq
import time
import numpy as np
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
        adata.obs['ground_truth'] = adata.obs["label"]
        df_meta = pd.read_table(file_fold + '/Annotation_STARmap_20180505_BY3_1k.txt',
                                sep='\t', index_col=0)
        adata.obs['Annotation'] = df_meta.loc[adata.obs_names, 'Annotation'].values


    if dataset == "Mouse_brain":
        adata = sq.datasets.visium_hne_adata()
        adata.var_names_make_unique()
        adata.obs['ground_truth'] = adata.obs["cluster"]

    if dataset == "Breast_cancer":
        file_fold = data_path + str(dataset) #please replace 'file_fold' with the download path
        adata = sc.read_visium(file_fold, count_file='filtered_feature_bc_matrix.h5',
                               load_images=True)
        adata.var_names_make_unique()
        df_meta = pd.read_table(file_fold + '/metadata.tsv',sep='\t',
                                index_col=0)
        adata.obs['ground_truth'] = df_meta.loc[adata.obs_names, 'ground_truth'].values
    return adata

def run_GraphST(adata, dataset, random_seed = np.random.randint(100),
                device=torch.device('cuda:1' if torch.cuda.is_available() else 'cpu'),
                save_data_path="/home/sda1/fangzy/data/st_data/Benchmark/GraphST/"):
    from GraphST import GraphST
    from GraphST.utils import clustering

    start = time.time()
    # define model
    model = GraphST.GraphST(adata, device=device, random_seed=random_seed)
    # train model
    adata = model.train()
    n_clusters = len(set(adata.obs["ground_truth"].dropna()))
    # set radius to specify the number of neighbors considered during refinement
    radius = 50
    adata.obsm["embedding"] = adata.obsm["emb_pca"]
    clustering(adata, n_clusters, radius=radius, method="mclust",
               refinement=False)
    completeness, hm, nmi, ari = eval_model(adata.obs['domain'], 
                                            adata.obs['ground_truth'])
    if dataset=="Breast_cancer":
        clustering(adata, n_clusters, radius=radius, method="mclust", 
                   refinement=True)
        completeness_r, hm_r, nmi_r, ari_r = eval_model(adata.obs['domain'], 
                                                        adata.obs['ground_truth'])
    else:
        completeness_r, hm_r, nmi_r, ari_r = completeness, hm, nmi, ari
    end = time.time()
    res = {}
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
    adata.obs["pred_label"] = adata.obs['domain']

    for metric in ["completeness", "hm", "nmi", "ari", "time",
                "completeness_1", "hm_1", "nmi_1", "ari_1"]:
        adata.uns[metric] = res[metric]
    # adata.write_h5ad(save_data_path+str(dataset)+".h5ad")
    return res, adata



save_data_path="/home/sda1/fangzy/data/st_data/Benchmark/GraphST/"
save_path = "/home/fangzy/project/st_cluster/code/compare/GraphST/res"
results = pd.DataFrame()
# "Breast_cancer", "Mouse_brain", 
for dataset in ["Breast_cancer"]:
    adata = read_data(dataset)
    best_ari = 0
    for i in range(1):
        random_seed = 0
        res, adata_h5= run_GraphST(adata.copy(), dataset, random_seed=random_seed)
        res["round"] = i
        results = results.append(res, ignore_index=True)
        results.to_csv(save_path +
                    "/res_STARmap.csv", header=True)
        if res["ari_1"] > best_ari:
            adata_h5.write_h5ad(save_data_path+str(dataset)+".h5ad")
# res_dataset_mean = results.groupby(["dataset"]).mean()
# res_dataset_mean.to_csv(save_path+"/other_data_mean.csv", header=True)

# res_dataset_std = results.groupby(["dataset"]).std()
# res_dataset_mean.to_csv(save_path+"/other_data_std.csv", header=True)

# res_mean = res_dataset_mean.mean()
# res_mean.to_csv(save_path+"/other_mean.csv", header=True)
# res_mean = res_dataset_mean.median()
# res_mean.to_csv(save_path+"/other_median.csv", header=True)
