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
        adata.obs['ground_truth'] = adata.obs["label"]
        df_meta = pd.read_table(file_fold + '/Annotation_STARmap_20180505_BY3_1k.txt',
                                sep='\t', index_col=0)
        adata.obs['Annotation'] = df_meta.loc[adata.obs_names,
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


def run_STAGATE(adata, dataset, random_seed=np.random.randint(100),
                device=torch.device(
                    'cuda:1' if torch.cuda.is_available() else 'cpu'),
                save_data_path="/home/sda1/fangzy/data/st_data/Benchmark/STAGATE/",
                n_clusters=None, rad_cutoff=150):
    import STAGATE_pyG as STAGATE
    start = time.time()
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    STAGATE.Cal_Spatial_Net(adata, rad_cutoff=rad_cutoff)
    STAGATE.Stats_Spatial_Net(adata)
    adata = STAGATE.train_STAGATE(
        adata, device=device, random_seed=random_seed)
    sc.pp.neighbors(adata, use_rep='STAGATE')
    sc.tl.umap(adata)

    if ("ground_truth" in adata.obs.keys()):
        n_clusters = len(set(adata.obs["ground_truth"].dropna()))
    else:
        n_clusters = n_clusters
    adata = STAGATE.mclust_R(
        adata, used_obsm='STAGATE', num_cluster=n_clusters)
    end = time.time()
    obs_df = adata.obs.dropna()
    adata.obs["pred_label"] = adata.obs["mclust"]
    adata.obsm["embedding"] = adata.obsm["STAGATE"]

    res = {}
    if ("ground_truth" in adata.obs.keys()):
        completeness, hm, nmi, ari = eval_model(adata.obs['mclust'],
                                                adata.obs['ground_truth'])
        res["nmi"] = nmi
        res["ari"] = ari
        res["completeness"] = completeness
        res["hm"] = hm
        res["dataset"] = dataset
        res["time"] = end-start
        res["nmi_1"] = nmi
        res["ari_1"] = ari
        res["completeness_1"] = completeness
        res["hm_1"] = hm

        for metric in ["completeness", "hm", "nmi", "ari", "time",
                       "completeness_1", "hm_1", "nmi_1", "ari_1"]:
            adata.uns[metric] = res[metric]
    # adata.write_h5ad(save_data_path+str(dataset)+".h5ad")
    return res, adata

# save_path = "/home/fangzy/project/st_cluster/code/compare/STAGATE/res"
# save_data_path="/home/sda1/fangzy/data/st_data/Benchmark/STAGATE/"
# results = pd.DataFrame()
# for dataset in ["Breast_cancer", "Mouse_brain", "STARmap"]:
#     best_ari = 0
#     adata = read_data(dataset)
#     for i in range(10):
#         random_seed = np.random.randint(100)
#         res, adata_h5 = run_STAGATE(adata.copy(), dataset, random_seed=random_seed)
#         res["round"] = i
#         results = results.append(res, ignore_index=True)
#         results.to_csv(save_path +
#                     "/res_other.csv", header=True)
#         if res["ari_1"] > best_ari:
#             adata_h5.write_h5ad(save_data_path+str(dataset)+".h5ad")
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
#     res, adata_h5 = run_STAGATE(adata.copy(), dataset,
#                               n_clusters=n_clusters_map[dataset],
#                               device=torch.device('cuda:1'))
#     adata_h5.write_h5ad(save_data_path+str(dataset)+"_" +
#                         str(n_clusters_map[dataset])+".h5ad")


save_path = "/home/fangzy/project/st_cluster/code/compare/STAGATE/res"
save_data_path = "/home/sda1/fangzy/data/st_data/Benchmark/STAGATE/"
results = pd.DataFrame()
for dataset in ["Breast_cancer"]:
    adata = read_data(dataset)
    for i in range(1):
        random_seed = 0
        res, adata_h5 = run_STAGATE(adata.copy(), dataset,
                                    random_seed=random_seed, rad_cutoff=400)
        res["round"] = i
        results = results.append(res, ignore_index=True)
        results.to_csv(save_path +
                       "/res_"+dataset+".csv", header=True)
        adata_h5.write_h5ad(save_data_path+str(dataset)+".h5ad")
res_dataset_mean = results.groupby(["dataset"]).mean()
res_dataset_mean.to_csv(save_path+"/other_data_mean.csv", header=True)

res_dataset_std = results.groupby(["dataset"]).std()
res_dataset_mean.to_csv(save_path+"/other_data_std.csv", header=True)

res_mean = res_dataset_mean.mean()
res_mean.to_csv(save_path+"/other_mean.csv", header=True)
res_mean = res_dataset_mean.median()
res_mean.to_csv(save_path+"/other_median.csv", header=True)
