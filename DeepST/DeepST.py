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
gpu_id = 1
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)


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
        adata.X = np.asarray(adata.X.todense())
        adata.var_names_make_unique()
        adata.obs['ground_truth'] = adata.obs["cluster"]

    if dataset == "Breast_cancer":
        import sys
        sys.path.append(
            "/home/fangzy/project/st_cluster/code/methods/DeepST-main/deepst")
        from DeepST import run
        from his_feat import image_feature, image_crop
        # please replace 'file_fold' with the download path
        file_fold = data_path + str(dataset)
        adata = sc.read_visium(file_fold, count_file='filtered_feature_bc_matrix.h5',
                               load_images=True)
        adata.var_names_make_unique()
        df_meta = pd.read_table(file_fold + '/metadata.tsv', sep='\t',
                                index_col=0)
        adata.obs['ground_truth'] = df_meta.loc[adata.obs_names,
                                                'ground_truth'].values

        library_id = list(adata.uns["spatial"].keys())[0]
        scale = adata.uns["spatial"][library_id]["scalefactors"][
            "tissue_hires_scalef"]
        image_coor = adata.obsm["spatial"] * scale
        adata.obs["imagecol"] = image_coor[:, 0]
        adata.obs["imagerow"] = image_coor[:, 1]
        adata.uns["spatial"][library_id]["use_quality"] = "hires"
        from pathlib import Path

        save_path_image_crop = Path(os.path.join(
            "/home/sda1/fangzy/data/st_data/Benchmark/DeepST/", 'Image_crop', f'{dataset}'))
        save_path_image_crop.mkdir(parents=True, exist_ok=True)
        adata = image_crop(adata, save_path=save_path_image_crop)
        adata = image_feature(adata, pca_components=50,
                              cnnType='ResNet50').extract_image_feat()
    if dataset == "Mouse_hippocampus":
        adata = sq.datasets.slideseqv2()
        adata.X = np.asarray(adata.X.todense())
        adata.var_names_make_unique()

    if dataset in ["Mouse_olfactory", "Mouse_olfactory_slide_seqv2"]:
        gene = pd.read_csv('/home/sda1/fangzy/data/st_data/raw_data/'+dataset+"/gene.csv", index_col=0)
        spatial = pd.read_csv('/home/sda1/fangzy/data/st_data/raw_data/'+dataset+"/spatial.csv", index_col=0)
        adata = sc.AnnData(np.array(gene))
        adata.obsm["spatial"] = np.array(spatial)


    if dataset in ["Mouse_brain_section_2", "Mouse_brain_section_1"]:
        # please replace 'file_fold' with the download path
        file_fold = data_path + str(dataset)
        adata = sc.read(file_fold+'/filtered_feature_bc_matrix.h5ad')
        adata.X = np.asarray(adata.X.todense())
        adata.var_names_make_unique()

    # if dataset == "Mouse_olfactory_slide_seqv2":
    #     adata = sc.read_h5ad(data_path+"/"+dataset+"/tutorial3.h5ad")
    return adata
                                                                

def run_DeepST(adata, dataset, platform="Visium", k=12,
               device=torch.device(
                   'cuda:1' if torch.cuda.is_available() else 'cpu'),
               save_data_path="/home/sda1/fangzy/data/st_data/Benchmark/DeepST/", n_clusters=None, priori=True,
               weights="weights_matrix_all"):
    # https://github.com/JiangBioLab/DeepST 一定要严格按照requirement安装对应版本的
    import os
    import sys
    sys.path.append(
        "/home/fangzy/project/st_cluster/code/methods/DeepST-main/deepst")
    from DeepST import run
    from his_feat import image_feature, image_crop
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

    start = time.time()
    deepen = run(save_path=save_path+str(dataset),
                 platform=platform,
                 pca_n_comps=200,
                 pre_epochs=800,  # According to your own hardware, choose the number of training
                 epochs=1000,  # According to your own hardware, choose the number of training
                 Conv_type="GCNConv",  # you can choose GNN types.
                 )
    # adata = deepen._get_adata(data_path, dataset)
    adata = deepen._get_augment(
        adata, adjacent_weight=0.3, neighbour_k=4, weights=weights,)
    graph_dict = deepen._get_graph(
        adata.obsm["spatial"], distType="BallTree", k=k)
    adata = deepen._fit(adata, graph_dict, pretrain=False)
    if priori:
        if ("ground_truth" in adata.obs.keys()):
            n_clusters = len(set(adata.obs["ground_truth"].dropna()))
        else:
            n_clusters = n_clusters
        adata = deepen._get_cluster_data(
            adata, n_domains=n_clusters, priori=True)
    else:
        adata = deepen._get_cluster_data(adata, priori=False)
    end = time.time()
    res = {}
    adata.obsm["embedding"] = adata.obsm["DeepST_embed"]
    adata.obs["pred_label"] = adata.obs['DeepST_refine_domain']
    if ("ground_truth" in adata.obs.keys()):
        completeness, hm, nmi, ari = eval_model(adata.obs['DeepST_refine_domain'],
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
    adata.obs["pred_label"] = adata.obs['DeepST_refine_domain']

    return res, adata


save_data_path = "/home/sda1/fangzy/data/st_data/Benchmark/DeepST/"
save_path = "/home/fangzy/project/st_cluster/code/compare/DeepST/res"
results = pd.DataFrame()
# "Breast_cancer",
for dataset in ["Breast_cancer"]:
    adata = read_data(dataset)
    best_ari = 0
    for i in range(1):
        res, adata_h5 = run_DeepST(adata.copy(), dataset)
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
# platform_map = {"Mouse_hippocampus": "slideseq",
#                 "Mouse_olfactory": "stereoseq",
#                 "Mouse_olfactory_slide_seqv2": "slideseq",}
# n_clusters_map = {"Mouse_hippocampus": 10, "Mouse_olfactory": 7,
#                   "Mouse_olfactory_slide_seqv2":9}
# for dataset in ["Mouse_olfactory_slide_seqv2"]:
#     print("------------------  "+dataset+" -------------------")
#     adata = read_data(dataset)
#     res, adata_h5 = run_DeepST(adata.copy(), dataset,
#                                platform=platform_map[dataset],
#                                n_clusters=n_clusters_map[dataset], k=6,
#                                device=torch.device('cuda:1'),
#                                weights="weights_matrix_nomd")
#     adata_h5.write_h5ad(save_data_path+str(dataset)+"_" +
#                         str(n_clusters_map[dataset])+".h5ad")

    # res, adata_h5 = run_DeepST(adata.copy(), dataset,
    #                            platform=platform_map[dataset], k=6,
    #                            device=torch.device('cuda:1'),
    #                            weights="weights_matrix_nomd", priori=False)
    # adata_h5.write_h5ad(save_data_path+str(dataset)+".h5ad")
