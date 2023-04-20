# Benchmarking SOTA spatial clustering methods
A comprehensive survey of state-of-the-art spatial clustering algorithms in spatially resolved transcriptomics
------
Spatial transcriptomics techniques enable researchers to quantify and localize mRNA transcripts at high resolution while preserving their spatial context. Spatial domain identification, or spatial clustering, is crucial in investigating spatial transcriptome technologies. Graph neural networks (GNNs) have emerged as a promising approach to performing cell type classification by leveraging gene expression, spatial location, and histology images. This manuscript comprehensively overviews state-of-the-art GNN-based spatial clustering algorithms for spatially resolved transcriptomics. We evaluate their performance on prevalent spatial transcriptome data, including clustering accuracy, robustness, achievement, requirement, computational efficiency, memory usage, and other relevant metrics. In each method, we estimate a total of 60 clustering situations by extending the essential spatial clustering frameworks in GNN selections, downstream clustering algorithms, principal component analysis (PCA) reduction, and refined correction. By comparing and analyzing the spatial clustering performance, we identify the relevant limitations of current applications and specify future research directions in this field. This survey provides novel insights and additional motivations for investigating spatial transcriptomics. 

## Workflow of spatial clustering task
![](https://github.com/narutoten520/Benchmark_SRT/blob/9e0608f6df2c785b0a845ccf0c9438e5e7610294/Figure3-github.png)

## Contents
* [Prerequisites](https://github.com/narutoten520/Benchmark_SRT#prerequisites)
* [Example usage](https://github.com/narutoten520/Benchmark_SRT#example-usage)
* [Benchmarking methods](https://github.com/narutoten520/Benchmark_SRT#benchmarking-methods)
* [Datasets Availability](https://github.com/narutoten520/Benchmark_SRT#datasets-availability)
* [License](https://github.com/narutoten520/Benchmark_SRT#license)
* [Trouble shooting](https://github.com/narutoten520/Benchmark_SRT#trouble-shooting)

### Prerequisites

1. Python (>=3.8)
2. Scanpy
3. Squidpy
4. Pytorch_pyG
5. Pandas
6. Numpy
7. Sklearn
8. Seaborn
9. Matplotlib
10. Torch_geometric

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Example usage
* Selecting the optimal GNN for spatial clustering task in GraphST
  ```sh
    running GraphST.py to choose the suitable GNN for GraphST on human breast cancner data
  ```
* Selecting the optimal GNN for spatial clustering task in STAGATE
  ```sh
    running STAGATE.py to choose the suitable GNN for GraphST on human breast cancner data
  ```
<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Benchmarking methods
Benchmarking methods used in this paper include: 
* [CCST](https://github.com/xiaoyeye/CCST)
* [DeepST](https://github.com/JiangBioLab/DeepST)
* [GraphST](https://github.com/JinmiaoChenLab/GraphST)
* [STAGATE](https://github.com/zhanglabtools/STAGATE)
* [SpaGCN](https://github.com/jianhuupenn/SpaGCN)
* [conST](https://github.com/ys-zong/conST)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Datasets Availability

* [Human DLPFC](https://github.com/LieberInstitute/spatialLIBD)
* [Mouse brain](https://squidpy.readthedocs.io/en/stable/auto_tutorials/tutorial_visium_hne.html)
* [Slide-seqV2](https://squidpy.readthedocs.io/en/stable/auto_tutorials/tutorial_slideseqv2.html)
* [Stereo-seq](https://stagate.readthedocs.io/en/latest/T4_Stereo.html)

<p align="right">(<a href="#readme-top">back to top</a>)</p>


### License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Trouble shooting

* data files<br>
Please down load the spatial transcriptomics data from the provided links.

* Porch_pyg<br>
Please follow the instruction to install pyG and geometric packages.
