# naf_cbct

Code for NAF: Neural Attenuation Fields for Sparse-View CBCT Reconstruction.

![](framework.pdf)

### Installation with conda

```
git clone git@github.com:Ruyi-Zha/naf_cbct.git
cd naf_cbct
conda env create -f environment.yml
conda activate naf
```

### Training

Download four CT datasets from [here](https://drive.google.com/drive/folders/1BJYR4a4iHpfFFOAdbEe5O_7Itt1nukJd?usp=sharing). Put them into the `./data` folder.

Experiments settings are stored in `./config` folder.

For example, train NAF with `chest_50` dataset:

```
python train.py --config ./config/chest_50.yaml
```



### Make your own simulation dataset

You can make your own simulation dataset with TIGRE toolbox. Please first install [TIGRE](https://github.com/CERN/TIGRE). 

Put your CT data in the format as follows. Examples can be seen in [here](https://drive.google.com/drive/folders/1BJYR4a4iHpfFFOAdbEe5O_7Itt1nukJd?usp=sharing).

```
├── raw                                                                                                       
│   ├── XXX (your CT name)
│   │   └── img.mat (CT data)
│   │   └── config.yml (Information about CT data and the geometry setting of CT scanner)
```

Then use TIGRE to generate simulated X-ray projections.

```
python dataGenerator/generateData.py --ctName XXX --outputName XXX_50
```



# Acknowledgement

Hash encoder is adapted from [torch-ngp](https://github.com/ashawkey/torch-ngp.git).

```
@misc{torch-ngp,
    Author = {Jiaxiang Tang},
    Year = {2022},
    Note = {https://github.com/ashawkey/torch-ngp},
    Title = {Torch-ngp: a PyTorch implementation of instant-ngp}
}

@article{tang2022compressible,
    title = {Compressible-composable NeRF via Rank-residual Decomposition},
    author = {Tang, Jiaxiang and Chen, Xiaokang and Wang, Jingbo and Zeng, Gang},
    journal = {arXiv preprint arXiv:2205.14870},
    year = {2022}
}
```

The framework of NAF is adapted from [nerf-pytorch](https://github.com/yenchenlin/nerf-pytorch.git).

```
@misc{lin2020nerfpytorch,
  title={NeRF-pytorch},
  author={Yen-Chen, Lin},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished={\url{https://github.com/yenchenlin/nerf-pytorch/}},
  year={2020}
}
```

Many thanks to the amazing [TIGRE toolbox](https://github.com/CERN/TIGRE.git).

