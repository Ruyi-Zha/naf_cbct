# NAF: Neural Attenuation Fields for Sparse-View CBCT Reconstruction

Code for [NAF: Neural Attenuation Fields for Sparse-View CBCT Reconstruction](https://arxiv.org/abs/2209.14540).

![NAF framework](framework.png)

```sh
@inproceedings{zha2022naf,
  title={NAF: Neural Attenuation Fields for Sparse-View CBCT Reconstruction},
  author={Zha, Ruyi and Zhang, Yanhao and Li, Hongdong},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={442--452},
  year={2022},
  organization={Springer}
}
```

### Installation

``` sh
# Create envorinment
conda create -n naf python=3.9
conda activate naf

# Install pytorch (hash encoder requires CUDA v11.3)
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113

# Install other packages
pip install -r requirements.txt
```

### Training

Download four CT datasets from [here](https://drive.google.com/drive/folders/1BJYR4a4iHpfFFOAdbEe5O_7Itt1nukJd?usp=sharing). Put them into the `./data` folder.

Experiments settings are stored in `./config` folder.

For example, train NAF with `chest_50` dataset:

``` sh
python train.py --config ./config/chest_50.yaml
```

### Make your own simulation dataset

You can make your own simulation dataset with TIGRE toolbox. Please first install [TIGRE](https://github.com/CERN/TIGRE).

Put your CT data in the format as follows. Examples can be seen in [here](https://drive.google.com/drive/folders/1BJYR4a4iHpfFFOAdbEe5O_7Itt1nukJd?usp=sharing).

```sh
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

Hash encoder and code structure are adapted from [torch-ngp](https://github.com/ashawkey/torch-ngp.git).

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

Many thanks to the amazing [TIGRE toolbox](https://github.com/CERN/TIGRE.git).
