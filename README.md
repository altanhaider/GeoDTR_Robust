# Overview
This is a re-implementation of the paper ["Cross-view Geo-localization via Learning Disentangled Geometric Layout Correspondence"](https://arxiv.org/abs/2212.04074). We extend this work by making it adaptable to limited FOV and non-north-aligned input ground images. This work was implemented as a class project for the course CS6540:Deep learning.

### Prerequisites
---
1. numpy
2. Pytorch >= 1.11
3. torchvision >= 0.12
4. tqdm
5. scipy
6. PIL


### Dataset
---
#### CVUSA

- We obtain the permission of CVUSA dataset from the owner by submit the [MVRL Dataset Request Form](https://mvrl.cse.wustl.edu/datasets/cvusa/).
- Please refer to the repo: [https://github.com/viibridges/crossnet](https://github.com/viibridges/crossnet)


### Training
---
```bash
python train.py \
--dataset CVUSA \
--data_dir path-to-your-data/ \
--n_des 8 \
--TR_heads 4 \
--TR_layers 2 \
--layout_sim strong \
--sem_aug strong \
--pt \
--cf \
--robust_aug strong \
--robust_loss_mse \
--robust_loss \
```
Toggling ```bash--cf``` for counterfactual learning schema, ```bash--pt``` for disable polar transformation, ```bash--verbose``` for progressive bar. ```bash--robust_aug``` applies random augmentation of chanign the FOV and alignment. ```bash--robust_loss``` and ```bash--robust_loss_mse``` are the triplet and MSE losses between the augmented ground image and original.
### Evaluation
---
```bash
python test.py \
--dataset CVUSA \
--data_dir path-to-your-data/ \
--model_path path-to-your-pretrained-weight
```

### Citation
---
```
@inproceedings{zhang2023cross,
  title={Cross-view geo-localization via learning disentangled geometric layout correspondence},
  author={Zhang, Xiaohan and Li, Xingyu and Sultani, Waqas and Zhou, Yi and Wshah, Safwan},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={37},
  number={3},
  pages={3480--3488},
  year={2023}
}
```
