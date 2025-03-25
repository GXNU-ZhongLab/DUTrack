# DUTrack
The official implementation for the **CVPR 2025** paper [_Dynamic Updates for Language Adaptation in Visual-Language Tracking_](https://arxiv.org/pdf/2503.06621).

[[Models](https://drive.google.com/drive/folders/1edieIddvSzy9219F4WShajvY4UmkKJV4?usp=sharing)][[Raw Results](https://drive.google.com/drive/folders/1tHpZx9WpgRzxQMbSKgkM48X5oUnM83ok?usp=drive_link)][[PreTrain](https://drive.google.com/drive/folders/15g87STgG4ZWr6ZTnKJdGL3Z-gwG0SVL8?usp=sharing)]

<!-- [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/joint-feature-learning-and-relation-modeling/visual-object-tracking-on-lasot)](https://paperswithcode.com/sota/visual-object-tracking-on-lasot?p=joint-feature-learning-and-relation-modeling)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/joint-feature-learning-and-relation-modeling/visual-object-tracking-on-got-10k)](https://paperswithcode.com/sota/visual-object-tracking-on-got-10k?p=joint-feature-learning-and-relation-modeling)

[//]: # ([![PWC]&#40;https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/joint-feature-learning-and-relation-modeling/visual-object-tracking-on-trackingnet&#41;]&#40;https://paperswithcode.com/sota/visual-object-tracking-on-trackingnet?p=joint-feature-learning-and-relation-modeling&#41;)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/joint-feature-learning-and-relation-modeling/visual-object-tracking-on-uav123)](https://paperswithcode.com/sota/visual-object-tracking-on-uav123?p=joint-feature-learning-and-relation-modeling)
 -->
<p align="center">
  <img width="85%" src="https://github.com/GXNU-ZhongLab/DUTrack/blob/main/aeest/framework.png" alt="Framework"/>
</p>



## Highlights

### :star2: Dynamic Updates Mutil-Modal Reference for Vision-Language Tracking Framework
DUTrack is a simple yet efficient vision-language tracker with the capability to **dynamically update** multi-modal references.It achieves state-of-the-art performance on five vision-language tracking benchmarks while maintaining competitive performance on two pure visual tracking benchmarks.

| Tracker     | LaSOT (AUC) | TNL2K (AUC)       | OTB-Lang(AUC) | LaSOT-ext(AUC) | MGIT(AUC) |
|:-----------:|:-----------:|:-----------------:|:-----------:|:-----------:|:-----------:|
| DUTrack-384 | 74.1        | 65.6              | 71.3        | 52.5       |71.0        |
| DUTrack-256 | 73.0        | 64.9              | 70.9        |50.5        |70.0        |



## Install the environment
**Conda**: Use the Anaconda (CUDA 11.2)
```
conda create -n DUTrack python=3.8
conda activate DUTrack
bash install.sh
```

## Set project paths
Run the following command to set paths for this project
```
python tracking/create_default_local_file.py --workspace_dir . --data_dir ./data --save_dir ./output
```
After running this command, you can also modify paths by editing these two files
```
lib/train/admin/local.py  # paths about training
lib/test/evaluation/local.py  # paths about testing
```

## Data Preparation
Put the tracking datasets in ./data. It should look like this:
   ```
   ${PROJECT_ROOT}
    -- data
        -- lasot
            |-- airplane
            |-- basketball
            |-- bear
            ...
        -- got10k
            |-- test
            |-- train
            |-- val
        -- coco
            |-- annotations
            |-- images
        -- trackingnet
            |-- TRAIN_0
            |-- TRAIN_1
            ...
            |-- TRAIN_11
            |-- TEST
   ```


## Training
Download pre-trained [MAE ViT-Base weights](https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth) and put it under `$PROJECT_ROOT$/pretrained_models` (different pretrained models can also be used, see [MAE](https://github.com/facebookresearch/mae) for more details).

```
python tracking/train.py --script ostrack --config vitb_256_mae_ce_32x4_ep300 --save_dir ./output --mode multiple --nproc_per_node 4 --use_wandb 1
```

Replace `--config` with the desired model config under `experiments/ostrack`. We use [wandb](https://github.com/wandb/client) to record detailed training logs, in case you don't want to use wandb, set `--use_wandb 0`.


## Evaluation
Download the model weights from [Google Drive](https://drive.google.com/drive/folders/1PS4inLS8bWNCecpYZ0W2fE5-A04DvTcd?usp=sharing) 

Put the downloaded weights on `$PROJECT_ROOT$/output/checkpoints/train/ostrack`

Change the corresponding values of `lib/test/evaluation/local.py` to the actual benchmark saving paths

Some testing examples:
- LaSOT or other off-line evaluated benchmarks (modify `--dataset` correspondingly)
```
python tracking/test.py ostrack vitb_384_mae_ce_32x4_ep300 --dataset lasot --threads 16 --num_gpus 4
python tracking/analysis_results.py # need to modify tracker configs and names
```
- GOT10K-test
```
python tracking/test.py ostrack vitb_384_mae_ce_32x4_got10k_ep100 --dataset got10k_test --threads 16 --num_gpus 4
python lib/test/utils/transform_got10k.py --tracker_name ostrack --cfg_name vitb_384_mae_ce_32x4_got10k_ep100
```
- TrackingNet
```
python tracking/test.py ostrack vitb_384_mae_ce_32x4_ep300 --dataset trackingnet --threads 16 --num_gpus 4
python lib/test/utils/transform_trackingnet.py --tracker_name ostrack --cfg_name vitb_384_mae_ce_32x4_ep300
```


