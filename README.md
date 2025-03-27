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

| Tracker     | LaSOT (AUC) | TNL2K (AUC)       | OTB-Lang(AUC) | LaSOT-ext(AUC) | MGIT(SR_IOU) |GOT-10K(AO) |
|:-----------:|:-----------:|:-----------------:|:-----------:|:-----------:|:-----------:|:-----------:|
| DUTrack-384 | 74.1        | 65.6              | 71.3        | 52.5       |71.0        |77.8        |
| DUTrack-256 | 73.0        | 64.9              | 70.9        |50.5        |70.0        |76.7        |



## Install the environment
**Conda env**: Use the Anaconda (CUDA 11.2)
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
Finally, you need to set the model paths for BLIP and BERT in the configuration file.
```
cfg.MODEL.BERT_DIR: path for [BERT](https://huggingface.co/google-bert/bert-base-uncased) 
cfg.MODEL.BLIP_DIR: path for [BLIP](https://huggingface.co/Salesforce/blip-image-captioning-base) 
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
            |-- lasot_train_concise
        -- got10k
            |-- test
            |-- train
                |-- got-10k_train_concise
                ...
            |-- val
        -- TNL2K
            |-- train
            |-- test
            |-- tnl2k_train_concise

   ```
**Notice** ：The files lasot_train_concise, got-10k_train_concise, and tnl2k_train_concise come from [DTVLT](http://videocube.aitestunion.com/) and contain partial language descriptions from the three datasets.

## Training
Download [pre-trained](https://drive.google.com/drive/folders/15g87STgG4ZWr6ZTnKJdGL3Z-gwG0SVL8?usp=sharing) and put it under `$PROJECT_ROOT$/pretrained_models` 

```
python tracking/train.py --script dutrack --config dutrack_256_full --save_dir ./output --mode multiple --nproc_per_node 2 --use_wandb 0
```

Replace `--config` with the desired model config under `experiments/dutrack`. 


## Evaluation
Download the model weights from [Models](https://drive.google.com/drive/folders/1edieIddvSzy9219F4WShajvY4UmkKJV4?usp=sharing) 

Put the downloaded weights on `$PROJECT_ROOT$/output/checkpoints/train/dutrack/`

Change the corresponding values of `lib/test/evaluation/local.py` to the actual benchmark saving paths

Some testing examples:
- LaSOT or other off-line evaluated benchmarks (modify `--dataset` correspondingly)
```
python tracking/test.py dutrack dutrack_256_full --dataset lasot --threads 16 --num_gpus 2
python tracking/analysis_results.py # need to modify tracker configs and names
```
- GOT10K-test
```
python tracking/test.py dutrack dutrack_256_full --dataset got10k_test --threads 16 --num_gpus 2
python lib/test/utils/transform_got10k.py --tracker_name dutrack --cfg_name dutrack_256_full
```
- MGIT
```
python tracking/test.py dutrack dutrack_256_full --dataset --dataset mgit --threads 16 --num_gpus 2
```

## Test FLOPs, and Speed
**Note**: The speeds reported in our paper were tested on a single V100 GPU.
```
# Profiling DUTrack-256
python tracking/profile_model.py --script dutrack --config dutrack_256_full
# Profiling DUTrack-384
python tracking/profile_model.py --script dutrack --config dutrack_384_full
```
## Debug
Simply set `--debug 1` during inference for debug, e.g.:
```
python tracking/test.py dutrack dutrack_256_full --dataset --dataset mgit --threads 1 --num_gpus 1 --debug 1
```
**Note** ：when setting `--debug 1`, the tracking results will not be saved as a txt file
## Acknowledgments
* Thanks for the [ODTrack](https://github.com/GXNU-ZhongLab/ODTrack) and [PyTracking](https://github.com/visionml/pytracking) library, which helps us to quickly implement our ideas.
* We use the implementation of the fast_ITPN from the [sunsmarterjie](https://github.com/sunsmarterjie/iTPN) repo.

## Citation
If our work is useful for your research, please consider citing:

```Bibtex
@inproceedings{Li2025dutrack,
  title={Dynamic Updates for Language Adaptation in Visual-Language Tracking},
  author={Xiaohai Li and and Bineng Zhong and Qihua Liang and Zhiyi Mo and Jian Nong and Shuxiang Song},
  booktitle={CVPR},
  year={2025}
}  
