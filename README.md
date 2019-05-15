# Hashing with Mutual Information
This repository contains the MATLAB implementation of the following paper:

**Hashing with Mutual Information**,  
Fatih Cakir*, Kun He*, Sarah Adel Bargal, and Stan Sclaroff.
**TPAMI** 2019 (to appear) ([arXiv](https://arxiv.org/abs/1803.00974))

If you use this code in your research, please cite:
```
@inproceedings{Cakir_deep_mihash,
  author    = {Fatih Cakir and Kun He and Sarah Adel Bargal and Stan Sclaroff},
  title     = {Hashing with Mutual Information},
  journal   = {CoRR},
  volume    = {abs/1803.00974},
  year      = {2018},
}
```

:warning: The **[hbmp](https://github.com/fcakir/deep-mihash/tree/hbmp)** branch contains the implementation of the following paper:

**Hashing with Binary Matrix Pursuit**,  
Fatih Cakir, Kun He, and Stan Sclaroff.
**ECCV** 2018 ([arXiv](http://openaccess.thecvf.com/content_ECCV_2018/html/Fatih_Cakir_Hashing_with_Binary_ECCV_2018_paper.html))

If you use this code in your research, please cite:
```
@InProceedings{Cakir_2018_ECCV,
author = {Cakir, Fatih and He, Kun and Sclaroff, Stan},
title = {Hashing with Binary Matrix Pursuit},
booktitle = {The European Conference on Computer Vision (ECCV)},
year = {2018}
}
```

## Setup
* Install or symlink [MatConvNet](http://www.vlfeat.org/matconvnet/) at `./matconvnet` (for training CNNs)
* Install or symlink [VLFeat](http://www.vlfeat.org/)  at `./vlfeat`
* [Download](https://www.dropbox.com/s/7ovbuheetguinj3/data.tar.gz?dl=0) necessary datasets to `./cachedir/data/` **Note**: Large file ~35GB
* [Download](https://www.dropbox.com/s/n2nxibo0ckdo6hp/models.tar.gz?dl=0) necessary model files to `./cachedir/models/`
* Create `./cachedir/results/` folder to hold experimental data
* In the root folder, run `startup.m`

:warning: Follow the setup instructions for **HBMP** in the **[hbmp](https://github.com/fcakir/deep-mihash/tree/hbmp)** branch.

## Example Commands
* The main functions for experimenting are `demo_imagenet.m` (for the ImageNet100 benchmark) and `demo_AP.m` (for other benchmarks such as CIFAR-10 and NUSWIDE). 
* The main arguments can be found in `get_opts.m`. 
* Below are examples commands to replicate some of the results in the paper. Please refer to *Section 5* of the paper and `get_opts.m` for experimental setting and parameter details. 
    * **CIFAR-1 32 bits**: `demo_AP('cifar',32,'vggf','split',1,'nbins',32,'sigmf', 
    [1 0],'lr', 1e-3,'lrdecay',0.5,'lrstep',50,'epoch',100,'obj','mi','gpus',0,'testInterval',10, 'batchSize', 256, 'continue', false, 'metrics', 'AP', 'ep1', true)`
    A MATLAB *diary* will be saved to the corresponding experimental folder. 
        * [Download](https://www.dropbox.com/s/v3wzo1qwmgcq3uv/diary_003.txt?dl=0) an example diary for the above experiment. You should get **~0.78-0.79** mAP at 100 epochs. Note that there might be slight differences. 
    * **CIFAR-2 32 bits**: `demo_AP('cifar',32,'vggf','split',2,'nbins',12,'sigmf', 
    [30 0],'lr', 2e-3,'lrdecay',0.5,'lrstep',50,'epoch',100,'obj','mi','gpus',0,'testInterval',10, 'batchSize', 256, 'continue', false, 'metrics', 'AP', 'ep1', true)`
        * [Download](https://www.dropbox.com/s/s7ga1wtq6n2qkyh/diary_001.txt?dl=0) an example diary for the above experiment. You should get **~0.93-0.94** mAP at 100 epochs. Note that there might be slight differences. 
    * **NUSWIDE-1 32 bits** : `demo_AP('nus',32,'vggf_ft','split',1, 'nbins',16,'sigmf', 
    [1 0],'lr', 0.05,'lrdecay',0.5,'lrstep',50, 'lrmult', 0.01, 'epoch',120,'obj','mi','gpus',0,'testInterval',10, 'batchSize', 250, 'continue', false, 'metrics', {'AP','AP@5000', 'AP@50000'}, 'ep1', true)`
        * [Download](https://www.dropbox.com/s/gte6e5ikk5jpb5j/nus-1-diary.txt?dl=0) an example diary for the above experiment. You should get **~0.82-0.83** mAP@5K at 120 epochs. Note that there might be slight differences.
    * **NUSWIDE-2 32 bits** : `demo_AP('nus',32,'vggf_ft','split',2, 'nbins',16,'sigmf', 
    [1 0],'lr', 0.01,'lrdecay',0.5,'lrstep',50, 'lrmult', 0.01, 'epoch',100,'obj','mi','gpus',0,'testInterval',5, 'batchSize', 250, 'continue', false, 'metrics', {'AP','AP@5000', 'AP@50000'}, 'ep1', true)`
        * [Download](https://www.dropbox.com/s/wgwx1n8swwme38g/nus-2-diary.txt?dl=0) an example diary for the above experiment. You should get **~0.81-0.82** mAP@50K at 100 epochs. Note that there might be slight differences.
    * **ImageNet100 48 bits**: `demo_imagenet(48, 'alexnet_ft', 'split', 1 , 'nbins', 16, 'gpus', 1, 'lr', 0.1, 'lrdecay', 0.05, 'lrmult', 0.01, 'lrstep', 100, 'nbins', 16, 'sigmf', [10 0], 'testInterval', 25, 'continue', true, 'normalize', 1, 'metrics', {'AP', 'AP@1000'}, 'epoch', 125)`
        * [Download](https://www.dropbox.com/s/34xb6wea3a7jsas/imagenet100-diary.txt?dl=0) an example diary for the above experiment. You should get **~0.68-0.69** mAP@1K at 125 epochs. Note that there might be slight differences.
## License
MIT License, see `LICENSE`

## Contact
For questions and comments, feel free to contact:

fcakirs@gmail.com

## Notes
- This implementation extends [MIHash](http://github.com/fcakir/mihash), and is specifically designed for deep learning experiments. Special thanks to [Kun](http://github.com/kunhe) and [Sarah](https://github.com/sbargal).
