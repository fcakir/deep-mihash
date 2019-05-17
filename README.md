# Hashing with Binary Matrix Pursuit
This branch contains the MATLAB implementation of the following paper. For development purposes, it is maintained separately from [deep-mihash](https://github.com/fcakir/deep-mihash).

**Hashing with Binary Matrix Pursuit**,  
Fatih Cakir, Kun He, and Stan Sclaroff.
**ECCV** 2018 ([conference page](http://openaccess.thecvf.com/content_ECCV_2018/html/Fatih_Cakir_Hashing_with_Binary_ECCV_2018_paper.html))

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

## Example Commands
* Only **CIFAR-10** and **NUSWIDE** datasets are supported currently. Support for **LabelMe** and **ImageNet100** will be added soon.
* The main functions for experimenting is `demo.m`. The main arguments can be found in `get_opts.m`. 
* Below are examples commands to replicate some of the results in the HBMP paper above. Please refer to *Section 4* of the paper and `get_opts.m` for experimental setting and parameter details. 
    * **CIFAR-1, 12 bits, AP** *(Table 1)*: 
    	* `demo('cifar',12, 'vggf', 'obj', 'hbmp', 'lr', 1e-3, 'weighted', 1, 'split',1, 'max_iter', 12, 'metrics', 'AP')`
    * **CIFAR-2, 32 bits, AP** *(Table 2)* : 
   	* `demo('cifar', 32,'vggf', 'obj', 'hbmp', 'lr', 1e-3, 'weighted', 1, 'split', 2, 'max_iter', 32, 'metrics', 'AP')`
    * **NUSWIDE-1, 32 bits, AP@5K** *(Table 1)* : 
   	* `demo('nus',32,'vggf_ft', 'obj', 'hbmp', 'lr', 1e-3, 'weighted', 1, 'max_iter', 32, 'split', 1, 'metrics', 'AP@5000')`
    * **NUSWIDE-1, 32 bits, AP@5K** *(constant, Table 5)* : 
   	* `demo('nus',32,'vggf_ft', 'obj', 'hbmp', 'lr', 1e-3, 'weighted', 0, 'max_iter', 32, 'split', 1, 'metrics', 'AP@5000')`
    * **NUSWIDE-2, 32 bits AP@50K** *(Table 2)* : 
    	* `demo('nus',32,'vggf_ft', 'obj', 'hbmp', 'lr', 1e-2, 'weighted', 1, 'max_iter', 32, 'split', 2, 'metrics', 'AP@50000')`
    * **NUSWIDE-2, 32 bits AP@50K** *(constant, Table 2)* : 
    	* `demo('nus',32,'vggf_ft', 'obj', 'hbmp', 'lr', 1e-2, 'weighted', 0, 'max_iter', 32, 'split', 2, 'metrics', 'AP@50000')`
    * **NUSWIDE-1, 32 bits NDCG** *(Table 4)* : 
    	* `demo('nus', 32,'vggf_ft', 'obj', 'hbmp', 'lr', 1e-3, 'weighted', 1, 'max_iter', 32, 'split', 1, 'metrics', 'NDCG')`
    * **NUSWIDE-1, 32 bits NDCG** (constant, Table 6)  : 
     * `demo('nus', 32,'vggf_ft', 'obj', 'hbmp', 'lr', 1e-3, 'weighted', 0, 'max_iter', 32, 'split', 1, 'metrics', 'NDCG')`


## License
MIT License, see `LICENSE`

## Contact
For questions and comments, feel free to contact: fcakirs@gmail.com

## Notes
- Special thanks to [Kun](http://github.com/kunhe).
