# Hashing with Mutual Information
This repository contains the MATLAB implementation of the following paper:

"Hashing with Mutual Information",  
Fatih Cakir*, Kun He*, Sarah Adel Bargal, and Stan Sclaroff.
TPAMI (revision) ([arXiv](https://arxiv.org/abs/1803.00974))

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

## Setup
* Install or symlink [MatConvNet](http://www.vlfeat.org/matconvnet/) at `./matconvnet` (for training CNNs)
* Install or symlink [VLFeat](http://www.vlfeat.org/)  at `./vlfeat`. 
* [Download](https://www.dropbox.com/s/7ovbuheetguinj3/data.tar.gz?dl=0) necessary datasets. **Note**: Large file ~35GB.
* In the root folder, run `startup.m`
## Example Commands
* The main functions for experimenting are `demo_AP.m` and `demo_imagenet.m`. 
* The main arguments can be found in `get_opts.m`. 
* Below are examples commands to replicate some of the results in the paper. Please refer to *Section 5* of the paper and `get_opts.m` for experimental setting and parameter details. 
    * CIFAR-1 32 bits: `demo_AP('cifar',32,'vggf','split',1,'nbins',32,'sigmf', 
    [1 0],'lr', 1e-3,'lrdecay',0.5,'lrstep',50,'epoch',100,'obj','mi','gpus',0,'testInterval',10, 'batchSize', 256, 'continue', false, 'metrics', 'AP', 'ep1', true)`
    You should get ~0.90 mAP at 100 epochs. A MATLAB *diary* will be saved to the corresponding experimental folder. 
    Here is an example diary for the above experiment. Note that there might be slight differences. 



## License
MIT License, see `LICENSE`

## Contact
For questions and comments, feel free to contact:

fcakirs@gmail.com

## Notes
- This implementation extends [MIHash](http://github.com/fcakir/mihash), and is specifically designed for deep learning experiments.
