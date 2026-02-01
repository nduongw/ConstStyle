## Instance Retrieval

Person Re-identification is adopted for evaluating the performance on the instance retrieval task. 



### Prerequisites
+ Create a conda virtual environment and activate it.
```
conda create --name uncertainty python=3.6
conda activate uncertainty
```
+ Install Pytorch and torchvision following the [official instructions](https://pytorch.org/), e.g.,
```
conda install pytorch==1.5.0 torchvision==0.6.0 cudatoolkit=10.1 -c pytorch
```
+ Install the dependent libraries.
```
pip install numpy six h5py Pillow scipy scikit-learn metric-learn
```


### Prepare Datasets

+ Download the raw datasets [DukeMTMC-reid](https://arxiv.org/abs/1609.01775) and [Market-1501](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Zheng_Scalable_Person_Re-Identification_ICCV_2015_paper.pdf), then unzip them under the directory like:

```
instance-retrieval/examples/data
├── dukemtmc
│   └── DukeMTMC-reID
└── market1501
    └── Market-1501-v15.09.15
...

### Getting Started
+ Train models on `dukemtmc` and test on unseen `market1501`:

```
sh scripts/train.sh dukemtmc market1501 uresnet50
```
+ Train models on `market1501` and test on unseen `dukemtmc`:
```
sh scripts/train.sh market1501 dukemtmc uresnet50
```

### Acknowledge

The implementation is based on [MMT](https://github.com/yxgeee/MMT), [DSU](https://github.com/lixiaotong97/DSU) and [CSU](https://github.com/freshman97/CSU). We thank them for their excellent projects.
