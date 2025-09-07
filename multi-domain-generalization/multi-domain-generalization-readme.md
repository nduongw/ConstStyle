## Multi-domain Generalization


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
pip install flake8 yapf isort yacs gdown tb-nightly future scipy scikit-learn
```
+ Setup the environment
```angular2html
python setup.py develop
```


### Dataset Preparation

Download the datasets from this [link](https://drive.google.com/file/d/1nIer0Zjj5hn5CcPmdW86xQ9zZm1VDqpw/view?usp=sharing), then place them under the directory like:

```
multi-domain-generalization/DATA
├── pacs
└── digit5
└── cifar10
...
```

### Getting Started
+ You can run the following script at the following path: `multi-domain-generalization/scripts/pacs` to run all the experiments with a specific algorithm.

```
bash scripts/pacs/conststyle.sh
```

### Acknowledge

The implementation is based on [Dassl](https://github.com/KaiyangZhou/Dassl.pytorch), [DSU](https://github.com/lixiaotong97/DSU) and [CSU](https://github.com/freshman97/CSU). We thank them for their excellent projects.
