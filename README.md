# Dual-decoder data decoupling training for semi-supervised medical image segmentation

**Authors:**
Bing Wang,Taifeng Huang,Ying Yang,Junhai Zhai,Xin Zhang
# Usage
We provide code, models, data_split and training weights for BrainMRI, COVID-19, LUNA16 and [ACDC](https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html) datasets.

#### 1. Clone the repo.;

```
git@github.com:TaifengHuang/DD-Net.git
```
#### 2. Put the data in './DD-Net/data/';
#### 3. Training;
```
cd DD-Net/code
python train_ddnet_acdc.py    #for ACDC training
python train_ddnet_2D.py    #for BrainMRI, COVID-19 and LUNA16 training
```
#### 4. Testing;
```
python test_ACDC.py    #for ACDC testing
python test_2D.py    #for BrainMRI, COVID-19 and LUNA16 testing
```
# Citation
# Acknowledgements:
Our code is adapted from [SSL4MIS](https://github.com/HiLab-git/SSL4MIS), [MC-Net](https://github.com/ycwu1997/MC-Net/blob/main/README.md). Thanks for these authors for their valuable works and hope our model can promote the relevant research as well.


