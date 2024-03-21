# NTIRE2024_GoodGame
> This repository is the official [NTIRE 2024](https://cvlai.net/ntire/2024/) implementation of Team GoodGame in [Stereo Image Super-Resolution Challenge - Track 2 Constrained SR & Realistic Degradation](https://codalab.lisn.upsaclay.fr/competitions/17246).
> The restoration results of the testing images can be downloaded from [here](https://pan.baidu.com/s/1jfYxN4fOrO__zpWSbQXteA?pwd=6666).
Our pretrained models can be downloaded from [here](https://pan.baidu.com/s/1N7ehWJOkc0CNKu2UOMI9QA?pwd=6666).
## Usage
Installation environment
```
python 3.8
pytorch 1.11.0
cuda 11.3
```

```
git clone git@github.com:Yuqi-Miao/NTIRE_GoodGame_track2.git
cd NTIRE_GoodGame_track2
pip install -r requirements.txt
python setup.py develop --no_cuda_ext
```

Modify the configuration file `options/train/CVHSSR_Sx4.yml` and `options/test/CVHSSR_Sx4.yml` as follows:
```
Train
dataroot_gt: ./data/Flickr1024/trainx4 # replace your dataset path
dataroot_lq: ./data/Flickr1024/trainx4 # replace your dataset path

Test
dataroot_gt: ./data/Flickr1024/Stereo_test/KITTI2012/hr # replace your dataset path
dataroot_lq: ./data/Flickr1024/Stereo_test/KITTI2012/lr_x4 # replace your dataset path
```
### Train
```
python -m torch.distributed.launch --nproc_per_node=2 --master_port=4329 basicsr/train.py -opt options/train/CVHSSR_Sx4.yml --launcher pytorch
```


### infer
Modify the configuration file `options/test/CVHSSR_Sx4.yml` as follows:
```
pretrain_network_g: ../pretrain/besk_ckpt.pth # replace your model checkpoint path
```
Then you can infer
```
python basicsr/demo_ssr.py -opt options/test/CVHSSR_Sx4.yml --img_pair_paths ntire_test_stereo_image_path --output_dir output_image_path
```
### Parameters
```
python basicsr/models/archs/CVHSSR_arch.py
```