PyTorch implementation of the article "[Driver Anomaly Detection: A Dataset and Contrastive Learning Approach](https://arxiv.org/pdf/2009.14660.pdf)".

# Driver-Anomaly-Detection

<div align="center" style="width:image width px;">
  <img  src="https://github.com/okankop/Driver-Anomaly-Detection/blob/master/visual/drinking_front.gif" width=350 alt="demo_front">
  <img  src="https://github.com/okankop/Driver-Anomaly-Detection/blob/master/visual/drinking_top.gif" width=350 alt="demo_top">
  
  <img  src="https://github.com/okankop/Driver-Anomaly-Detection/blob/master/visual/adjusting_mirror_front.gif" width=350 alt="demo_front">
  <img  src="https://github.com/okankop/Driver-Anomaly-Detection/blob/master/visual/adjusting_mirror_top.gif" width=350 alt="demo_top">
  
  <img  src="https://github.com/okankop/Driver-Anomaly-Detection/blob/master/visual/pick_up_sth_front.gif" width=350 alt="demo_front">
  <img  src="https://github.com/okankop/Driver-Anomaly-Detection/blob/master/visual/pick_up_sth_top.gif" width=350 alt="demo_top">
</div>


## Illustration of Applied Methodology

<p align="center"><img src="https://github.com/okankop/Driver-Anomaly-Detection/blob/master/visual/visual.png" align="middle" width="550" title="applied methodology" /><figcaption>Fig. 1:  Using contrastive learning, normal driving template vector <b>v<sub>n</sub></b> is learnt during training. At test time, any clip whose embedding is deviating more than threshold γ from normal driving template <b>v<sub>n</sub></b> is considered as anomalous driving. Examples are taken from new introduced Driver Anomaly Detection (DAD) dataset for front (left) and top (right) views on depth modality. 
 </figcaption></figure></p>

## Dataset Preperation

The DAD dataset can be downloaded from its [official website](https://www.ei.tum.de/mmk/dad/) or [Gdrive](https://drive.google.com/drive/folders/1TVzNODlKiRKj60cwadnXofEsBkCDl1DZ?usp=sharing).


## Running the Code

Model configurations are given as follows:
```
  ShuffleNetV1-2.0x : --model_type shufflenet    --width_mult 2.0 
  ShuffleNetV2-2.0x : --model_type shufflenetv2  --width_mult 2.0
  MobileNetV1-2.0x  : --model_type mobilenet     --width_mult 2.0
  MobileNetV2-1.0x  : --model_type mobilenetv2   --width_mult 1.0
  ResNet-18         : --model_type resnet   --model_depth 18   --shortcut_type A
  ResNet-50         : --model_type resnet   --model_depth 50   --shortcut_type B
  ResNet-101        : --model_type resnet   --model_depth 101  --shortcut_type B
```
Please check all 3D CNN models in the 'models' folder and run the code by providing the necessary parameters. An example run is given as follows:
- Training from scratch:
```
python main.py \
  --root_path /usr/home/sut/datasets/DAD/DAD/ \
  --mode train \
  --view top_depth \
  --model_type resnet \
  --model_depth 18 \
  --shortcut_type A \
  --pre_train_model False \
  --n_train_batch_size 10 \
  --a_train_batch_size 150 \
  --val_batch_size 70\
  --learning_rate 0.01 \
  --epochs 250 \
  --cal_vec_batch_size 100 \
  --tau 0.1 \
  --train_crop 'random' \
  --n_scales 5 \
  --downsample 2 \
  --n_split_ratio 1.0 \
  --a_split_ratio 1.0 \
```
- Resuming training from a checkpoint: (the resumed models consist of a base encoder model and a projection head model)
```
python main.py \
  --root_path /usr/home/sut/datasets/DAD/DAD/ \
  --mode train \
  --view top_depth \
  --model_type resnet \
  --model_depth 18 \
  --shortcut_type A \
  --pre_train_model False \
  --n_train_batch_size 10 \
  --a_train_batch_size 150 \
  --val_batch_size 70\
  --learning_rate 0.01 \
  --epochs 250 \
  --cal_vec_batch_size 100 \
  --tau 0.1 \
  --resume_path 'best_model_resnet_top_depth.pth' \
  --resume_head_path 'best_model_resnet_top_depth_head.pth' \
  --train_crop 'random' \
  --n_scales 5 \
  --downsample 2 \
  --n_split_ratio 1.0 \
  --a_split_ratio 1.0 \
```
- Training from a pretrained model. Find the corredponding model type in models.py and set the 'pre_model_path' as the path of the pretrained model. Then set '--pre_train_model True ':

  In model.py file:
```
pre_model_path = './premodels/kinetics_resnet_18_RGB_16_best.pth'
```
```
python main.py \
  --root_path /usr/home/sut/datasets/DAD/DAD/ \
  --mode train \
  --view top_depth \
  --model_type resnet \
  --model_depth 18 \
  --shortcut_type A \
  --pre_train_model True \
  --n_train_batch_size 10 \
  --a_train_batch_size 150 \
  --val_batch_size 70\
  --learning_rate 0.01 \
  --epochs 250 \
  --cal_vec_batch_size 100 \
  --tau 0.1 \
  --train_crop 'random' \
  --n_scales 5 \
  --downsample 2 \
  --n_split_ratio 1.0 \
  --a_split_ratio 1.0 \
```
**Augmentations**

There are several augmentation techniques available. Please check spatial_transforms.py and temporal_transforms.py for the details of the augmentation methods.


**Evaluation**

You should train four models of two views and two modalities separatly. After training, set '--mode test \', the accuracy and AUC of these models and the results after fusion will be shown.


## Running the Demo

Please download the <i>pretrained model weights</i> and <i>normal driving templates</i> from "[here](https://drive.google.com/drive/folders/1HekFrIvjUTRNZpbsSJd69h5p29wVWP8o?usp=sharing)".

Modify the paths in [demo_live.py](https://github.com/okankop/Driver-Anomaly-Detection/blob/master/demo_live.py) according to the dataset path and pretrainde model paths.

Run the following command to start live demo:
```
python live_demo.py
```




## Citation

Please cite the following article if you use this code or pre-trained models:

```bibtex
@inproceedings{kopuklu2021driver,
  title={Driver anomaly detection: A dataset and contrastive learning approach},
  author={Kopuklu, Okan and Zheng, Jiapeng and Xu, Hang and Rigoll, Gerhard},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={91--100},
  year={2021}
}
```

## Acknowledgement
We thank Yonglong Tian for releasing his [codebase](https://github.com/HobbitLong/CMC), which we build our work on top.
