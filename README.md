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

## Running the Code
Model configurations are given as follows:

ShuffleNetV1-2.0x : --model_type shufflenet    --width_mult 2.0
ShuffleNetV2-2.0x : --model_type shufflenetv2  --width_mult 2.0
MobileNetV1-2.0x  : --model_type mobilenet     --width_mult 2.0
MobileNetV2-1.0x  : --model_type mobilenetv2   --width_mult 1.0
ResNet-18         : --model_type resnet   --model_depth 18   --shortcut_type A
ResNet-50         : --model_type resnet   --model_depth 50   --shortcut_type B
ResNet-101        : --model_type resnet   --model_depth 101  --shortcut_type B


## Citation

Please cite the following article if you use this code or pre-trained models:

```bibtex
@article{kopuklu2019driver,
  title={Driver Anomaly Detection: A Dataset and Contrastive Learning Approach},
  author={K{\"o}p{\"u}kl{\"u}, Okan and Zheng, Jiapeng and Xu, Hang and Rigoll, Gerhard},
  journal={arXiv preprint arXiv:2009.14660},
  year={2020}
}
```

## Acknowledgement
We thank Yonglong Tian for releasing his [codebase](https://github.com/HobbitLong/CMC), which we build our work on top.
