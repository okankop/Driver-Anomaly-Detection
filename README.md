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


## Citation

Please cite the following article if you use this code or pre-trained models:

```bibtex
@article{kopuklu2019driver,
  title={Driver Anomaly Detection: A Dataset and Contrastive Learning Approach},
  author={K{\"o}p{\"u}kl{\"u}, Okan and Zheng, Jiapeng and Xu, Hang and Rigoll, Gerhard},
  journal={arXiv preprint arXiv:2009.14660},
  year={2020}
}

## Acknowledgement
We thank Yonglong Tian for releasing his [codebase](https://github.com/HobbitLong/CMC), which we build our work on top.
