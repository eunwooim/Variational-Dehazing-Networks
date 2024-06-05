![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Git](https://img.shields.io/badge/git-%23F05033.svg?style=for-the-badge&logo=git&logoColor=white)
# Deep Variational Bayesian Modeling of Haze Degradation Process

This is official PyTorch implementation for the paper "Deep Variational Bayesian Modeling of Haze Degradation".
This repository contains the official implementation for SANFlow introduced in the following paper:

[[Paper](https://dl.acm.org/doi/10.1145/3583780.3614838)] [[Checkpoints](https://hanyangackr0-my.sharepoint.com/:u:/g/personal/junsung6140_m365_hanyang_ac_kr/EYI6hz3mDOJLkgwl-TAEMxABYpG8FE-5-Q0fSlgUK58mng)]

Authors: Eun Woo Im, Junsung Shin, Sungyong Baik, Tae Hyun Kim

### Abstract
Relying on the representation power of neural networks, most recent works have often neglected several factors involved in haze degradation, such as transmission (the amount of light reaching an observer from a scene over distance) and atmospheric light. These factors are generally unknown, making dehazing problems ill-posed and creating inherent uncertainties. To account for such uncertainties and factors involved in haze degradation, we introduce a variational Bayesian framework for single image dehazing. We propose to take not only a clean image and but also transmission map as latent variables, the posterior distributions of which are parameterized by corresponding neural networks: dehazing and transmission networks, respectively. Based on a physical model for haze degradation, our variational Bayesian framework leads to a new objective function that encourages the cooperation between them, facilitating the joint training of and thereby boosting the performance of each other. In our framework, a dehazing network can estimate a clean image independently of a transmission map estimation during inference, introducing no overhead. Furthermore, our model-agnostic framework can be seamlessly incorporated with other existing dehazing networks, greatly enhancing the performance consistently across datasets and models.


### Citation
If you find our work useful in your research, please cite:

```
@inproceedings{im2023deep,
  title={Deep Variational Bayesian Modeling of Haze Degradation Process},
  author={Im, Eun Woo and Shin, Junsung and Baik, Sungyong and Kim, Tae Hyun},
  booktitle={Proceedings of the 32nd ACM International Conference on Information and Knowledge Management},
  pages={895--904},
  year={2023}
}
```