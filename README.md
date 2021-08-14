This repo contains GANs review for topics of computer vision and time series
=======

## News 
[2021/07/11] Our preprint ["Generative Adversarial Networks in Time Series: A Survey and Taxonomy", Eoin Brophy and Zhengwei Wang and Qi She and Tomas E. Ward](https://arxiv.org/pdf/2107.11098.pdf) is out. This work is currently in progress.

[2021/02/14] Our paper [“Generative Adversarial Networks in Computer Vision: A Survey and Taxonomy” Zhengwei Wang and Qi She and Tomas E. Ward](https://dl.acm.org/doi/abs/10.1145/3439723) ([arxiv version](https://arxiv.org/pdf/1906.01529.pdf)) has been published at **ACM Computing Surveys**，and we will continue to polish this work into the 5th version. Details of selected papers and codes can refer to [GAN_CV folder](https://github.com/sheqi/GAN_Review/tree/master/GAN_CV).

[2020/11/24] Our paper [“Generative Adversarial Networks in Computer Vision: A Survey and Taxonomy” Zhengwei Wang and Qi She and Tomas E. Ward](https://arxiv.org/pdf/1906.01529.pdf) gets acceptted into **ACM Computing Surveys**，and we will continue to polish this work into the 5th version.

[2020/06/20] We have updated our **4th** version of GAN survey for **computer vision** paper ! It inlcudes more recent GANs proposed at CVPR, ICCV 2019/2020, more intuitive visualization of GAN Taxonomy.

[2020/10/04] GANs related to our latest paper will be updated shortly. 


## Generative Adversarial Networks in Computer Vision

<p align='center'><img src='./pic/GANs_taxonomy.png' width=1000></img>

A Survey and Taxonomy of the Recent GANs Development in computer vision. Please refer to the details in recent review paper [“Generative Adversarial Networks in Computer Vision: A Survey and Taxonomy” Zhengwei Wang and Qi She and Tomas E. Ward](https://dl.acm.org/doi/abs/10.1145/3439723) ([arxiv version](https://arxiv.org/pdf/1906.01529.pdf)). We also provide a list of papers related to GANs on computer vision in the GAN_CV.csv file.

If you find this useful in your research, please consider citing:

    @article{wang2021generative,
      title={Generative Adversarial Networks in Computer Vision: A Survey and Taxonomy},
      author={Wang, Zhengwei and She, Qi and Ward, Tomas E},
      journal={ACM Computing Surveys (CSUR)},
      volume={54},
      number={2},
      pages={1--38},
      year={2021},
      publisher={ACM New York, NY, USA}
    }


We have classified the two GAN-variants research lines based on recent GAN developments, below we provide a summary and the demo code of these models. We have tested the codes below and tried to summary some of <b>lightweight</b> and <b>easy-to-reuse</b> module of state-of-the-art GANs.

## Architecture-variant GANs
LAPGAN:  
https://github.com/jimfleming/LAPGAN (TensorFlow)  
https://github.com/AaronYALai/Generative_Adversarial_Networks_PyTorch (PyTorch)

DCGAN:   
https://github.com/carpedm20/DCGAN-tensorflow (TensorFlow)  
https://github.com/last-one/DCGAN-Pytorch (PyTorch)

BEGAN:  
https://github.com/carpedm20/BEGAN-tensorflow (TensorFlow)  
https://github.com/anantzoid/BEGAN-pytorch (PyTorch)

PROGAN:  
https://github.com/tkarras/progressive_growing_of_gans (TensorFlow)  
https://github.com/nashory/pggan-pytorch (PyTorch)

SAGAN:  
https://github.com/brain-research/self-attention-gan (TensorFlow)   
https://github.com/heykeetae/Self-Attention-GAN (PyTorch)

BigGAN:    
https://github.com/taki0112/BigGAN-Tensorflow (TensorFlow)  
https://github.com/ajbrock/BigGAN-PyTorch (PyTorch)

Your Local GAN:  
https://github.com/giannisdaras/ylg (TensorFlow)  
https://github.com/188zzoon/Your-Local-GAN (PyTorch)

AutoGAN:  
https://github.com/VITA-Group/AutoGAN (PyTorch)

MSG-GAN:  
https://github.com/akanimax/msg-stylegan-tf (TensorFlow)  
https://github.com/akanimax/msg-gan-v1 (PyTorch)


## Loss-variant GANs
WGAN:  
https://github.com/ChengBinJin/WGAN-TensorFlow (TensorFlow)   
https://github.com/Zeleni9/pytorch-wgan (PyTorch)

WGAN-GP:  
https://github.com/changwoolee/WGAN-GP-tensorflow (TensorFlow)   
https://github.com/caogang/wgan-gp (PyTorch)

LSGAN:  
https://github.com/xudonmao/LSGAN (TensorFlow)  
https://github.com/meliketoy/LSGAN.pytorch (PyTorch)

f-GAN:  
https://github.com/LynnHo/f-GAN-Tensorflow (TensorFlow)

UGAN:  
https://github.com/gokul-uf/TF-Unrolled-GAN (TensorFlow)   
https://github.com/andrewliao11/unrolled-gans (PyTorch)

LS-GAN:  
https://github.com/maple-research-lab/lsgan-gp-alt (TensorFlow)  
https://github.com/maple-research-lab/glsgan-gp (PyTorch)

MRGAN:  
https://github.com/wiseodd/generative-models/tree/master/GAN/mode_regularized_gan (TensorFlow and PyTorch) 

Geometric GAN:  
https://github.com/lim0606/pytorch-geometric-gan (PyTorch) 

RGAN:  
https://github.com/AlexiaJM/RelativisticGAN (TensorFlow and PyTorch)

SN-GAN:  
https://github.com/taki0112/Spectral_Normalization-Tensorflow (TensorFlow) 
https://github.com/christiancosgrove/pytorch-spectral-normalization-gan (PyTorch)

RealnessGAN:  
https://github.com/taki0112/RealnessGAN-Tensorflow (TensorFlow)  
https://github.com/kam1107/RealnessGAN (PyTorch)

Sphere GAN:  
https://github.com/taki0112/SphereGAN-Tensorflow (TensorFlow)  
https://github.com/Dotori-HJ/SphereGAN-Pytorch-implementation (PyTorch)

Self-supervised GAN:  
https://github.com/zhangqianhui/Self-Supervised-GANs (TensorFlow)  
https://github.com/vandit15/Self-Supervised-Gans-Pytorch (PyTorch)


 
