# While Editting......

# Fully Convolutional Auto-Encoding Generative Adversarial Networks for 3D Multi Object Scenes
This repository contains the source code for Fully Convolutional Auto-Encoding Generative Adversarial Networks for 3D Multi Object Scenes which is my work at Stanford AI Lab as a visiting scholar.  
Special thanks to Christopher Choi and Prof. Silvio Savarese.

<img src="https://github.com/yunishi3/FC-alphaGAN/blob/master/Images/Interpolation.gif" width="500">

### Contents
0. [Introduction](#Introduction)
0. [Dataset](#Dataset)
0. [Models](#Models)
0. [Experiments](#Experiments)
0. [Evaluations](#Evaluations)
0. [Installation](#Installataion)
0. [References](#References)

## Introduction  
 The generative models utilizing Generative Adversarial Networks or Variational Auto-Encoder are one of the hottest topic in deep learning and computer vision. That doesn’t only enable high quality generation, but also has a lot of possibility for representation learning, feature extraction, and applications to some recognition tasks without supervision using probabilistic spaces and manifolds.  
 Especially 3D multi object generative models, which allows to synthesize a variety of novel 3D multi objects and recognize the objects including shapes, objects and layouts, should be an extremely important tasks for AR/VR and robotics fields.  
 However, 3D generative models are still less developed. Basic generative models of only single objects are published as [1],[2]. But multi objects are not. Therefore I have tried end-to-end 3D multi object generative models using novel generative adversarial network architectures.
 
## Dataset
I used ground truth voxel data of SUNCG dataset. [3]  
http://suncg.cs.princeton.edu/  
  
I modified this dataset as follows.  
 - Downsized from 240x144x240 to 80x48x80.  
 - Got rid of trimming by camera angles.  
 - Chose the scenes that have over 10000 amount of voxels.    

As a result, over 185K scenes were gathered which has 12 classes (empty, ceiling, floor, wall, window, chair, bed, sofa, table, tvs, furn, objs).  
<img src="https://github.com/yunishi3/FC-alphaGAN/blob/master/Images/DatasetImages.png">　　

This dataset is really sparse which means around 92% voxels in this scene are empty class averagely. In addition, this dataset has plenty of varieties such as living room, bathroom, restroom, dining room, garage etc.



## Models
### Network Architecture
The network architecture of this work is fully convolutional auto- encoding generative adversarial networks. This is inspired by 3DGAN[1] and alphaGAN[4]. And fully convolution and classifying multi objects are novel architectures.  
<img src="https://github.com/yunishi3/FC-alphaGAN/blob/master/Images/NetworkArchitecture.png">  

This network combined variatinal auto-encoder with generative adversarial networks. Also the KL divergence loss of variational auto- encoder is replaced with adversarial auto-encoder using code-discriminator as alphaGAN architectuires[4]. In this work, the shape of the latet space is 5x3x5x16 and this is calculated by fully convolution layer. Fully convolution allows to extract features more specifically like semantic segmentation tasks. As a result, fully convolution enables reconstruction performance to get better.  
 Adversarial auto-encoding architecture allows to loosen the constraint of distributions and treat this distribution as implicit. As a result, adversarial auto-encoder enables reconstruction and generation performance to get better. 
In addition, generative adversarial network architecture enables reconstruction and generation performance to get better as well.

<img src="https://github.com/yunishi3/FC-alphaGAN/blob/master/Images/GeneratorNetwork.png">　　

#### -Encoder
The basic architecture of encoder is similar with discriminator network of 3DGAN[1]. The difference is the last layer which is 1x1x1 fully convolution.  
#### -Generator
The basic architecture of generator is also similar with 3DGAN[1] as above figure. The difference is the last layer which has 12 channels and is activated by softmax. Also, the first layer of latent space is flatten. 
#### -Discriminator
The basic architecture of discriminator is also similar with 3DGAN[1]. The difference is the activation layers which are layer normalization.
#### -Code Discriminator
Code discriminator is same as alphaGAN[4] which has 2 hidden layers of 750 dimensions.

### Loss Functions　
* Reconstruction loss  　
	<img src="https://latex.codecogs.com/gif.latex?L_{rec}=\sum&space;_{n}^{class}w_{n}\left(&space;-\gamma&space;x\log&space;\left(&space;x_{rec}\right)&space;-\left(&space;1-\gamma&space;\right)&space;\left(&space;1-x\right)&space;\log&space;\left(&space;1-x_{rec}\right)&space;\right)" />  

	<img src="https://latex.codecogs.com/gif.latex?w" /> is occupancy normalized weights with every batch to weight the importance of small objects. <img src="https://latex.codecogs.com/gif.latex?\gamma" /> is a hyperparameter which weights the relative importance of false positives against false negatives.

* GAN loss
<img src="https://latex.codecogs.com/gif.latex?L_{GAN}\left(&space;D\right)&space;=-\log&space;\left(&space;D(x)&space;\right)&space;-\log&space;\left(&space;1-D(x_{rec})\right)&space;-\log&space;\left(&space;1-D(x_{gen})\right)" />  
<img src="https://latex.codecogs.com/gif.latex?L_{GAN}\left(&space;G\right)&space;=-\log&space;\left(&space;D(x_{rec})&space;\right)&space;-\log&space;\left(&space;D(x_{gen})&space;\right)" />  

* Distribution GAN loss
<img src="https://latex.codecogs.com/gif.latex?L_{cGAN}\left(&space;D\right)&space;=-\log&space;\left(&space;D_{code}(z)&space;\right)&space;-\log&space;\left(&space;1-D_{code}(z_{enc})\right)" />  
<img src="https://latex.codecogs.com/gif.latex?L_{cGAN}\left(&space;E\right)&space;=-\log&space;\left(&space;D_{code}(z_{enc})&space;\right)" />

### Optimization
* Encoder  
	<img src="https://latex.codecogs.com/gif.latex?\min&space;_{E}\left(&space;L_{cGAN}\left(&space;E\right)&space;&plus;\lambda&space;L_{rec}\right)" />  

* Generator  
<img src="https://latex.codecogs.com/gif.latex?\min&space;_{G}\left(&space;\lambda&space;L_{rec}&plus;L_{GAN}\left(&space;G\right)&space;\right)" />  

* Dicriminator  
<img src="https://latex.codecogs.com/gif.latex?\min&space;_{D}\left(&space;L_{GAN}\left(&space;D\right)&space;\right)" />

* Code Discriminator  
<img src="https://latex.codecogs.com/gif.latex?\min&space;_{C}\left(&space;L_{cGAN}\left(&space;D\right)&space;\right)" />  

<img src="https://latex.codecogs.com/gif.latex?\lambda" /> is a hyperparameter which weights the reconstruction loss.


## Experiments
Adam optimaizer was used for each architectures by learning rate 0.0001.
This network was trained for 75000 iterations, batch size was 20.
### Learning curve
<img src="https://github.com/yunishi3/FC-alphaGAN/blob/master/Images/LearningCurve_Recons.png" width="200"><img src="https://github.com/yunishi3/FC-alphaGAN/blob/master/Images/LearningCurve_Gen.png" width="200"><img src="https://github.com/yunishi3/FC-alphaGAN/blob/master/Images/LearningCurve_Discrim.png" width="200">  
<img src="https://github.com/yunishi3/FC-alphaGAN/blob/master/Images/LearningCurve_Code_encode.png" width="200"><img src="https://github.com/yunishi3/FC-alphaGAN/blob/master/Images/LearningCurve_Code_discrim.png" width="200">  


### Visualization
#### -Reconstruction　　
Here is the results of reconstruction using encoder and generator.  
<img src="https://github.com/yunishi3/FC-alphaGAN/blob/master/Images/Reconstruction_Result.png">  

Almost voxels are reconstructed although small objects are disappeared. Numerical evaluations using IoU and mAP are discribed below.  

#### -Generation from normal distribution
Here is the results of generation from normal distribution using generator.  
<img src="https://github.com/yunishi3/FC-alphaGAN/blob/master/Images/Generation_Result.png">  

As above figure, alphaGAN architecture worked better than standard fully convolutional VAE architecture. But this was not enough to represent realistic scene objects. This is assumed by that encoder was not generalized to the distribution and the probabilistic space was extremely complicated because of the sparsity of dataset. In order to solve this problem, the probabilistic space should be isolated to each objects and layout.


### Reconstruction Performance
Here is the numerical evaluation of reconstruction performance.
#### -Intersection over Union(IoU)
IoU is difined as [5]. The bar chart describes IoU performances of each classes. The line chart describes over all IoU performances.  
<img src="https://github.com/yunishi3/FC-alphaGAN/blob/master/Images/IoU_class.png">  
<img src="https://github.com/yunishi3/FC-alphaGAN/blob/master/Images/IoU_all.png" width="300">  

#### -mean Average Precision(mAP)
The bar chart and line chart describes same as IoU.  
<img src="https://github.com/yunishi3/FC-alphaGAN/blob/master/Images/mAP_class.png">  
<img src="https://github.com/yunishi3/FC-alphaGAN/blob/master/Images/mAP_all.png" width="300">  

This results gives following considerations.
* Fully convolution enables reconstruction performance to get better even though the number of dimensions are same.
* AlphaGAN architecture enables reconstruction performance to get better.

## Evaluations
### Interpolation
Here is the interpolation results. Gif image of interpolation is posted on the top of this document.  
<img src="https://github.com/yunishi3/FC-alphaGAN/blob/master/Images/Interpolation.png">  
The latent space walkthrough gives smooth transitions between scenes.

### Interpretation of latent space
This below charts are the 2D represented mapping by SVD of 200 encoded samples. Gray scale gradations are followed to 1D embedding by SVD of centroid coordinates of scenes. Left is fully convolution, right is standard 1D latent vector of 1200 dimension.  
<img src="https://github.com/yunishi3/FC-alphaGAN/blob/master/Images/latent_space_visualization.png">  
This means fully convolution enables the latent space to be related to spatial contexts compare to standard VAE.  
 
The below figures describe the effects of individual spatial dimensions composed of 5x3x5 as the latent space. The normal distribution noises were given on the indivisual dimension, and the level of change from original scene was represented by red colors.  
<img src="https://github.com/yunishi3/FC-alphaGAN/blob/master/Images/noise_visualization.png">  
This means each spatial dimensions is related to generation of each spaces.




## Installation
 This package requires python2.7. If you don't have fllowing prerequisists, you need to download them using pip, apt-get etc before downloading the repository.  
 Also at least 12GB GPU memory is required.  
### Prerequisists
Following is my environment.
* Base  
```
 - tensorflow 0.10.0  
 - numpy 1.11.2      
 - easydict 1.6  
```
 
* -Evaluation  
```
 - sklearn.metrics 0.15.2  
```

* -Visualization  
```
 - vtk 5.8.0   
```

### Download
* Download the repository and go to the directory.  
`$ git clone https://github.com/yunishi3/FC-alphaGAN.git`  
`$ cd FC-alphaGAN`    

* Unzip the dataset (It would be 57GB)  
`$ tar xfvz Scenevox.tar.gz`
 

### Training
`$ python main.py --mode train`  

### Evaluation
If you want to use pretrained model, you can use the checkpoint files as following instructions.  
* Unzip the Checkpoint.
It contains checkpoint100000* as confirmation epoch = 100000  
`$ tar xfvz Checkpt.tar.gz`  

Or if you want to evaluate your trained model, you can replace confirmation epoch 100000 with another confirmation epoch which you want to confirm.

#### -Evaluate reconstruction performances from test data and make visualization files.
`$ python main.py --mode evaluate_recons --conf_epoch 100000`  

After execute, you could get following files in the eval directory.  
 - real.npy : Reference models chosen as test data.  
 - reons.npy : Reconstructed models encoded and decoded from reference models.  
 - generate.npy : Generated models from normal distribution.  
 - AP.csv : mean average precision result of this reconstruction models.  
 - IoU.csv : Intersection over Union result of this reconstruction models.  

#### -Evaluate the interpolation
`$ python main.py --mode evaluate_interpolate --conf_epoch 100000`  

After execute, you could get interpolation files like above interpolation results. Files would be so many.

#### -Evaluate the effect of individual spatial dimensions
`$ python main.py --mode evaluate_noise --conf_epoch 100000`

After execute, you could get noise files like above interpretation of latent space results. Files would be so many.


### Visualization
In order to visualize thier npy files generated by evaluation process, you need to use python vtk. Please see the [6] to know the details of this code. I just modified original code to fit the 3D multi object scenes.  
* Go to the eval directory  
`$ cd eval`  

* Visualize all and save the png files.  
`$ python screenshot.py ***.npy`  

* Visualize only 1 files.  
`$ python visualize.py ***.npy -i 1`  
You can change the visualize model using index -i


## References
[1]Jiajun Wu, Chengkai Zhang, Tianfan Xue, William T. Freeman, Joshua B. Tenenbaum; Learning a Probabilistic Latent Space of Object Shapes via 3D Generative-Adversarial Modeling; arXiv:1610.07584v1  
[2]Andrew Brock, Theodore Lim, J.M. Ritchie, Nick Weston; Generative and Discriminative Voxel Modeling with Convolutional Neural Networks; arXiv:1608.04236v2  
[3]Shuran Song, Fisher Yu, Andy Zeng, Angel X. Chang, Manolis Savva, Thomas Funkhouser; Semantic Scene Completion from a Single Depth Image; arXiv:1611.08974v1  
[4]Mihaela Rosca, Balaji Lakshminarayanan, David Warde-Farley, Shakir Mohamed; Variational Approaches for Auto-Encoding Generative Adversarial Networks; arXiv:1706.04987v1  
[5]Christopher B. Choy, Danfei Xu, JunYoung Gwak, Kevin Chen, Silvio Savarese; 3D-R2N2: A Unified Approach for Single and
Multi-view 3D Object Reconstruction; arXiv:1604.00449v1  
[6]https://github.com/zck119/3dgan-release/tree/master/visualization/python  


