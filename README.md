# RGB-D Salient Object Detection Using Conditional GAN
This code is a Pytorch implementation of [RGB-D salient object detection using cGAN](http://dcollection.sogang.ac.kr:8089/dcollection/public_resource/pdf/000000063091_20191107151546.pdf).
Two-stream network generates the pixel-wise saliency map and PatchGAN discriminator learns to determine whether the generated saliency map is real or fake.
<center><img src="./imgs/figure_1.png"></center>

## Requirements
* Python 3
* Pytorch >= 1.1
* Numpy, PIL

##  Experimental enviroment
* Ubuntu 18.04
* Nvidia Geforce GTX 1080Ti
* [Pytorch 1.3 docker image](https://hub.docker.com/r/pytorch/pytorch/)

## Getting started
1. Clone this repository
   ```
    git clone https://github.com/wonjjo/RGB-D_Salient_Object_Detection.git
   ```
   
2. Prepare datasets
    We use [NLPR](https://sites.google.com/site/rgbdsaliency/dataset) and [NJUDS2000](https://svalianju.wixsite.com/home/salient-object-detection) RGB-D saliency detection datasets to train the networks. (additionally [DUT-OMRON](http://saliencydetection.net/dut-omron/), [HKU-IS](https://sites.google.com/site/ligb86/hkuis), and [MSRA10K](https://mmcheng.net/msra10k/) RGB saliency datasets are used with synthetic depth maps that was generated using [pix2pix](https://arxiv.org/abs/1611.07004).
    
3. Training
    ```
    cd RGB-D_Salient_Object_Detection
    python main.py \
        --mode train \
        --input_dir path/to/trainset \
        --output_dir path/to/logs \
        --max_epochs 100 \
        --[args]
   See below for more args. 
    ```
    
4. Testing
    ```
    python main.py \
        --mode test \
        --input_dir path/to/testset \
        --output_dir path/to/output_images \
        --checkpoint path/to/saved_logs \
        --n_epochs 100
    ```
    
5. More details of args.
    There are several options on running <text>main.py</text> with --[args].
    ```
    --mode ["train", "test] : train or test mode selection
    --input_dir [path/to/imgs] : Folder path which containing input images
    --output_dir [path/to/output] : Folder path to save logs in training or output images in testing
    --tensorboard_dir [path/to/logs] : Folder path to save tensorboard logs
    --checkpoint  [path/to/logs] : Folder path to resume training or use for testing
    --n_epochs [100] : Load checkpoint from trained models with "n_epochs"
    --max_epochs [100] : Number of epochs in training step
    --batch_size [16] : Size of mini-batch
    --cuda : Using GPU
    --threds : Number of threds for data loading
    --ngf [64] : Number of filters on first convolution layer of the generator
    --ndf [16] : Number of filters on first convolution layer of the discriminator
    --lr [0.0002] : Learning rate
    --beta1 [0.9] : Mementum of Adam optimizer
    --ce_weight [10.0] : Weight on CrossEntropyLoss term of loss function
    --gan_weight[1.0] : Weight on GANLoss term of loss function
    ```
    
6. Pretrained model
    If you want to testing with pretrained model, download [this](https://drive.google.com/file/d/1Rzn16s-E69E2BvYFH6DuTXxj3LnkNvJW/view?usp=sharing) and put it path/to/logs. The model was trained by using datasets as described in step 2. You can simply test the model with the following command.
    ```
    cd RGB-D_Salient_Object_Detection
    python main.py \
        --mode test \
        --input_dir path/to/testset \
        --output_dir path/to/output_images \
        --checkpoint path/to/pretrained_model \
        --pretrained
    ```

7. Running the code using Pytorch < 1.1
    From Pytorch 1.1, TensorBoard has been officially supported. If you used Pytorch < 1.1, follow the below steps.
    First, install the TensorboardX.
    ```
    pip install TensorboardX
    ```
    Second, modify line 18 in <text>utils.py</text>.
    ```
    del line 18: from torch.utils.tensorboard import SummaryWriter
    new line 18: from tensorboardX import SummaryWriter
    ```
    
## Architecture
<center><img src="./imgs/figure_2.png"></center>

## Results
* NLPR testset
<center><img src="./imgs/figure_3.png"></center>

* NJUDS2000 testset
<center><img src="./imgs/figure_4.png"></center>

* F-measure scores
Compared to not using depth maps completely and not using only synthetic depth maps in training step.

    | Dataset | Only RGB | RGB + real depth map | RGB + real and synthetic depth map | 
    | :----------: | :---------: | :----------: | :----------: |
     | NLPR | 0.7705 | 0.7780 | **0.8103** |
     | NJUDS2000 | 0.8014 | 0.8405 | **0.8567** |  
 
## More details
Please see [this](http://dcollection.sogang.ac.kr:8089/dcollection/public_resource/pdf/000000063091_20191107151546.pdf).
