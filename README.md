

# A Py-thonic Implementation of Image Captioning, C[aA]RNet!   
[![forthebadge made-with-python](http://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)   ![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)  [![PyPI license](https://img.shields.io/pypi/l/ansicolortags.svg)](https://pypi.python.org/pypi/ansicolortags/)  

**Convolutional(and|Attention)RecurrentNet!**
The goal of the project is to define a neural model for retrieve a caption given an image.

Given a dataset, a neural net composed by:
- Encoder (Pre-trained Residual Neural Net.)
- Decoder (A LSTM Model)
Will represent the image in a space defined by the encoder, this representation is fed to the
decoder (in different ways) that will learn to provide as output a caption,
contestualized by the image and by the words generated at each timestep by the LSTM.

Here you can find the table of content, before you start remember:

> You can close the project, the story ends. 
You ca read the project, you stay in wonderland, and I show you how deep the rabbit hole goes.


# Table of contents
- [A Py-thonic Implementation of Image Captioning, C[aA]RNet!](#a-py-thonic-implementation-of-image-captioning--c-aa-rnet-)
- [Table of contents](#table-of-contents)
- [Prerequisite Knowledge](#prerequisite-knowledge)
- [How to run the code](#how-to-run-the-code)
  * [Python supported versions](#python-supported-versions)
  * [Libraries Dependency](#libraries-dependency)
  * [Enviroment Variable](#enviroment-variable)
  * [CLI Explanation](#cli-explanation)
  * [GPUs Integration](#gpus-integration)
- [Data Pipeline](#data-pipeline)
  * [Dataset Format](#dataset-format)
    + [Images](#images)
    + [Results](#results)
- [What does the script produce](#what-does-the-script-produce)
  * [During training](#during-training)
  * [During evaluation](#during-evaluation)
- [Project structure](#project-structure)
  * [Filesystem](#Filesystem)
  * [Interfaces](#interfaces)
  * [Encoder](#encoder)
    + [CResNet50](#cresnet50)
    + [CResNet50Attention](#cresnet50attention)
  * [Decoder](#decoder)
    + [RNetvI](#rnetvi)
    + [RNetvH](#rnetvh)
    + [RNetvHC](#rnetvhc)
    + [RNetvHCAttention](#rnetvhcattention)
- [Training Procedure](#training-procedure)
  * [Loss type](#loss-type)
    + [Remark: Loss in the attention version](#remark--loss-in-the-attention-version)
- [Results](#results-1)
  * [Personal Experiments](#personal-experiments)
- [References](#references)
  * [Authors](#authors)

# Prerequisite Knowledge
For better understanding the code and the information inside, since this repository has the scope to be understandable for all the curious and not only for the people involved in this kind of topic, maybe is useful to take a look at this references:

-[Pytorch documentation](https://pytorch.org/docs/stable/index.html)
-[Convolutional Neural Net (Stanford Edu)](https://cs231n.github.io/convolutional-networks/)
-[Recurrent Neural Net (Stanford Edu)](https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-recurrent-neural-networks#architecture)
-[Residual Neural Net (D2L AI)](https://d2l.ai/chapter_convolutional-modern/resnet.html)


# How to run the code
[![Linux](https://svgshare.com/i/Zhy.svg)](https://svgshare.com/i/Zhy.svg) [![macOS](https://svgshare.com/i/ZjP.svg)](https://svgshare.com/i/ZjP.svg) [![Windows](https://svgshare.com/i/ZhY.svg)](https://svgshare.com/i/ZhY.svg)

The code can be run in every OS, feel free to use whatever you want. Of course a high-end machine is mandatory, since the huge amout of needed data can lead to out-of-memory error in low-end machine. 
**Remember that the dataset need to be downloaded before launching and it must respect the dataset format**
How to prepare the dataset for C[aA]RNet training?

 1. [Download](https://www.kaggle.com/hsankesara/flickr-image-dataset) the dataset.
 2. Extract it in the root of the repository.
 3. Rename the folder into *dataset*
 4. Rename the images folder into *images*

If you have a special situation, you can modify the [VARIABLE.py](#enviroment-variable) file and/or some optional parameters before launching the script ([CLI Explanation](#cli-explanation)
).

## Python supported versions
The code is ready to run for every version of python greater than 3.6.
As you will see also in the code, some facilities are not available in python versions lower than 3.9. All this tricky situations are marked into the code with a comment, so you can choose what you prefer by un/commenting them.

## Libraries Dependency
| Library | Version  |
| ------------ | ------------ |
|  Torch | 1.3.0+cu100  |
|  Torchvision | 0.4.1+cu100  |
|  Pillow | 8.4.0  |
|  Numpy | 1.19.5  |
|  Pandas | 1.1.5  |

Naturally inside the root of the package is present a requirements.txt file, you can install in your enviroment (or v.env.) all the required packages with the command below, executed in the shell with the enviroment activated:
```bash
pip install -r requirements.txt
```
## Enviroment Variable
Since some attributes of the repository are useful in more than one file, create an enviroment container is a way to accomplish this necessity.
Use a `.env` file is the most straightforward method, but since we want full compatibility among OS, a `VARIABLE.py` is a good compromise.
The CONSTANT defined are the following:
|COSTANT| MEANING |
|--|--|
| MAX_CAPTION_LENGTH | From the dataset are picked only samples whose caption has a length less or equal than this value |
|IMAGES_SUBDIRECTORY_NAME| The directory name which contains all the images (It must be under the root of the dataset folder) | 
| CAPTION_FILE_NAME | The file under the root of the dataset folder which contains all the captions.|
| EMBEDDINGS_REPRESENTATION | The way of creating the word embedding. UN-USED FOR NOW|

## CLI Explanation
The code can be run through a shell, here you can find how you can execute correctly the script, what are the custom parameters that you can feed before the execution and what are the meaning for each of them.
First of all the **always present part** is the invocation of the interpreter and the main file:
```bash
python main.py
```
After, the helper is prompted and you can see something like this:
```bash
usage: main.py [-h] [--attention ATTENTION]
               [--attention_dim ATTENTION_DIM]
               [--dataset_folder DATASET_FOLDER]
               [--image_path IMAGE_PATH]
               [--splits SPLITS [SPLITS ...]]
               [--batch_size BATCH_SIZE]
               [--epochs EPOCHS] [--lr LR]
               [--workers WORKERS]
               [--device DEVICE]
               {RNetvI,RNetvH,RNetvHC,RNetvHCAttention}
               {train,eval} encoder_dim hidden_dim
```

The mandatory part is composed by this parameters:
| Parameter  | Meaning  | Particular behavior  |
| ------------ | ------------ | ------------ |
| decoder| The decoder that you want use for the decoding part of the model, options: {RNetvI,RNetvH,RNetvHC,RNetvHCAttention} | Description of each type of decoder can be found in the next chapters  |
| mode  | The way of working of the net, training for training mode and eval for evaluation mode  | options: {train,eval}  |
| encoder_dim  | The dimesion of image projection of the encoder  |  Decoder=RNetvI => Don't care / Decoder=RNetvHCAttention => 2048  |
| hidden_dim  | The capacity of the LSTM  |   |

Optional parameters:
| Parameter | Meaning | Default |
|--|--|--|
| --attention | Use attention model. Default False | If enabled, decoder and encoder are forced to be CResNet50Attention and RNetvHCAttention |
| --attention_dim | Capacity of the attention unit. (Default 1024)||
| --dataset_folder | The folder containing all the samples. (Default "./dataset")| Used only in training mode|
| --image_path | The absolute path of the image that we want to retrieve the caption. (Default '') | Used only in evaluation mode |
| --splits | Fraction of data to be used in train set, val set and test set (Default: 60 30 10) | Used only in training mode|
| --batch_size | Mini batch size (Default: 32) | Used only in training mode |
| --epochs | Number of training epochs (Default: 500) | Used only in training mode |
| --lr | Learning rate (Adam) (default: 1e-3) | Used only in training mode |
| --workers | Number of working units used to load the data (Default: 4) | Used only in training mode |
| --device| Device to be used for computations \in {cpu, cuda:0, cuda:1, ...} (Default: cpu) | Used only in training mode |


## GPUs Integration

As you already seen in the cli explanation chapter, this code has support for GPUs (only NVIDIA atm.).
You need the CUDA Driver installed, if you want mantain the consistency among the torch version installed by default with requirements.txt and a cuda driver version, you can install the v440 version of NVIDIA driver + Cuda 10.2.

# Data Pipeline

For a better understanding of what happens inside the script is useful to visualize the data pipeline, considering:

- The Dataset: the container of all the examples (no separation is considered among training set, and others..)
- The Vocabulary: each example has a collection of words as a caption, it is useful have a vocabulary containing all of them.
- The C[aA]RNet: the neural network, no distinction for now among with/without Attention. 

Further explanation on the architecture of each single entity are in the following sections.
For now it is enough know that the script need these 3 entities for working with data.
![Data Pipeline](https://i.imgur.com/d8OtmUu.png)
Imagine to split each operation in a timestep.
- T_0: The dataset is loaded 
- T_1:
  - a) The dataset is casted into a Dataloader (pytorch class).
  - b) Given the dataset, an associated vocabulary is defined.
 - T_2: A dataloader is created
 - T_3: C[aA]RNet will use both dataloader and vocabulary for training operations, in evaluation mode instead, only the vocabulary is taken in consideration since the dataloader has size 1.
 
## Dataset Format

The way on how the Dataset is defined, follow the structure proposed by this dataset Flickr30k Image Dataset: https://www.kaggle.com/hsankesara/flickr-image-dataset

The filesystem structure is the following:

dataset/
├─ images/
│  ├─ pippo_pluto_paperino.jpg
├─ results.csv

 ### Images
Images folder contain the jpeg images, each of them must have a name without space.
`pippo_pluto_paperino.jpg`

### Results
File which contain the collection of captions.
**Since** a caption could contain the symbol comma *(,)* , the separator of each column will be a pipe *(|)*.
The first row of the file is the header, the column associated are defined in the table below:
| Parameter | Type     | Description                       |
| :-------- | :------- | :-------------------------------- |
| `image_name`      | `string` | File name of the associated caption |
| `comment_number`      | `int` | Index of the caption |
| `comment`*      | `string` | The caption |

*The caption, for a better tokenization, must have each word separated by a white_space character.
Moreover the simbol dot (".") define the end of the caption.

# What does the script produce
Since the project born also with the purpose of continue the developing, maybe for further features is also useful describe what the net produce as output, mainly the outputs can be divide in two large groups:

- What the net produce during the training.
- What the net produce for an evaluation.

## During training
During the training procedure the following outputs are produced:

 1. For each mini-batch of each epoch: store the loss and accuracy in a Dataframe.
 2. For each epoch, store the accuracy on validation set in a Dataframe.
 3. For each epoch, store a sample of caption generation on the last element of the last mini-batch in the validation set.
 4. Every time that the net reaches the best value in accuracy on validation data, the net is stored in non-volatile memory.

### 1
The Dataframe is stored as csv file *train_results.csv* at the end of each epoch, with the following structure:
 The first row of the file is the header, the column associated are defined in the table below:
| Parameter | Type     | Description                       |
| :-------- | :------- | :-------------------------------- |
|   Epoch   | `int` | The epoch id |
|   Batch   | `int` | The batch id |
|   Loss    | `float` | The loss evaluated for this batch|
|   Accuracy    | `float` | The accuracy evaluated for this batch |

### 2
The Dataframe is stored as csv file *validation_results.csv* at the end of each epoch, with the following structure:
 The first row of the file is the header, the column associated are defined in the table below:
| Parameter | Type     | Description                       |
| :-------- | :------- | :-------------------------------- |
|   Epoch   | `int` | The epoch id |
|   Accuracy    | `float` | The accuracy evaluated for the validation set in this epoch|

### 3
The features extracted from the last image of the last batch in the validation set are fed to the net in evaluation mode.
A caption.png file is generated. It includes the caption generated from C[aA]RNet and the source image.
If the attention is enabled, a file named attention.png is also produced and it includes for each word generated the associate attention in the source image.

### 4
Every time, during the training, a pick of accuracy in the evaluation of the validation set is reached, the net is stored in non-volatile memory.
The directory on which the net are stored is hidden and it is called *.saved* under the root of the repository.
This file are crucial for further training improvement and for evaluations after training.
The pattern of each file is the following:

 - Encoder: NetName_encoderdim_hiddendim_attentiondim_C,pth
 - Decoder: NetName_encoderdim_hiddendim_attentiondim_R,pth
 
Of course these parameters depend on what we provide at the training.

## During evaluation
The image is loaded, pre-processed, and fed to C[aA]RNet.
A caption.png file is generated. It includes the caption generated from C[aA]RNet and the source image.
If the attention is enabled, a file named attention.png is also produced and it includes for each word generated the associate attention in the source image.

# Project structure
The structure of the project take into account the possibility of expansion from the community or by a personal further implamentation.
This diagram is only general, and has the scope of grabbing what you could expect to see in the code, so the entities are empty and connected following their depencies.
Each method has a related docstring, so use it as reference.
![UML](https://i.imgur.com/xmGekz5.jpg)

## Filesystem
The Filesystem structure of the project has this form:

    C[aA]RNet/
    ├─ NeuralModels/
    │  ├─ Attention/
    │  │  ├─ IAttention.py
    │  │  ├─ SoftAttention.py
    │  ├─ Decoder/
    │  │  ├─ IDecoder.py
    │  │  ├─ RNetvH.py
    │  │  ├─ RNetvHC.py
    │  │  ├─ RNetvHCAttention.py
    │  │  ├─ RNetvI.py
    │  ├─ Encoder/
    │  │  ├─ IEncoder.py
    │  │  ├─ CResNet50.py
    │  │  ├─ CResNet50Attention.py
    │  ├─ CaARNet.py
    │  ├─ Dataset.py
    │  ├─ FactoryModels.py
    │  ├─ Metrics.py
    │  ├─ Vocabulary.py
    ├─ VARIABLE.py
    ├─ main.py
| File | Description |
|--|--|
| `VARIABLE.py` | Costant value used in the project|
| `main.py` | Entry point for execute the net|
| `IAttention.py` | The interface for implementing a new Attention model |
| `SoftAttention.py` | Soft Attention implementation |
| `IDecoder.py` | The interface for implementing a new decoder |
| `RNetvH.py` | Decoder implementation as LSTM H-version |
| `RNetvHC.py` | Decoder implementation as LSTM HC-version |
| `RNetvHCAttention.py` | Decoder implementation as LSTM HC-version with Attention mecchanism|
| `IEncoder.py` | The interface for implementing a new encoder |
| `CResNet50.py` | ResNet50 as encoder |
| `CResNet50Attention.py` | ResNet50 as encoder ready for attention mechanism |
| `CaRNet.py` | C[aA]RNet implementation |
| `Dataset.py` |  Manager for a dataset |
| `FactoryModels.py` | The Factory Design Pattern Implementation for every neural model proposed |
| `Metrics.py` | Produce report file |
| `Vocabulary.py` | Vocabulary manager entity |


## Interfaces
Interfaces are used for definig a contract among all of you that want to implement a new Encoder, Decoder or Attention model. 
Follow the interface is mandatory, in the docstring you can see also the suggested parameter for each method. 

## Encoder
The two encoder proposal are based on ResNet50 *(He et al. 2015, Deep Residual Learning for Image Recognition)*.
Depending on if we want attention or not, it removes one or more layer from the original net.

![ResNet50](https://www.researchgate.net/publication/336805103/figure/fig4/AS:817882309079050@1572009746601/ResNet-50-neural-network-architecture-56.ppm)
(ResNet-50 neural network architecture [56].) [Privacy-Constrained Biometric System for Non-Cooperative Users](https://www.researchgate.net/publication/336805103_Privacy-Constrained_Biometric_System_for_Non-Cooperative_Users)
### CResNet50
The 1st implementation remove the last layer from ResNet50, exposing the GlobalAveragePooling. Next to the pooling a linear layer of dimension *encoder_dim* is added, it will receive as input what the AveragePooling produce, in case of ResNet50 2048inputs.
 
### CResNet50Attention
The 2nd implementation remove the 2 last layers from ResNet50 (AveragePooling + FC), and expose the last convolutional layer that produce a tensor of shape: (Heigth/32, Width/32, 2048). 
Each portion has a 2048 vector representation. 
By default the total number of portions with a squared RGB images as input (3,224,224) is 49.

## Decoder
The decoder is based on the concept of Recurrent Neural Network, specifically in the declination of LSTM (Long-Short Term Memory) a type of RNN that exploit the way of updating the hidden state of the Network.
![LSTM](https://www.researchgate.net/profile/Xuan_Hien_Le2/publication/334268507/figure/fig8/AS:788364231987201@1564972088814/The-structure-of-the-Long-Short-Term-Memory-LSTM-neural-network-Reproduced-from-Yan.png)
*(The structure of the Long Short-Term Memory (LSTM) neural network. Reproduced from Yan [38].)* [Application of Long Short-Term Memory (LSTM) Neural Network for Flood Forecasting](https://www.researchgate.net/publication/334268507_Application_of_Long_Short-Term_Memory_LSTM_Neural_Network_for_Flood_Forecasting)
Each model starts from this idea and try different solution for feeding the initial context retrieved by the encoder:

 1. RNetvI: The image context is the input of the LSTM at time t_0.
 2. RNetvH: The image context is placed in the hidden state at time t_0.
 3. RNetvHC: The image context is placed in the hidden and cell state at time t_0.
 4. RNetvHCAttention: The image context is placed in the hidden and cell state, and at each time step t a vectorial representation of the focus of attention is concatenated with the input of the LSTM. 
 
### RNetvI
The 1st implementation use the image context as the first input of the lstm.

![RNetvI](https://i.imgur.com/PAxWnQy.png)
(Vinyals et al. 2014) [Show and Tell: A Neural Image Caption Generator](https://arxiv.org/abs/1411.4555) 

The only constraint is that the image context need to be projected into the word embeddings space.

### RNetvH
RNetvH initialize at time t_0 only the hidden state with the image context retrieved by the ResNet.

![RNetvH](https://i.imgur.com/9b2vVt3.jpg)
(Vinyals et al. 2014) [Show and Tell: A Neural Image Caption Generator](https://arxiv.org/abs/1411.4555) (Modified version by christiandimaio)
### RNetvHC
RNetvHC initialize at time t_0 both the hidden and cell state with the image context retrieved by the ResNet
![RNetvHC](https://i.imgur.com/pCrj3TS.jpg)
(Vinyals et al. 2014) [Show and Tell: A Neural Image Caption Generator](https://arxiv.org/abs/1411.4555) (Modified version by christiandimaio)

### RNetvHCAttention
This implementation combine a RNetvHC with Attention mecchanism.
![RNetvHCAttention](https://i.imgur.com/3GlMyJh.png)
Credit to christiandimaio et al. 2022

# Training Procedure
The training procedure involve the training set and the validation set.

 - The training set is splitted into mini-batch of defined size (parameter) and shuffled.
	 - For each mini-batch: 
		 - Provide the batch to the encoder that will produce a context vector for each element in the batch.
		 - Assuming that the tensor containing the captions (already translated into vector of id referred to word into the vocabulary) associated with the images batch are padded with zeros and ordered with a decreasing length.
		 - The context vectors and the captions are feeded into the Decoder.
		 - The output of the decoder will be the input of the method pack_padded_sequence, that will remove the pad region for each caption.
		 - The loss is evaluated and the backpropagation + weight update is done.
 - The accuracy is evaluated for the validation set.
	 - If we have a new best model, the net is stored in files.

## Loss type
The loss used is the CrossEntropyLoss, because in pytorch internally use a soft-max over each output t (remember the outputs of the lstm have the dimension of the vocabulary and we want the most likely word) and a NegativeLogLikelihood.
<p align="center">
  <img src="https://i.imgur.com/PBZbhjR.png" />
</p>
Where p_t:
<p align="center">
  <img src="https://i.imgur.com/iz2a86l.png" />
</p>

The loss used follow the paper (Vinyals et al. 2014) [Show and Tell: A Neural Image Caption Generator](https://arxiv.org/search/cs?searchtype=author&query=Vinyals%2C+O)
### Remark: Loss in the attention version
In the attention version we add a second term to the loss, the double stochastic regularization.
<p align="center">
  <img src="https://i.imgur.com/mNbrTo5.png" />
</p>
This can be interpreted as encouraging the model to pay equal attention to every part of the image over the course of generation.

  
# Results
to do

## Personal Experiments
to do

# References

 -  (Vinyals et al. 2014) [Show and Tell: A Neural Image Caption Generator](https://arxiv.org/abs/1411.4555) 
 - (Xu et al. 2015) [Show, Attend and Tell: Neural Image Caption Generation with Visual Attention](https://arxiv.org/pdf/1502.03044.pdf)
 - Material, Code and Beautiful lessons by Professor [Stefano Melacci](https://www3.diism.unisi.it/~melacci/)

## Authors

- [@christiandimaio](https://www.github.com/christiandimaio)

