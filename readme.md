# Setup


Conda environment
```
conda create --name pytorch
conda activate pytorch
```
Conda install command generated here: [https://pytorch.org/](https://pytorch.org/)
```
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
```

```
# install jupyter support
conda install ipykernel jupyter

# make a pytorch kernel
python -m ipykernel install --user --name pytorch --display-name "Pytorch"

# requirements
pip install -r requirements.txt

```

Start jupyter notebook from the project directory with this command
```
jupyter notebook
```

# Purpose

The goal is to implement this model into an aerial imaging pipeline. A segmentation model will detect targets and a classification model will assign classification labels to the objects.

## The Data

[generate_targets.py](https://github.com/Lukeasargen/targetClassifier/blob/main/generate_targets.py) has a generator class which can produce both classification and segmentation data.

### Synthetic Classification Data
Targets before preprocessing are very noisey and have severe aliasing effects. To combat this, the targets are generated at higher resolution defined by the expansion_factor variable (3 or 4 times expansion does not increase cpu load dramatically). Then, in the processing step before the images are batched, the target images are resized using the torchvision implementation transforms.Resize(). Since this will been necessary for real images, it is assumed to be valid for training. Below are samples of the classification targets.

The classification labels are the angle of the **letter**, **shape**, **shape color**, **letter**, and **letter color**.

High resolution targets:

![high_res_targets](/images/readme/high_res_targets.jpeg)

Targets after resizing:

![classify_processed](/images/readme/classify_processed.png)

Example of the labels:

![classify_labels](/images/readme/classify_labels.png)


### Synthetic Segmentation Data
Segmentation images have the same aliasing issues as the classification. Segmentation images are generated at a higher resolution defined by the expansion_factor variable and downsampling smooths the images. The general process for creating these images follows these steps:

1. The image is divided into grid cells based on the smallest target specified by the target_size argument. Then the grid size is uniformly sampled between the largest and smallest size. The grid determines the position of the targets and the scale of the target is randomly sampled. Below shows an example of the smallest grid and the largest grid filled with 100% probability. The images are 256x256 and the target_size is 32.

    Smallest grid:

    ![grid_smallest](/images/readme/grid_smallest.png)


    Largest grid:

    ![grid_largest](/images/readme/grid_largest.png)


2. Then the grid cells are filled at random based on the fill probability. Additionally, for non square images, the targets are translated within their grid cell so the targets appear at different locations in every sample.

    Example (input_size=512, target_size=32, fill_prob=0.4):
    ![gen_segment_example](/images/readme/gen_segment_example.png)

A few more examples from the segmentation generator:

![visualize_segment](/images/readme/visualize_segment.jpeg)

# Classification

The goal is to make a model capable of extracting multiple target characteristics. Further, the real time operation of this model calls for more efficient image analysis. Multitask learning is used. The basic methodolgy is to have a single backbone network that outputs a latent representation and multiple classification heads which are learned on the shared representation.

## Classification Model 1 - Multitask Resnet

Multitask Resnet has backbone made from a Resnet like feature extractor and uses linear layers to classify each task. 

This model was not trained using sound methodology. The whole network was initialized randomly and all weights were trained at the same time. This makes several assumptions which decrease the performance of the multitask model compared to a single task model.

_Invalid Assumption - The labels can be learned jointly._ This is not valid because the labels are independent. For instance, the observation that the shape is red is irrelevant to the orientation.

_Invalid Assumption - The labels have the same difficultly._ The joint loss used was simply the sum of all task losses. While training, it was clear that the tasks varied in difficulty and using an equal weight resulted in certain tasks dominating the overall loss. In other words, the loss scale for each task was different. To inhibit the effect of different loss scales, the joint loss used was an uncertainty weighted loss for each task. The uncertainty is a learned parameter (sigma) and comes from this paper:<br>
Kendall, Alex, Yarin Gal, and Roberto Cipolla. "Multi-task learning using uncertainty to weigh losses for scene geometry and semantics." [[arXiv](https://arxiv.org/abs/1705.07115)]

The model was trained on 32x32 images normalized by the dataset mean and standard deviation. The ResNet backbone had filters [32, 32, 64, 128] and has 2 residual blocks at each level. The blocks have 2 squences of Convolution-Batchnorm-LeakyRelu and a residual connection. The The task heads are linear layers. The optimizer is SGD with 0.9 momentum, nestrov acceleration, weight decay of 5e-4, and 256 mini batch size. The learning rate is starts at 1e-2 and ramps up linearly for 23 epochs to the base learning rate 6e-2. The learning is divided by 10 at 62 and 68 epochs and training ends at 72 epochs. The tasks use a cross entropy loss.

Example of the metrics from a single training run:

![multitask_resnet_run](/images/readme/multitask_resnet_run.png)

The sigma value can be thought of as the variance of the output predictions. The learning rate of sigma is 400 times less than the rest of the model. This was decided by experimentation. Increasing or decreasing from 400 negatively effects some of the tasks.

Here is an example of training with the sigma learning rate too high:

![sigma_too_high](/images/readme/sigma_too_high.png)

Here is an example of training with the sigma learning rate too low:

![sigma_too_low](/images/readme/sigma_too_low.png)

Even though the model converges for all reasonable values of the sigma learning rate, 400 times less gives good results summarized in the table below. The values are percent correct on a validation set.
| Sigma Learning Rate | Orientation | Shape | Letter | Shape Color | Letter Color |
|-|-|-|-|-|-|
| lr/1 | 87.93 | 98.22 | 95.50 | 99.94 | 97.63 |
| lr/10 | 87.91 | 98.61 | 95.04 | 99.91 | 97.25 |
| lr/20 | 88.32 | 98.75 | 94.94 | 99.95 | 97.64 |
| lr/100 | 87.77 | 98.77 | 95.1 | 99.91 | 97.48 |
| lr/400 | 88.45 | 98.72 | 95.67 | 99.95 | 97.50 |
| lr/1000 | 88.17 | 98.61 | 95.31 | 99.96 | 97.45 |
| lr/5000 | 88.22 | 98.85 | 95.21 | 99.93 | 97.70 |
| lr/10000 | 83.67 | 96.92 | 90.47 | 99.96 | 95.71 |


## Classification Model 2 - Multitask Resnet (2 Stages)

The only differance from this and model 1 is the training. A small grid search will be used to determine which features can be improved when learned jointly. For example, learning orienation and letter is likely to boost training since the feature learned by the letter classifier can be used to classify the orientation of the letter. It is also assumed shape can be learned with letter and orientation since all tasks use patterns and lines. The results are shown in the table below:

_In progress..._

## Classification Model 3 - Ideas
- Squeezenet or efficient channel attention
- Swish or Mish activation
- Add spectral norm to convs
- Non-linear classifier heads
- Hyperbolic model for orientation
- Spaital pyramidal pooling
- Deep layer aggregation (deeplabv3)

## Classification - Summary

The best weights of each model were tested on a validation set and the results are in the table below:

| Accuracy | Orientation | Shape | Letter | Shape Color | Letter Color |
|-|-|-|-|-|-|
| Multitask Resnet | 89.46 | 99.07 | 96.95 | 99.95 | 98.30 |
| Multitask Resnet (2-stages) |  |  |  |  |  |


# Segmentation

The goal is to perform a pixel wise binary classification for the presence of a target.

The segementation models are trained on 256x256 images and batch size of 8. The Dice loss is used. The optimizer is SGD with 0.9 momentum, nestrov acceleration, and weight decay 5e-4. The learning rate starts at 1e-1 and is stepped by 0.2 using ReduceLROnPlateau with a patience of 50.

## Segemenation Model 1 - UNet

Ronneberger, Olaf, Philipp Fischer, and Thomas Brox. "U-Net: Convolutional Networks for Biomedical Image Segmentation." [[arXiv](https://arxiv.org/abs/1705.07115)].

Our implementation matches the UNet paper with the addition of batchnorm and different activation functions.

Example of the metrics from a training run:

![unet_run](/images/readme/unet_run.png)

## **Experiments with UNet**


### Weight Decay - 5e-4
**Weight decay** was found to improve performance. Weight decay of **5e-4** is used for all training. This experiment was done before the metrics were logged.

### Downsampling - Maxpool
For **downsampling**, **maxpool** gives best results. (Relu activation)
| Downsampling | Dice | Jaccard | Accuracy(>0.5) | Tversky(α=0.3,β=0.7) | Focal(α=0.5,γ=2) | BCE |
|-|-|-|-|-|-|-|
| Maxpool | 91.08 | 83.64 | 96.07 | 90.99 | 0.00003 | 0.036 |
| Avgpool | 90.53 | 82.71 | 96.08 | 90.21 | 0.00003 | 0.037 |
| Stride=2 | 89.25 | 80.63 | 95.75 | 89.37 | 0.00005 | 0.045 |

### Filter Count - 16 Filters is enough for experiments
Since the model needs to be fast, choosing the right **filter** count is important. **16 filters** was choosen. The same inference time for filters between 2 and 16 is assumed to be other opertations in the data pipline and time spent in the model is extremely low. (Relu activation)
| Filters | Inference (ms) | Dice | Jaccard | Accuracy(>0.5) | Tversky(α=0.3,β=0.7) | Focal(α=0.5,γ=2) | BCE |
|-|-|-|-|-|-|-|-|
| 2 | 2.75 | 92.39 | 85.93 | 95.98 | 92.18 | 0.00063 | 0.094 |
| 4 | 2.78 | 94.49 | 89.58 | 96.58 | 94.21 | 0.00033 | 0.083 |
| 8 | 2.78 | 96.62 | 93.48 | 96.78 | 96.10 | 0.00013 | 0.050 |
| 16 | 2.75 | 96.98 | 94.14 | 96.89 | 96.52 | 0.00005 | 0.045 |
| 32 | 3.81 | 97.30 | 94.74 | 96.47 | 96.95 | 0.00005 | 0.044 |
| 64 | 10.63 | 97.57 | 95.26 | 96.90 | 97.26 | 0.00002 | 0.034 |

### Activation - Relu
**Activation** effects accuracy and inference speed. Silu and Mish had the best scores. Leaky relu with a negative slope of 0.2 was tested; it is possible to tune the negative slope or use PRelu, but the results from Relu, Silu, and Mish are sufficient so this was not explored. **Relu is planned to be used in the final model.** Silu and Mish are still being considered if the final pipeline can run with bottlenecking at the model.
| Activation | Inference (ms) | Dice | Jaccard | Accuracy(>0.5) | Tversky(α=0.3,β=0.7) | Focal(α=0.5,γ=2) | BCE |
|-|-|-|-|-|-|-|-|
| Relu | 2.71 | 97.52 | 95.17 | 96.82 | 97.34 | 0.00111 | 0.135 |
| Leaky | 2.72 | 97.45 | 95.03 | 96.83 | 97.26 | 0.00112 | 0.135 |
| Silu | 3.02 | 97.59 | 95.29 | 96.90 | 97.44 | 0.00080 | 0.121 |
| Mish | 3.28 | 97.55 | 95.22 | 96.91 | 97.40 | 0.00081 | 0.120 |

### Batch Size - Large as possible

Each model sees the exact **same number of images**.

| Batch Size | Iterations | Dice | Jaccard | Accuracy(>0.5) | Tversky(α=0.3,β=0.7) | Focal(α=0.5,γ=2) | BCE |
|-|-|-|-|-|-|-|-|
| 64  | 800 | 96.40 | 93.07 | 96.41 | 96.21 | 0.00010 | 0.055 |
| 32  | 1600 | 94.66 | 89.86 | 96.27 | 94.25 | 0.00017 | 0.053 |
| 16  | 3200 | 95.71 | 91.78 | 96.52 | 95.28 | 0.00008 | 0.053 |
| 8  | 6400 | 96.22 | 92.72 | 96.72 | 95.81 | 0.00010 | 0.057 |
| 4  | 12800 | 95.99 | 92.31 | 96.70 | 95.61 | 0.00020 | 0.062 |

Each model has the **same number of update steps**.

| Batch Size | Iterations | Dice | Jaccard | Accuracy(>0.5) | Tversky(α=0.3,β=0.7) | Focal(α=0.5,γ=2) | BCE |
|-|-|-|-|-|-|-|-|
| 64  | 3200 | 96.36 | 92.98 | 96.68 | 96.01 | 0.00005 | 0.047 |
| 32  | 3200 | 96.03 | 92.36 | 96.56 | 95.67 | 0.00007 | 0.052 |
| 16  | 3200 | 95.71 | 91.78 | 96.52 | 95.28 | 0.00008 | 0.053 |
| 8  | 3200 | 95.12 | 90.71 | 96.45 | 94.66 | 0.00014 | 0.063 |
| 4  | 3200 | 93.27 | 87.46 | 96.03 | 92.56 | 0.00074 | 0.097 |

It is clear that more iterations and larger batch size improves results for these training hyperparameters. For sake of time, the experiments will continue to use a batch of 16. The final model will be trained with the largest batch size possible.

### Optimizer - Use Adam
Optimizer is extremely important. So far only SGD with momentum has been used. For this experiment, the intial learning rate is in the table. The learning rate is stepped by 0.2 at 150 and 210 epochs. Training ends at 240 epochs. SGD, and RMSprop all use a momentum of 0.9 and all optimizers have a weight decay of 5e-4 except Adam.
| Optimizer | Initial LR | Dice | Jaccard | Accuracy(>0.5) | Tversky(α=0.3,β=0.7) | Focal(α=0.5,γ=2) | BCE |
|-|-|-|-|-|-|-|-|
| SGD | 1e-1 | 96.27 | 92.82 | 96.74 | 95.94 | 0.00008 | 0.053 |
| RMSprop | 1e-3 | 92.14 | 85.55 | 95.84 | 91.34 | 0.00101 | 0.119 |
| Adam | 4e-3 | 96.51 | 93.26 | 96.68 | 96.23 | 0.00092 | 0.123 |
| AdamW | 4e-3 | 96.37 | 93.01 | 96.74 | 96.20 | 0.00094 | 0.121 |
| Adagrad | 1e-2 | 94.38 | 89.39 | 96.47 | 93.87 | 0.00019 | 0.067 |

### Loss Function - Jaccard Loss (IOU)
Even though you can backprop through all these metrics, it is clear that some perform poorly. BCE and Focal losses perform worst and will not be used. Dice and Jaccard extremely high performance with slight differences. Tversky loss has parameters which can tune the models bias for false positives or false negatives; this loss performs well, but it will not be used due to poor understanding of the parameters. **For further training, jaccard will be used.** The final model will be trained on both losses for a final comparison.
| Loss | Dice | Jaccard | Accuracy(>0.5) | Tversky(α=0.3,β=0.7) | Focal(α=0.5,γ=2) | BCE |
|-|-|-|-|-|-|-|
| Jaccard | 96.23 | 92.73 | 96.69 | 95.88 | 0.00110 | 0.133 |
| Dice | 96.14 | 92.57 | 96.65 | 95.80 | 0.00101 | 0.130 |
| Dice+Jaccard | 96.13 | 92.56 | 96.76 | 95.75 | 0.00103 | 0.130 |
| Tversky(α=0.3,β=0.7) | 95.68 | 91.73 | 96.46 | 96.11 | 0.00139 | 0.144 |
| Dice+BCE | 95.64 | 91.65 | 96.70 | 95.41 | 0.00001 | 0.028 |
| Jaccard+BCE | 95.48 | 91.35 | 96.41 | 95.18 | 0.00002 | 0.034 |
| Tversky(α=0.7,β=0.3) | 95.36 | 91.14 | 96.37 | 94.22 | 0.00149 | 0.150 |
| BCE | 93.46 | 87.73 | 96.51 | 93.47 | 0.00001 | 0.027 |
| Focal(α=0.5,γ=2) | 88.69 | 79.71 | 96.20 | 88.93 | 0.00005 | 0.045 |


### Gaussian Noise on inputs
| STD | Dice | Jaccard | Accuracy(>0.5) | Tversky(α=0.3,β=0.7) | Focal(α=0.5,γ=2) | BCE |
|-|-|-|-|-|-|-|
| 0.00 |  |  |  |  |  |  |


## Segemenation Model 2 - UNet++

Zhou, Zongwei,  Md Mahfuzur Rahman Siddiquee, Nima Tajbakhsh, and Jianming Liang. "UNet++: A Nested U-Net Architecture for Medical Image Segmentation." [[arXiv](https://arxiv.org/abs/1705.07115)].

To increase the strength of the segmentation model, the model is upgraded to UNet++. This model made several notable changes. UNet++ concatenates same size feature maps using dense blocks and residual connections. The motivation is to improve the gradient signal by reducing the path length between encoder features and decoder features. 

Our implementation matches the UNet++ paper as closely as possible. The same training is used as our implementation of UNet. Deep supervision is implemented as described in the paper. Deep supervision is done by using a 1x1 convolution on each dense block the same size as the input image to convert the block into a mask with the correct number of output channels. Then each mask is compared to the true mask and the loss is the sum of each masks loss. During inference, the output is the mean of all the masks.

The UNet++ implementation can be used with and without deep supervision with models called "unet_nested" and "unet_nested_deep", respectively.

Example of the metrics from a training run:

![unet_nested_run](/images/readme/unet_nested_run.png)

## **Experiments with UNet++**

UNet++ improves the UNet architecture in multiple ways: 1) **ensembles of UNets** at different depths 2) **aggregates feature maps** of the same size into dense blocks and 3) adds a **deep supervision** loss. Ensembling improves the model by having an architecture which supports multiple scales of information in it's decoder. Further, aggregating feature maps is known to give good resuls in many architectures, however it is unclear if combining features maps of the same size is an optimal way to merge features in the decoder. It is assumed that a dense layer will learn which input features have the best information and block the rest. Finally, deep supervision is not required in the architecture, however it provides more gradient signals for the smaller decoders in the ensemble.

### Deep Supervision
To test which model architecture would improve the task, several versions were trained and the results are in the table below. DS = Deep Supervision.
| Model | Filters | Inference (ms) | Dice | Jaccard | Accuracy(>0.5) | Tversky(α=0.3,β=0.7) | Focal(α=0.5,γ=2) | BCE |
|-|-|-|-|-|-|-|-|-|
| UNet (Relu) | 8 | 2.16 | 96.91 | 94.01 | 96.99 | 96.65 | 0.000445 | 0.098 |
| UNet (Relu) | 16 | 2.19 | 97.35 | 94.84 | 97.08 | 97.16 | 0.000785 | 0.120 |
| UNet (Relu) | 32 | 2.63 | 97.29 | 94.72 | 97.06 | 97.08 | 0.001258 | 0.140 |
| UNet++ (Relu) | 8 | 4.00 | 97.23 | 94.61 | 97.01 | 97.02 | 0.000921 | 0.122 |
| UNet++ (Relu) | 16 | 4.07 | 97.44 | 95.01 | 97.10 | 97.27 | 0.001920 | 0.164 |
| UNet++ (Relu) | 32 | 6.51 | 97.45 | 95.03 | 97.15 | 97.27 | 0.001871 | 0.161 |
| UNet++ (Relu, DS) | 8 | 4.17 | 97.12 | 94.41 | 96.96 | 96.91 | 0.000533 | 0.105 |
| UNet++ (Relu, DS) | 16 | 4.21 | 97.55 | 95.21 | 97.27 | 97.39 | 0.000832 | 0.122 |
| UNet++ (Relu, DS) | 32 | 6.71 | 97.44 | 95.01 | 97.23 | 97.28 | 0.000631 | 0.111 |

## Segmentation Summary

Here are the metrics from models with the highest dice coefficient from several architectures.

| Model | Dice | Jaccard | Accuracy(>0.5) | Tversky(α=0.3,β=0.7) | Focal(α=0.5,γ=2) | BCE |
|-|-|-|-|-|-|-|
| UNet (Relu) | 97.44 | 95.02 | 96.64 | 97.16 | 0.000456 | 0.043 |
| UNet++ (Relu) | 97.52 | 95.17 | 96.77 | 97.25 | 0.000029 | 0.038 |
