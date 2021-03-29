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

The model was trained on 32x32 images normalized by the dataset mean and standard deviation. The ResNet backbone had filters [32, 32, 64, 128] and has 2 residual blocks at each level. The blocks have 2 squences of Convolution-Batchnorm-LeakyRelu and a residual connection. The The task heads are linear layers. The optimizer is SGD with 0.9 momentum, nestrov acceleration, weight decay of 5e-4, and 256 mini batch size. The learning rate is starts at 1e-2 and ramps up linearly for 23 epochs to the base learning rate 6e-2. The learning is divided by 10 at 62 and 68 epochs and training ends at 72 epochs. The uncertainty of the loss scales has a learning rate 1000 times less than the model weights. The tasks use a cross entropy loss.

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

In progress

## Classification Model 3 - Ideas
- Squeezenet or efficient channel attention
- Leaky, Swish, or Mish activation
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

## Comments on  validation

Generating random targets during training removes the need for a validation set. This is  verified this by creating a set of 300k target images and labels. This was split into 291k training and 9k validation samples. Here are the results from a training run:

![training_split_run](/images/readme/training_split_run.png)

The validation is the dashed line and training is solid. These results are easily reproducible for various changes in hyperparameters. Given how close the validation line flows the training line, it's not worth the compute to do validation while training.

A validation set created from real images of targets is in progress.

# Segmentation

The goal is to perform a pixel wise binary classification for the presence of a target.

The segementation models are trained on 256x256 images and batch size of 8. The Dice loss is used. The optimizer is SGD with 0.9 momentum, nestrov acceleration, and weight decay 5e-4. The learning rate starts at 1e-1 and is stepped by 0.2 using ReduceLROnPlateau with a patience of 50.

## Segemenation Model 1 - UNet

Ronneberger, Olaf, Philipp Fischer, and Thomas Brox. "U-Net: Convolutional Networks for Biomedical Image Segmentation." [[arXiv](https://arxiv.org/abs/1705.07115)].

Our implementation matches the UNet paper with the addition of batchnorm and different activation functions.

![unet_graph](/images/readme/unet_graph.png)

Example of the metrics from a training run:

![unet_run](/images/readme/unet_run.png)

## **Additional experiments with UNet**


### Weight Decay
**Weight decay** was found to improve performance. Weight decay of **5e-4** is used for all training. This experiment was done before the metrics were logged.

### Downsampling
For **downsampling**, **maxpool** gives best results. (Relu activation)
| Downsampling | Dice | Jaccard | Accuracy(>0.5) | Tversky(α=0.3,β=0.7) | Focal(α=0.5,γ=2) | BCE |
|-|-|-|-|-|-|-|
| Maxpool | 91.08 | 83.64 | 96.07 | 90.99 | 0.00003 | 0.036 |
| Avgpool | 90.53 | 82.71 | 96.08 | 90.21 | 0.00003 | 0.037 |
| Stride=2 | 89.25 | 80.63 | 95.75 | 89.37 | 0.00005 | 0.045 |

### Filter Count
Since the model needs to be fast, choosing the right **filter** count is important. **16 filters** was choosen.The same inference time for filters between 2 and 16 is assumed to be other opertations in the data pipline and time spent in the model is extremely low. (Relu activation)
| Filters | Inference | Dice | Jaccard | Accuracy(>0.5) | Tversky(α=0.3,β=0.7) | Focal(α=0.5,γ=2) | BCE |
|-|-|-|-|-|-|-|-|
| 2 | 2.58 ms | 92.39 | 85.93 | 95.98 | 92.18 | 0.00063 | 0.094 |
| 4 | 2.58 ms | 94.49 | 89.58 | 96.58 | 94.21 | 0.00033 | 0.083 |
| 8 | 2.58 ms | 96.62 | 93.48 | 96.78 | 96.10 | 0.00013 | 0.050 |
| 16 | 2.58 ms | 96.98 | 94.14 | 96.89 | 96.52 | 0.00005 | 0.045 |
| 32 | 3.75 ms | 97.30 | 94.74 | 96.47 | 96.95 | 0.00005 | 0.044 |
| 64 | 10.61 ms | 97.57 | 95.26 | 96.90 | 97.26 | 0.00002 | 0.034 |

### Activation
**Activation** effects accuracy and inference speed. Relu and Mish had the best scores. Silu has lowest overall inference time, but it's poor performance excludes it. Leaky relu with a negative slope of 0.2 was tested; it is possible to tune the negative slope or use PRelu, but the results from Relu and Mish are sufficient so this was not explored. **Relu is planned to be used in the final model.** Mish is still being considered if the final pipeline can run with Mish and not bottleneck.
| Activation | Inference | Dice | Jaccard | Accuracy(>0.5) | Tversky(α=0.3,β=0.7) | Focal(α=0.5,γ=2) | BCE |
|-|-|-|-|-|-|-|-|
| Relu | 2.53 ms | 97.38 | 94.89 | 96.86 | 97.05 | 0.00004 | 0.042 |
| Leaky | 2.57 ms | 95.67 | 92.28 | 96.32 | 95.57 | 0.00012 | 0.057 |
| Silu | 2.29 ms | 92.97 | 86.91 | 96.19 | 92.18 | 0.00076 | 0.070 |
| Mish | 3.01 ms | 97.15 | 94.46 | 96.83 | 96.73 | 0.00004 | 0.042 |

### Input Size
Since the data is generated at train time, the **input size** is a cpu bottleneck. Previous experiments used 256, but **192** gives comparable results and training time is reasonable. It is clear that the model improves with input size. A final production model will be trained with the largest image size that can fit on the GPU.
| Input Size | Model Speed | Training Duration | Dice | Jaccard | Accuracy(>0.5) | Tversky(α=0.3,β=0.7) | Focal(α=0.5,γ=2) | BCE |
|-|-|-|-|-|-|-|-|-|
| 512 | 12.99 ms | 40.14 minutes | 96.38 | 93.03 | 96.61 | 96.04 | 0.00013 | 0.056 |
| 384 | 8.38 ms | 23.30 minutes | 96.09 | 92.48 | 96.44 | 95.77 | 0.00009 | 0.056 |
| 320 | 6.59 ms | 16.82 minutes | 96.31 | 92.90 | 96.36 | 96.00 | 0.00008 | 0.053 |
| 256 | 5.24 ms | 9.81 minutes | 96.10 | 92.50 | 96.67 | 95.70 | 0.00011 | 0.057 |
| 192 | 5.23 ms | 5.90 minutes | 95.90 | 92.13 | 96.71 | 95.54 | 0.00011 | 0.058 |
| 128 | 5.23 ms | 3.12 minutes | 95.52 | 91.44 | 96.56 | 95.13 | 0.00013 | 0.062 |

### Batch Size

### Fill Probability

### Optimizer

### Learning Rate

## Segemenation Model 2 - UNet++

Zhou, Zongwei,  Md Mahfuzur Rahman Siddiquee, Nima Tajbakhsh, and Jianming Liang. "UNet++: A Nested U-Net Architecture for Medical Image Segmentation." [[arXiv](https://arxiv.org/abs/1705.07115)].

To increase the strength of the segmentation model, the model is upgraded to UNet++. This model made several notable changes. UNet++ concatenates same size feature maps using dense blocks and residual connections. The motivation is to improve the gradient signal by reducing the path length between encoder features and decoder features. 

Our implementation matches the UNet++ paper as closely as possible. The same training is used as our implementation of UNet. Deep supervision is implemented as described in the paper. Deep supervision is done by using a 1x1 convolution on each dense block the same size as the input image to convert the block into a mask with the correct number of output channels. Then each mask is compared to the true mask and the loss is the sum of each masks loss. During inference, the output is the mean of all the masks.

The UNet++ implementation can be used with and without deep supervision with models called "unet_nested" and "unet_nested_deep", respectively.

![unet_nested_deep_graph](/images/readme/unet_nested_deep_graph.png)

Example of the metrics from a training run:

![unet_nested_run](/images/readme/unet_nested_run.png)

## **Additional experiments with UNet++**

UNet++ improves the UNet architecture in multiple ways: 1) **ensembles of UNets** at different depths 2) **aggregates feature maps** of the same size into dense blocks and 3) adds a **deep supervision** loss. Ensembling improves the model by having an architecture which supports multiple scales of information in it's decoder. Further, aggregating feature maps is known to give good resuls in many architectures, however it is unclear if combining features maps of the same size is an optimal way to merge features in the decoder. It is assumed that a dense layer will learn which input features have the best information and block the rest. Finally, deep supervision is not required in the architecture, however it provides more gradient signals for the smaller decoders in the ensemble.

### Deep Supervision
To test which model architecture would improve the task, several versions were trained and the results are in the table below. DS = Deep Supervision.
| Model | Inference | Dice | Jaccard | Accuracy(>0.5) | Tversky(α=0.3,β=0.7) | Focal(α=0.5,γ=2) | BCE |
|-|-|-|-|-|-|-|-|
| UNet (Relu) | 2.53 ms | 97.30 | 94.75 | 97.00 | 97.09 | 0.000033 | 0.039 |
| UNet (Mish) | 3.01 ms | 97.11 | 94.38 | 96.75 | 96.91 | 0.000045 | 0.044 |
| UNet++ (Relu) | 6.08 ms | 97.14 | 94.44 | 96.69 | 96.94 | 0.000047 | 0.044 |
| UNet++ (Mish) | 6.79 ms | 97.10 | 94.37 | 96.63 | 96.90 | 0.000058 | 0.047 |
| UNet++ (Relu, DS) | 6.27 ms | 97.09 | 94.34 | 96.85 | 96.79 | 0.000049 | 0.044 |
| UNet++ (Mish, DS) | 7.09 ms | 96.96 | 94.11 | 96.75 | 96.64 | 0.000069 | 0.048 |

## Segmentation Summary

Here are the metrics from models with the highest dice coefficient from several architectures.

| Model | Dice | Jaccard | Accuracy(>0.5) | Tversky(α=0.3,β=0.7) | Focal(α=0.5,γ=2) | BCE |
|-|-|-|-|-|-|-|
| UNet (Relu) | 97.44 | 95.02 | 96.64 | 97.16 | 0.000456 | 0.043 |
| UNet++ (Relu) | 97.52 | 95.17 | 96.77 | 97.25 | 0.000029 | 0.038 |
