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

The sigma value can be thought of as the variance of the output predictions. The learning rate of sigma is 1000 times less than the rest of the model. This was decided by experimentation. Increasing or decreasing from 1000 negatively effects some of the tasks.

Here is an example of training with the sigma learning rate too high:

![sigma_too_high](/images/readme/sigma_too_high.png)

Here is an example of training with the sigma learning rate too low:

![sigma_too_low](/images/readme/sigma_too_low.png)

Even though the model converges for all reasonable values of the sigma learning rate, 1000 times less gives good results summarized in the table below.

### Classification Accuracy
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

The only differance from this and model 1 is the training. A samll grid search will be used to determine which features can be improved when learned jointly. For example, learning orienation and letter is likely to boost training since the feature learned by the letter classifier can be used to classify the orientation of the letter. It is also assumed shape can be learned with letter and orientation since all tasks use patterns and lines. The results are shown in the table below:

In progress

## Classification Model 3 - Ideas
- Squeezenet or efficient channel attention
- Leaky, Swish, of Mish activation
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

Our implementation matches the UNet paper with the addition of batchnorm and different activation functions.

Ronneberger, Olaf, Philipp Fischer, and Thomas Brox. "U-Net: Convolutional Networks for Biomedical Image Segmentation." [[arXiv](https://arxiv.org/abs/1705.07115)].

![unet_graph](/images/readme/unet_graph.png)

Example of the metrics from a training run:

![unet_run](/images/readme/unet_run.png)

## Segemenation Model 2 - UNet++

To increase the strength of the segmentation model, the model is upgraded to UNet++. This model made several notable changes. UNet++ concatenates same size feature maps using dense blocks and residual connections. The motivation is to improve the gradient signal by reducing the path length between encoder features and decoder features. 

Our implementation matches the UNet++ paper as closely as possible. The same training is used as our implementation of UNet. Deep supervision is implemented as described in the paper. During training, 1x1 convolution is applied to each dense block the same size as the input image to convert the block into a map with the correct number of output channels. Then each mask is compared to the true mask and the loss is the sum of each masks loss. During inference, the output is the mean of all the masks.

Zhou, Zongwei,  Md Mahfuzur Rahman Siddiquee, Nima Tajbakhsh, and Jianming Liang. "UNet++: A Nested U-Net Architecture for Medical Image Segmentation." [[arXiv](https://arxiv.org/abs/1705.07115)].

![unet_nested_graph](/images/readme/unet_nested_graph.png)

Example of the metrics from a training run:

![unet_nested_run](/images/readme/unet_nested_run.png)


## Segmentation Summary

| Metric | Dice | Jaccard | Accuracy(>0.5) | Tversky(α=0.3,β=0.7) | Focal(α=0.5,γ=2) | BCE |
|-|-|-|-|-|-|-|
| UNet (Relu) | 97.44 | 95.02 | 96.64 | 97.16 | 0.0004565 | 0.04290 |
| UNet++ (Relu) | 97.52 | 95.17 | 96.77 | 97.25 | 0.00002932 | 0.03840 |
