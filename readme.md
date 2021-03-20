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


2. Then the grid cells are filled at random based on the fill probability. Additional, for now sqaure images, the targets are translated within their grid cell.

    Example (input_size=512, target_size=32, fill_prob=0.4):
    ![gen_segment_example](/images/readme/gen_segment_example.png)

A few examples from the segmentation generator:

![visualize_segment](/images/readme/visualize_segment.jpeg)


## Classification Model 1 - Multitask Resnet

Reasoning

Image of the model architecture

Explanation of the loss calculation

Weighted loss from here:<br>
Kendall, Alex, Yarin Gal, and Roberto Cipolla. "Multi-task learning using uncertainty to weigh losses for scene geometry and semantics." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2018.
[[arXiv](https://arxiv.org/abs/1705.07115)].


Full training details:
```
# Model Config

# Training Hyperparameters

# Optimization

# Dataset parameters


```

Example of the metrics from a single training run:

![multitask_resnet_run](/images/readme/multitask_resnet_run.png)


## Results
