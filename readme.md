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


Weighted loss from here:<br>
Kendall, Alex, Yarin Gal, and Roberto Cipolla. "Multi-task learning using uncertainty to weigh losses for scene geometry and semantics." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2018.
[[arXiv](https://arxiv.org/abs/1705.07115)].
