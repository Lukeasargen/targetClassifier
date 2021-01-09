import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn  # Building the model
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchsummary import summary

from save_load import load_multitask_resnet
from dataset import LiveTargetDataset


def forward(model, input):
    preds = model(input)
    tasks_preds = np.zeros((len(preds)))
    for i in range(len(preds)):
        tasks_preds[i] = torch.max(preds[i], dim=1)[1]
    return tasks_preds


def evaluate(model, datalodaer):
    # Pass all the validate data and store the predictions
    model.eval()  # Set layers to eval
    torch.no_grad():  # Don't track gradients
        tasks_preds = np.zeros((val_size, 5))
        all_targets = np.zeros((val_size, 5))
        for idx, (data, target) in enumerate(datalodaer):
            all_targets[idx] = target.numpy()
            tasks_preds[idx][i] = forward(model, data)
    return tasks_preds, all_targets


if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    path = "runs/run0130_best_loss.pth"
    model, input_size, mean, std = load_multitask_resnet(path, device)

    mean = np.array(mean)
    std = np.array(std)

    print("input_size :", input_size)
    print("mean :", mean)
    print("std :", std)

    target_size = 30
    scale = (1.0, 1.0)
    rotation = True
    expansion_factor = 4
    val_size = 1000
    batch_size = 1

    val_transforms = T.Compose([
        T.Resize((input_size)),  # Make shortest edge this size
        T.CenterCrop(input_size),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std)
    ])

    val_dataset = LiveTargetDataset(transforms=val_transforms, input_size=expansion_factor*input_size,
        target_size=expansion_factor*target_size, length=val_size, scale=scale, rotation=rotation)

    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size)

    def labels_to_words(label):
        out = []
        out.append(int(label[0]))
        out.append(val_dataset.gen.shape_options[int(label[1])])
        out.append(val_dataset.gen.letter_options[int(label[2])])
        out.append(val_dataset.gen.color_options[int(label[3])])
        out.append(val_dataset.gen.color_options[int(label[4])])
        return out

    print("angle_quantization :", val_dataset.gen.angle_quantization)
    print("shape_options :", val_dataset.gen.shape_options)
    print("letter_options :", val_dataset.gen.letter_options)
    print("shape_color_options :", val_dataset.gen.color_options)
    print("letter_color_options :", val_dataset.gen.color_options)

    data, target = next(iter(val_loader))
    preds_idx = forward(model, data)

    print("target: ", target.squeeze().numpy())
    print(" preds: ", preds_idx)
    print("target: ", labels_to_words(target.squeeze().long().tolist()))
    print(" preds: ", labels_to_words(preds_idx))

    img = data.squeeze()
    img = T.Normalize(-mean/std,1/std)(img)
    img = img.permute(1,2,0) 
    plt.imshow(img)
    plt.show()

    # tasks_preds, all_targets = evaluate(model, val_loader)

