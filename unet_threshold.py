
import matplotlib.pyplot as plt
import numpy as np  # Random selection
from PIL import Image
import torch
import torchvision.transforms as T  # Image processing

from dataset_segment import LiveSegmentDataset
from models.unet import load_unet
from metrics import dice_coeff


if __name__ == "__main__":

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    input_size = 256

    target_transforms = T.Compose([
        T.RandomPerspective(distortion_scale=0.5, p=1.0, interpolation=Image.BICUBIC),
    ])
    train_dataset = LiveSegmentDataset(length=256,
        input_size=input_size, target_size=20,
        expansion_factor=3, bkg_path='backgrounds/validate', fill_prob=0.5,
        target_transforms=target_transforms)
    
    # Load new model
    model = load_unet(path="runs/unet/run00631_final.pth", device=device)
    model.eval()
    transforms = T.Compose([
        T.Resize((input_size)),  # Make shortest edge this size
        T.CenterCrop(input_size),
        T.ToTensor(),
    ])
    
    transform_mask = T.Compose([T.Grayscale(num_output_channels=1), T.ToTensor()])
    
    thresholds = np.arange(0, 1, 0.05)

    print(thresholds)

    output = []

    samples = 40

    for threshold in thresholds:
        total = 0.0
        for i in range(samples):
            img, mask = train_dataset.gen.gen_segment()
            img_batch = transforms(img).to(device).unsqueeze(0)
            with torch.no_grad():
                out = model(img_batch)
            total += dice_coeff(out, transform_mask(mask).unsqueeze(0).to(device)).item()
        total /= samples
        output.append(total)

    print(output)

    fig, ax = plt.subplots()

    ax.set_xlabel('threshold')
    ax.set_ylabel('dice')
    ax.tick_params(axis='y')
    ax.plot(thresholds, output)
    
    plt.show()
