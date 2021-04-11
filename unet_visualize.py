import os

import numpy as np  # Random selection
from PIL import Image
import torch
import torchvision.transforms as T  # Image processing

from generate_targets import TargetGenerator
from models.unet import load_unet
from metrics import dice_coeff, tversky_measure, jaccard_iou
from helper import pil_loader


def load_unet_regular(path, device):
    model = load_unet(path=path, device=device)
    model.eval()
    transforms = T.Compose([
        T.ToTensor(),
    ])
    return model, transforms


if __name__ == "__main__":

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    threshold = 0.99

    # Dataset parameters
    input_size = 256
    bkg_path = 'backgrounds/validate'
    target_size = 20  # Smallest target size
    scale = None # (1.0, 1.0) # None=random scale
    fill_prob = 0.5
    expansion_factor = 3  # generate higher resolution targets and downscale, improves aliasing effects
    target_transforms = T.Compose([
        T.RandomPerspective(distortion_scale=0.5, p=1.0),
    ])
    gen = TargetGenerator(input_size=input_size,
                target_size=target_size, expansion_factor=expansion_factor,
                target_transforms=target_transforms, bkg_path=bkg_path)

    # Load model
    first_model, first_transforms = load_unet_regular("runs/unet/run00863_final.pth", device)
    second_model, second_transforms = load_unet_regular("runs/unet/run00842_final.pth", device)

    # Create the visual
    nrows = 4
    ncols = 1
    rows = []
    for i in range(nrows):
        row = []
        for j in range(ncols):
            print("Image {}. Fill={:.02f}".format(i, fill_prob))
            img, mask = gen.gen_segment(scale=scale, fill_prob=fill_prob)
            mask_ten = T.Compose([T.Grayscale(num_output_channels=1), T.ToTensor()])(mask).to(device)

            # Process first model
            first_img = first_transforms(img).to(device).unsqueeze(0)
            first_out = first_model.predict(first_img)

            dice = dice_coeff(first_out, mask_ten)
            jaccard = jaccard_iou(first_out, mask_ten)
            tverskyfp = tversky_measure(first_out, mask_ten, alpha=1.0, beta=0.0)
            tverskyfn = tversky_measure(first_out, mask_ten, alpha=0.0, beta=1.0)
            print("Model 1 : dice={:.04f}. jaccard={:.04f}. tversky(fp)={:.04f}. tversky(fn)={:.04f}.".format(dice, jaccard, tverskyfp, tverskyfn))

            first_out = first_out.detach().cpu().numpy()
            first_preds = (np.repeat(first_out[0][:, :, :], 3, axis=0).transpose(1, 2, 0) > threshold)
            row.append(img)
            row.append(first_preds*255)  # multiple to 255 rgb scale

            first_fp_mask = ((1-np.array(mask)/255) * first_preds)
            first_fn_mask = (np.array(mask)/255 * (1-first_preds))
            row.append(first_fp_mask*img)
            row.append(first_fn_mask*img)


            # Process second model
            second_img = second_transforms(img).to(device).unsqueeze(0)
            second_out = second_model.predict(second_img)

            dice = dice_coeff(second_out, mask_ten)
            jaccard = jaccard_iou(second_out, mask_ten)
            tverskyfp = tversky_measure(second_out, mask_ten, alpha=1.0, beta=0.0)
            tverskyfn = tversky_measure(second_out, mask_ten, alpha=0.0, beta=1.0)
            print("Model 2 : dice={:.04f}. jaccard={:.04f}. tversky(fp)={:.04f}. tversky(fn)={:.04f}.".format(dice, jaccard, tverskyfp, tverskyfn))

            second_out = second_out.detach().cpu().numpy()
            second_preds = (np.repeat(second_out[0][:, :, :], 3, axis=0).transpose(1, 2, 0) > threshold)
            row.append(img)
            row.append(second_preds*255)  # multiple to 255 rgb scale

            second_fp_mask = ((1-np.array(mask)/255) * second_preds)
            second_fn_mask = (np.array(mask)/255 * (1-second_preds))
            row.append(second_fp_mask*img)
            row.append(second_fn_mask*img)

        rows.append( np.hstack(row) )
    grid_img = np.vstack(rows)
    # print(grid_img.shape)
    im = Image.fromarray(grid_img.astype('uint8'), 'RGB')
    im.show()
    im.save("images/unet_visualize.png")
