
import numpy as np  # Random selection
from PIL import Image
import torch
import torchvision.transforms as T  # Image processing

from generate_targets import TargetGenerator
from models.old_unet import old_UNet
from models.unet import load_unet


def load_old_unet(path, device):
    model = old_UNet(in_channels=3, out_channels=1, features=16).to(device)
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    # TODO : did this model use input normalization
    set_mean = [0.4792167,  0.52474296, 0.27591285]
    set_std = [0.17430544, 0.15489028, 0.151296  ]
    transforms = T.Compose([
        T.ToTensor(),
        # T.Normalize(mean=set_mean, std=set_std)
    ])
    return model, transforms


def load_unet_regular(path, device):
    model = load_unet(path=path, device=device)
    model.eval()
    transforms = T.Compose([
        T.ToTensor(),
    ])
    return model, transforms


if __name__ == "__main__":

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    threshold = 0.5

    # Dataset parameters
    input_size = 400
    bkg_path = 'backgrounds/validate'
    target_size = 20  # Smallest target size
    fill_prob = 1.0
    expansion_factor = 3  # generate higher resolution targets and downscale, improves aliasing effects
    target_transforms = T.Compose([
        T.RandomPerspective(distortion_scale=0.5, p=1.0, interpolation=Image.BICUBIC),
    ])
    gen = TargetGenerator(input_size=input_size,
                target_size=target_size, expansion_factor=expansion_factor,
                target_transforms=target_transforms, bkg_path=bkg_path)

    # Load old model
    first_model, first_transforms = load_old_unet("dev/old_unet_weights.pth", device)
    # first_model, first_transforms = load_unet_regular("runs/unet/run00631_final.pth", device)

    # Load new model
    second_model, second_transforms = load_unet_regular("runs/unet/run00633_final.pth", device)
    # second_model, second_transforms = load_unet_regular("runs/unet/run00631_final.pth", device)

    # Create the visual
    nrows = 4
    ncols = 1
    rows = []
    for i in range(nrows):
        row = []
        for j in range(ncols):
            img, mask = gen.gen_segment(fill_prob=fill_prob)

            first_img = first_transforms(img).to(device).unsqueeze(0)
            second_img = second_transforms(img).to(device).unsqueeze(0)

            with torch.no_grad():
                first_out = first_model(first_img).detach().cpu().numpy()
                second_out = second_model(second_img).detach().cpu().numpy()

            row.append(img)

            # Process old model
            first_preds = (np.repeat(first_out[0][:, :, :], 3, axis=0).transpose(1, 2, 0) > threshold)
            first_fp_mask = ((1-np.array(mask)/255) * first_preds)
            first_fn_mask = (np.array(mask)/255 * (1-first_preds))
            row.append(first_preds*255)  # multiple to 255 rgb scale
            row.append(first_fp_mask*img)
            row.append(first_fn_mask*img)

            row.append(img)

            # Process new model
            second_preds = (np.repeat(second_out[0][:, :, :], 3, axis=0).transpose(1, 2, 0) > threshold)
            second_fp_mask = ((1-np.array(mask)/255) * second_preds)
            second_fn_mask = (np.array(mask)/255 * (1-second_preds))
            row.append(second_preds*255)  # multiple to 255 rgb scale
            row.append(second_fp_mask*img)
            row.append(second_fn_mask*img)

        rows.append( np.hstack(row) )
    grid_img = np.vstack(rows)
    # print(grid_img.shape)
    im = Image.fromarray(grid_img.astype('uint8'), 'RGB')
    im.show()
    im.save("images/unet_visualize_old_new.png")
