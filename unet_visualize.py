
import numpy as np  # Random selection
from PIL import Image
import torch
import torchvision.transforms as T  # Image processing

from dataset_segment import LiveSegmentDataset
from old_unet import old_UNet
from models.unet import load_unet



if __name__ == "__main__":    

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    in_channels = 3
    out_channels = 1
    filters = 16

    threshold = 0.5

    # Dataset parameters
    train_size = 256
    input_size = 400
    target_size = 20
    bkg_path = 'backgrounds'  # path to background images, None is a random color background
    target_size = 20  # Smallest target size
    fill_prob = 0.9
    expansion_factor = 1  # generate higher resolution targets and downscale, improves aliasing effects
    target_transforms = T.Compose([
        T.RandomPerspective(distortion_scale=0.5, p=1.0, interpolation=Image.BICUBIC),
    ])
    train_dataset = LiveSegmentDataset(length=train_size,
        input_size=input_size, target_size=target_size,
        expansion_factor=expansion_factor, bkg_path=bkg_path, fill_prob=fill_prob,
        target_transforms=target_transforms)

    # Load old model
    old_model = old_UNet(in_channels, out_channels, filters).to(device)
    checkpoint = torch.load("old_unet_weights.pth", map_location=device)
    old_model.load_state_dict(checkpoint['model'])
    old_model.eval()
    old_set_mean = [0.4792167,  0.52474296, 0.27591285]
    old_set_std = [0.17430544, 0.15489028, 0.151296  ]
    old_transforms = T.Compose([
        T.Resize((input_size)),  # Make shortest edge this size
        T.CenterCrop(input_size),
        T.ToTensor(),
        # T.Normalize(mean=old_set_mean, std=old_set_std)
    ])

    # Load new model
    new_model, new_data = load_unet(path="runs/unet/run00555_final.pth", device=device)
    new_model.eval()
    new_transforms = T.Compose([
        T.Resize((input_size)),  # Make shortest edge this size
        T.CenterCrop(input_size),
        T.ToTensor(),
        T.Normalize(mean=new_data['mean'], std=new_data['std'])
    ])
    

    # Create the visual
    nrows = 5
    ncols = 1
    rows = []
    for i in range(nrows):
        row = []
        for j in range(ncols):
            img, mask = train_dataset.gen.gen_segment(fill_prob=0.3)

            old_img = old_transforms(img).to(device).unsqueeze(0)
            new_img = new_transforms(img).to(device).unsqueeze(0)

            with torch.no_grad():
                old_out = old_model(old_img).detach().cpu().numpy()
                new_out = new_model(new_img).detach().cpu().numpy()

            row.append(img)
            # row.append(mask)

            # Process old model
            old_preds = (np.repeat(old_out[0][:, :, :], 3, axis=0).transpose(1, 2, 0) > threshold)
            old_fp_mask = ((1-np.array(mask)/255) * old_preds)
            old_fn_mask = (np.array(mask)/255 * (1-old_preds))
            row.append(old_preds*255)  # multiple to 255 rgb scale
            row.append(old_fp_mask*img)
            row.append(old_fn_mask*img)

            row.append(img)
            # row.append(mask)

            # Process new model
            new_preds = (np.repeat(new_out[0][:, :, :], 3, axis=0).transpose(1, 2, 0) > threshold)
            new_fp_mask = ((1-np.array(mask)/255) * new_preds)
            new_fn_mask = (np.array(mask)/255 * (1-new_preds))
            row.append(new_preds*255)  # multiple to 255 rgb scale
            row.append(new_fp_mask*img)
            row.append(new_fn_mask*img)

        rows.append( np.hstack(row) )
    grid_img = np.vstack(rows)
    # print(grid_img.shape)
    im = Image.fromarray(grid_img.astype('uint8'), 'RGB')
    im.show()
    im.save("images/unet_visualize_old_new.png")
