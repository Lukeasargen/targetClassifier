import time

import torch


def model_time(model, data, n_trials=10, device='cpu', patches=1, only_eval=True, only_amp=False):
    model = model.to(device)
    data = data.to(device)
    n_images = n_trials*data.size(00)
    n_patches = n_images/patches
    warmup = 10
    if only_amp:
        with torch.cuda.amp.autocast():
            for i in range(n_trials):
                yhat = model(data)
    else:
        for i in range(warmup):
            yhat = model(data)

    print("Forward Method | Latency (ms) | Batches/sec | Patches/sec | Images/sec")

    if not only_eval:
        model.train()
        if not only_amp:
            t0 = time.time()
            for i in range(n_trials):
                yhat = model(data)
            duration = time.time() - t0
            print("train          | {:12.4f} | {:11.4f} | {:11.4f} | {:10.4f} ".format(1000*duration/n_trials, n_trials/duration, n_images/duration, n_patches/duration))

        t0 = time.time()
        with torch.cuda.amp.autocast():
            for i in range(n_trials):
                yhat = model(data)
        duration = time.time() - t0
        print("train w amp    | {:12.4f} | {:11.4f} | {:11.4f} | {:10.4f} ".format(1000*duration/n_trials, n_trials/duration, n_images/duration, n_patches/duration))

    model.eval()
    t0 = time.time()
    for i in range(n_trials):
        with torch.no_grad():
            yhat = model(data)
    duration = time.time() - t0
    print("eval (logits)  | {:12.4f} | {:11.4f} | {:11.4f} | {:10.4f} ".format(1000*duration/n_trials, n_trials/duration, n_images/duration, n_patches/duration))

    t0 = time.time()
    for i in range(n_trials):
        yhat = model.predict(data)
    duration = time.time() - t0
    print("predict        | {:12.4f} | {:11.4f} | {:11.4f} | {:10.4f} ".format(1000*duration/n_trials, n_trials/duration, n_images/duration, n_patches/duration))

    t0 = time.time()
    with torch.cuda.amp.autocast():
        for i in range(n_trials):
            yhat = model.predict(data)
    duration = time.time() - t0
    print("predict w amp  | {:12.4f} | {:11.4f} | {:11.4f} | {:10.4f} ".format(1000*duration/n_trials, n_trials/duration, n_images/duration, n_patches/duration))


if __name__ == "__main__":

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # from models.multitask import load_multitask_resnet
    # path = "runs/multitask/run00479_best_loss.pth"
    # model, input_size = load_multitask_resnet(path, device)
    # data = torch.rand((1, 3, input_size, input_size))
    # model_time(model, data, n_trials=100, device=device)

    from models.unet import UNet
    model_type = 'unet' # unet, unet_nested, unet_nested_deep
    in_channels = 3
    out_channels = 1
    filters = 16
    activation = 'relu'  # relu, leaky_relu, silu, mish

    batch = 16
    input_size = 192
    n_trials = 200
    only_eval = False
    only_amp = False

    data = torch.rand((batch, in_channels, input_size, input_size)).to(device)

    model = UNet(in_channels, out_channels, model_type, filters, activation).to(device)
    # model = torch.jit.script(model)

    img_width = 4056
    img_height = 3040
    from math import floor
    n_w, n_h = floor(img_width/input_size), floor(img_height/input_size)  # Center area only
    patch_per_image = n_w*n_h

    print("Batch :", batch)
    print("Input Size :", input_size)
    print("Patches per image : {} ({}, {})".format(patch_per_image, n_w, n_h))
    print("Total Pixels :", img_height*img_width)
    print("Segmented Pixels :", patch_per_image*input_size*input_size)
    print("Coverage : {:.2f}%".format((patch_per_image*input_size*input_size)/(img_height*img_width)))

    model_time(model, data, n_trials=n_trials, device=device, patches=patch_per_image, only_eval=only_eval, only_amp=only_amp)

