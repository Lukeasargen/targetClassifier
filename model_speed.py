import time

import torch


def model_time(model, data, n_trials=10, device='cpu'):

    model = model.to(device)
    data = data.to(device)

    print("For train() : ", end="")
    model.train()
    t0 = time.time()
    for i in range(n_trials):
        yhat = model(data)
    duration = time.time() - t0
    print("Latency={:.4f} ms. FPS={:.3f}".format(1000*duration/n_trials, n_trials/duration))

    print("For train() with amp.autocast() : ", end="")
    t0 = time.time()
    with torch.cuda.amp.autocast():
        for i in range(n_trials):
            yhat = model(data)
    duration = time.time() - t0
    print("Latency={:.4f} ms. FPS={:.3f}".format(1000*duration/n_trials, n_trials/duration))

    print("For eval() : ", end="")
    model.eval()
    t0 = time.time()
    for i in range(n_trials):
        yhat = model(data)
    duration = time.time() - t0
    print("Latency={:.4f} ms. FPS={:.3f}".format(1000*duration/n_trials, n_trials/duration))

    print("For predict() : ", end="")
    t0 = time.time()
    for i in range(n_trials):
        yhat = model.predict(data)
    duration = time.time() - t0
    print("Latency={:.4f} ms. FPS={:.3f}".format(1000*duration/n_trials, n_trials/duration))

    print("For predict() with amp.autocast() : ", end="")
    t0 = time.time()
    with torch.cuda.amp.autocast():
        for i in range(n_trials):
            yhat = model.predict(data)
    duration = time.time() - t0
    print("Latency={:.4f} ms. FPS={:.3f}".format(1000*duration/n_trials, n_trials/duration))


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

    input_size = 192
    batch = 8

    model = UNet(in_channels, out_channels, model_type, filters, activation).to(device)
    data = torch.rand((batch, in_channels, input_size, input_size))
    model_time(model, data, n_trials=400, device=device)

