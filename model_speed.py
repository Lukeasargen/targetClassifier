import time

import torch


def model_time(model, data, n_trials=10, device='cpu'):

    model = model.to(device)
    data = data.to(device)

    model.train()
    t0 = time.time()
    for i in range(n_trials):
        yhat = model(data)
    duration = time.time() - t0
    print("For model.train() :")
    print("  Duration {:.6f} s. Seconds per input={:.6f} s. FPS={:.4f}".format(duration, duration/n_trials, n_trials/duration))

    model.eval()
    t0 = time.time()
    for i in range(n_trials):
        yhat = model(data)
    duration = time.time() - t0
    print("For model.eval() :")
    print("  Duration {:.6f} s. Seconds per input={:.6f} s. FPS={:.4f}".format(duration, duration/n_trials, n_trials/duration))


    t0 = time.time()
    with torch.no_grad():
        for i in range(n_trials):
            yhat = model(data)
    duration = time.time() - t0
    print("For no_grad() :")
    print("  Duration {:.6f} s. Seconds per input={:.6f} s. FPS={:.4f}".format(duration, duration/n_trials, n_trials/duration))


if __name__ == "__main__":

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # from models.multitask import load_multitask_resnet
    # path = "runs/multitask/run00479_best_loss.pth"
    # model, input_size = load_multitask_resnet(path, device)
    # data = torch.rand((1, 3, input_size, input_size))
    # model_time(model, data, n_trials=100, device=device)

    from models.unet import UNet
    model = UNet(in_channels=3, out_channels=1, filters=16)
    input_size = 400
    data = torch.rand((1, 3, input_size, input_size))
    model_time(model, data, n_trials=200, device=device)

