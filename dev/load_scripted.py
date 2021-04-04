import torch

if __name__ == "__main__":

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = torch.jit.load("runs/scripted_unet.pt", map_location=device)

    batch_size = 4

    x = torch.randn(batch_size, model.in_channels, model.input_size, model.input_size).to(device)

    print(model.in_channels)
    print(model.out_channels)
    print(model.model_type)
    print(model.input_size)

    logits = model(x)
    print("logits :", type(logits), len(logits), logits[0].shape)

    mask = model.predict(x)
    print("mask :", type(mask), mask.shape)
    