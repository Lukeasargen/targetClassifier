
import torch

from models.multitask import BuildMultiTaskResnet


def save_multitask_resnet(model, save_path, input_size, mean, std):
    data = {
        'model': model.state_dict(),
        'model_args': model.model_args,
        'input_size': input_size,
        'mean': mean,
        'std': std
    }
    torch.save(data, save_path)

def load_multitask_resnet(path, device='cpu'):
    data = torch.load(path, map_location=device)
    model = BuildMultiTaskResnet(**data['model_args'])
    model.load_state_dict(data['model'])
    return model, data['input_size'], data['mean'], data['std']


if __name__ == "__main__":

    # Make a MultiTaskResnet
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    input_size = 32
    in_channels = 3
    backbone_features = 128
    avgpool_size = 4
    filters = [64, 64, 128, 128]
    blocks = [2, 2, 2]
    bottleneck = False
    groups = 1
    width_per_group = None
    max_pool = False
    num_classes = [2, 3]

    set_mean = [0.528, 0.424, 0.374]
    set_std = [0.178, 0.171, 0.192]

    save_path = lambda r : 'runs/run{:04d}_best_loss.pth'.format(r)

    print(save_path(1, 3))

    m1 = BuildMultiTaskResnet(backbone_features, num_classes, in_channels, avgpool_size,
                filters, blocks, bottleneck, groups, width_per_group, max_pool)

    # create fake input
    x = torch.rand((1, 3, input_size, input_size))
    print(x.shape)
    
    # pass input and print output
    out1 = m1(x)

    print(input_size)
    print(set_mean)
    print(set_std)
    print("out1 :", out1)

    path = save_path(0)
    save_multitask_resnet(m1, path, input_size, set_mean, set_std)

    # load model
    m2, in2, mean, std = load_multitask_resnet(path, device)
    
    print(in2)
    print(mean)
    print(std)
    out2 = m2(x)
    print("out2 :", out2)
