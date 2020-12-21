import torch
import torch.nn as nn  # Building the model, loss func

x = torch.randn(10, 3)
y = torch.FloatTensor(10, 3).random_(2)

# double the loss for class 1
class_weight = torch.FloatTensor([1.0, 2.0, 1.0])
# double the loss for last sample
element_weight = torch.FloatTensor([1.0]*9 + [2.0]).view(-1, 1)
element_weight = element_weight.repeat(1, 3)

bce_criterion = nn.BCEWithLogitsLoss(weight=None, reduction='none')
multi_criterion = nn.MultiLabelSoftMarginLoss(weight=None, reduction='none')

bce_criterion_class = nn.BCEWithLogitsLoss(weight=class_weight, reduction='none')
multi_criterion_class = nn.MultiLabelSoftMarginLoss(weight=class_weight, reduction='none')

bce_criterion_element = nn.BCEWithLogitsLoss(weight=element_weight, reduction='none')
multi_criterion_element = nn.MultiLabelSoftMarginLoss(weight=element_weight, reduction='none')

bce_loss = bce_criterion(x, y)
multi_loss = multi_criterion(x, y)

print("bce_loss :", bce_loss)
print("multi_loss :", multi_loss)

bce_loss_class = bce_criterion_class(x, y)
multi_loss_class = multi_criterion_class(x, y)

print("bce_loss_class :", bce_loss_class)
print("multi_loss_class :", multi_loss_class)

bce_loss_element = bce_criterion_element(x, y)
multi_loss_element = multi_criterion_element(x, y)

print("bce_loss_element :", bce_loss_element)
print("multi_loss_element :", multi_loss_element)

print(torch.allclose(bce_loss.mean(1), multi_loss))
print(torch.allclose(bce_loss_class.mean(1), multi_loss_class))
print(torch.allclose(bce_loss_element.mean(1), multi_loss_element))
