
from torch import nn
def change_dropout_rate(model, new_dropout_rate):
    for name, module in model.named_modules():
        if isinstance(module, nn.Dropout):
            module.p = new_dropout_rate