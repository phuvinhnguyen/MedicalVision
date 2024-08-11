
from torch import nn
def change_dropout_rate(model, new_dropout_rate):
    for name, module in model.named_modules():
        if isinstance(module, nn.Dropout):
            module.p = new_dropout_rate

def model_params(model):
    print(f'''Model state:
- All parameter: {sum(p.numel() for p in model.parameters())}
- Trainable parameter: {sum(p.numel() for p in model.parameters() if p.requires_grad)}''')
    
def set_all_params_to_trainable(model):
    for param in model.parameters():
        param.requires_grad = True