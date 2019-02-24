import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import copy
from train_unet import train_model
from models import ResNetUNet


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    num_class = 2

    model = ResNetUNet(num_class).to(device)

    if torch.cuda.is_available():
        print("MODEL ON CUDA")
        model.cuda()
    # freeze backbone layers
    # Comment out to finetune further
    for l in model.base_layers:
        for param in l.parameters():
            param.requires_grad = False

    optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=10, gamma=0.1)

    model = train_model(model, optimizer_ft, exp_lr_scheduler, num_epochs=15)
