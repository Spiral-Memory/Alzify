import torchvision.models as models
import torch.nn as nn

def build_model(pretrained = True, fine_tune = True):
    if pretrained:
        pass
        # print('Loading pretrained model...')
    elif not pretrained:
        print('Loading model without pretrained weights...')
    model = models.shufflenet_v2_x2_0(pretrained=pretrained)
    # model = models.squeezenet1_0(pretrained=pretrained)
    if fine_tune:
        print('Fine tuning model...')
        for param in model.parameters():
            param.requires_grad = True
    elif not fine_tune:
        pass
        # print('Freezing model...')
        for param in model.parameters():
            param.requires_grad = False
    model.fc = nn.Linear(2048,4)
    # model.classifier[1] = nn.Conv2d(512, 4, kernel_size=(1,1), stride=(1,1))
    return model

