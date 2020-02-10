""" DeepLabv3 Model download and change the head for your prediction"""
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision import models


def createDeepLabv3(outputchannels, backboneFreez=False):
    model = models.segmentation.deeplabv3_resnet101(pretrained=True, progress=True)
    
    if backboneFreez:
        for param in model.parameters():
            param.requires_grad = False

    # Allocate a new classifier
    model.classifier = DeepLabHead(2048, outputchannels)

    # Set the model in training mode
    model.train()
    return model
