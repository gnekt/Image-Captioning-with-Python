import torch.nn as nn
import torch
import torchvision.models as models

class CResNet50(nn.Module):
    def __init__(self, projection_size: int, device: str = "cpu"):
        """Constructor of the Encoder NN

        Args:
            projection_size (int): The dimension of projection into the space of RNN (Could be the input or the hidden state).
            
            device (str, optional): The device on which the operations will be performed. Default "cpu".
        """
        super(CResNet50, self).__init__()
        
        self.device = torch.device(device)
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters(): # Freezing weights 
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]   # remove last fc layer
        self.resnet = nn.Sequential(*modules)
        
        self.linear = nn.Linear(resnet.fc.in_features, projection_size) # define a last layer 
        
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Forward operation of the nn

        Args:
            images (torch.tensor): The tensor of the image in the form (Batch Size, Channels, Width, Height)

        Returns:
            [torch.tensor]: Features Projection in the form (Batch Size, Projection Dim.)
        """
        
        features = self.resnet(images)
        
        features = features.reshape(features.size(0), -1).to(self.device)
        features = self.linear(features)
        
        return features