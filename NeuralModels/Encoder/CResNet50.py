import torch.nn as nn
import torch
import torchvision.models as models

class CResNet50(nn.Module):
    """
        Encoder Built with a resnet50 with the last layer removed.
    """
    
    def __init__(self, encoder_dim: int, device: str = "cpu"):
        """Constructor of the Encoder

        Args:
            encoder_dim (int): 
                The dimensionality of the features vector extracted from the image
            
            device (str, optional): Default "cpu".
                The device on which the operations will be performed.
        """
        super(CResNet50, self).__init__()
        
        self.encoder_dim = encoder_dim
        self.device = torch.device(device)
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters(): # Freezing weights 
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]   # remove last fc layer, expose the GlobalAveragePooling
        self.resnet = nn.Sequential(*modules)
        
        self.linear = nn.Linear(resnet.fc.in_features, encoder_dim) # define a last fc layer 
        
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Forward operation of the nn

        Args:
            images (torch.tensor):  `(batch_dim, channels, heigth, width)`
                The tensor of the images.

        Returns:
            [torch.tensor]: `(batch_dim, encoder_dim)`
                Features Projection for each image in the batch.
                
        """
        
        features = self.resnet(images) # Out: (batch_dim, 2048, 1, 1), 2048 is a Design choice of ResNet50 of last conv.layer.
        
        features = features.reshape(features.size(0), -1).to(self.device)
        features = self.linear(features) # In: (batch_dim, 2048)
        
        return features