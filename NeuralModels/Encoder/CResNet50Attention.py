import torch.nn as nn
import torch
import torchvision.models as models

class CResNet50Attention(nn.Module):
    def __init__(self, encoder_dim: int, number_of_splits_into_image: int = 3, device: str = "cpu"):
        """Constructor of the Encoder NN

        Args:
            encoder_dim (int): The dimension of projection into the space of RNN (Could be the input or the hidden state).
            
            device (str, optional): The device on which the operations will be performed. Default "cpu".
        """
        super(CResNet50Attention, self).__init__()
        
        self.device = torch.device(device)
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters(): # Freezing weights 
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-2]   # Expose the last convolutional layer. 512 Filters of size 3x3. Output of the ConvLayer -> (H_in/32,W_in/32,512) 
        
        self.encoder_dim = 2048 
        
        self.number_of_splits_into_image = number_of_splits_into_image
        
        # Q. Why (H_in/32, W_in/32)
        # A. Due to the resnet50 implementation, each convolutional layer will reduce the dimensionality of Heigth and Width by 2 times.
        
        self.resnet = nn.Sequential(*modules)
        
        # Applies a 2D adaptive average pooling over an input signal composed of several input planes.
        self.avg = torch.nn.AdaptiveAvgPool2d((number_of_splits_into_image,number_of_splits_into_image)) # IN : (BatchSize,2048,H_in/32,W_in/32) -> OUT : (BatchSize,2048,H_projection_size,W_projection_size)
        
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Forward operation of the nn

        Args:
            images (torch.tensor): The tensor of the image in the form (Batch Size, Channels, Width, Height)

        Returns:
            [torch.tensor]: Features Projection in the form (Batch Size, Projection Dim.)
        """
        
        features = self.resnet(images)
        features = self.avg(features)  # (batch_size, 2048, H_projection_size, W_projection_size)
        features = features.permute(0, 2, 3, 1)  # (batch_size, H_projection_size, W_projection_size, 2048)
        return features