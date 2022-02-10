# MIT License

# Copyright (c) 2022 christiandimaio

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

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
        
        print(f"Construction of CResNet50:\n \
                Encoder dimension: {encoder_dim},\n \
                Device: {device}")
        
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