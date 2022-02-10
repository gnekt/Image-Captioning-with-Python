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

class CResNet50Attention(nn.Module):
    def __init__(self, encoder_dim: int, number_of_splits: int = 7, device: str = "cpu"):
        """Constructor of the Encoder NN

        Args:
            encoder_dim (int): Unused. Internally resnet with 2 layers removed represent each pixel in a vector of 2048 components.
            
            number_of_splits (int):
                How many pieces do you want to split the images, for border.
                    Examples:
                        number_of_splits = 7 -> The images will be splitted into 49 pieces (7x7). 
                
            device (str, optional): Default "cpu".
                The device on which the operations will be performed. 
        """
        super(CResNet50Attention, self).__init__()
        
        self.device = torch.device(device)
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters(): # Freezing weights 
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-2]   # Expose the last convolutional layer. 2048 Filters of size 1x1. Output of the ConvLayer -> (H_in/32,W_in/32,2048) 
        
        self.encoder_dim = 2048 
        
        self.number_of_splits = number_of_splits
        
        print(f"Construction of CResNet50Attention:\n \
                Encoder dimension: {self.encoder_dim},\n \
                Number of splits: {number_of_splits**2},\n \
                Device: {device}")
        # Q. Why (H_in/32, W_in/32)
        # A. Due to the resnet50 implementation, each convolutional layer will reduce the dimensionality of Heigth and Width by 2 times.
        
        self.resnet = nn.Sequential(*modules)
        
        
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Forward operation of the nn

        Args:
            images (torch.tensor):  `(batch_dim, Channels, Width, Height)`
                The tensor of the images.

        Returns:
            [torch.tensor]: `(batch_dim, H_splits, W_splits, encoder_dim)`
                Features Projection Tensor 
        """
        
        features = self.resnet(images) # Out: (batch_dim, 2048,Heigth/32, Width/32) 
        features = features.permute(0, 2, 3, 1)  # (batch_dim, H_splits, W_splits, 2048)
        return features