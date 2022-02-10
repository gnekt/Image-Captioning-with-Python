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

import torch
import torch.nn as nn

class IAttention(nn.Module):
    """
        Class interface for Attention unit 
        Args are intended as suggested.
    """
    def __init__(self, *args):
        """Constructor for an Attention model 

        Args:
            encoder_dim (int): 
                The number of features extracted from the image.
            hidden_dim (int): 
                The capacity of the LSTM.
            attention_dim (int): 
                The capacity of the Attention Model.
        """
        super(IAttention, self).__init__()
        
    def forward(self, *args):
        """Compute z_t given images and hidden state at t-1 for all the element in the batch.

        Args:
            images (torch.Tensor): `(batch_dim, image_portions, encoder_dim)`
                The tensor of the images in the batch. 
            lstm_hidden_states (torch.Tensor): `(batch_dim, hidden_dim)`
                The hidden states at t-1 of the elements in the batch. 

        Returns:
            (Tuple[torch.Tensor,torch.Tensor]): `[(batch_dim, encoder_dim), (batch_dim, image_portions)]`
                Z_t and the alphas evaluated for each portion of the image, for each image in the batch.
        """
        pass
    
    