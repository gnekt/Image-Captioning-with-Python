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
    
    