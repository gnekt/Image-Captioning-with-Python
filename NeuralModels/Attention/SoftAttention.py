from re import S
import torch.nn as nn
import torch
import torchvision.models as models

class SoftAttention(nn.Module):
    """
        Simple implementation of Bahdanau Attention model.
    """
    
    def __init__(self, encoder_dim: int , hidden_dim: int, attention_dim: int):
        
        super(SoftAttention, self).__init__()
        
        self.attention_dim_size = attention_dim
        
        self.image_features_projection = encoder_dim
        
        self.image_attention_projection = nn.Linear(encoder_dim, attention_dim)
        
        self.lstm_hidden_state_attention_projection = nn.Linear(hidden_dim, attention_dim)
        
        self.attention = nn.Linear(attention_dim, 1)
        
        self.ReLU = nn.ReLU()
        
        self.out = nn.Softmax(dim=1)
        
        # the soft attention model predicts a gating scalar β from previous hidden state ht−1 at each time step t, such that, φ ({ai} , {αi}) = β PL i αiai, where βt = σ(fβ(ht−1))
        # Par. 4.2.1
        self.f_beta = nn.Linear(hidden_dim, self.image_features_projection)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, images, lstm_hidden_states):
        
        _batch_size = images.shape[0]
        
        _images_attention = self.image_attention_projection(images)
        
        _lstm_attention = self.lstm_hidden_state_attention_projection(lstm_hidden_states)
        
        _attention = self.ReLU(self.attention(_images_attention + _lstm_attention.unsqueeze(1)).squeeze(2))
        
        _alphas_t = self.out(_attention)
        
        betas_t = self.f_beta(lstm_hidden_states)
        
        gate = self.sigmoid(betas_t)

        _z_t = gate * (images * _alphas_t.unsqueeze(2)).sum(dim=1)
        
        return _z_t, _alphas_t
        