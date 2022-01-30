from re import S
import torch.nn as nn
import torch
import torchvision.models as models

class Attention(nn.Module):
    """
        Simple implementation of Bahdanau Attention model.
    """
    
    def __init__(self, image_features_projection: int , lstm_hidden_size: int, attention_size: int):
        
        super(Attention, self).__init__()
        
        self.attention_dim_size = attention_size
        
        self.image_attention_projection = nn.Linear(image_features_projection, attention_size)
        
        self.lstm_hidden_state_attention_projecton = nn.Linear(lstm_hidden_size, attention_size)
        
        self.attention = (attention_size, 1)
        
        self.ReLU = nn.ReLU()
        
        self.out = nn.Softmax(dim=1)
        
    def forward(self, images, lstm_hidden_states):
        
        _batch_size = images.shape[0]
        
        _images_attention = self.image_attention_projection(images.reshape(_batch_size,-1, self.attention_dim_size))
        
        _lstm_attention = self.image_attention_projection(lstm_hidden_states)
        
        _attention = self.ReLu(self.attention(_images_attention + _lstm_attention.unsqueeze(1)).squeeze(2))
        
        _alphas_t = self.out(_attention)
        
        _z_t = (images * _alphas_t.unsqueeze(2)).sum(dim=1)
        
        return _z_t, _alphas_t
        