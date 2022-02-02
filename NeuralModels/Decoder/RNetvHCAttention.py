import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple,List
from ..Attention.IAttention import IAttention


class RNetvHCAttention(nn.Module):
    """
        Class implementing LSTM unit with Attention model
    """
    def __init__(self, hidden_dim: int, padding_index: int, vocab_size: int, embedding_dim: int, device: str = "cpu", attention: IAttention = None):
        """Define the constructor for the RNN Net

        Args:
            hidden_dim (int): 
                The Capacity of the LSTM Cell.
            padding_index (int): 
                The index of the padding id, given from the vocabulary associated to the dataset.
            vocab_size (int)): 
                The size of the vocabulary associated to the dataset.
            embedding_size (int): 
                The number of dimension associated to the input of the LSTM cell.
            device (str, optional): Default "cpu"
                The device on which the operations will be performed. 
        """
        super(RNetvHCAttention, self).__init__()

        print(f"Construction of RNetvHC:\n\t Hidden Number of Dimensions: {hidden_dim},\n\t Padding Index: {padding_index},\n\t Vocabulary Size: {vocab_size},\n\t Embedding Number of Dimension: {embedding_dim},\n\t Device: {device}")
        
        self.device = torch.device(device)
        # Embedding layer that turns words into a vector of a specified size
        self.words_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_index)
        
        self.attention = attention 
        
        self.attention_dim = attention.attention_dim
        
        self.encoder_dim = attention.encoder_dim
        
        self.vocab_size = vocab_size
        
        # The initial memory state and hidden state of the LSTM are predicted by an average of the annotation vectors fed through two separate MLPs (init,c and init,h):
        self.h_0 = nn.Linear(self.encoder_dim, hidden_dim)
        self.c_0 = nn.Linear(self.encoder_dim, hidden_dim)
        
        # The LSTM takes embedded word vectors (of a specified size) as input
        # and outputs hidden states of size hidden_dim
        self.lstm_unit = torch.nn.LSTMCell(self.encoder_dim + embedding_dim, hidden_dim)
        
        
        # The linear layer that maps the hidden state output dimension
        # to the number of words we want as output, vocab_size
        self.linear_1 = nn.Linear(hidden_dim, vocab_size)
                

    def init_h_0_c_0(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Init hidden and cell state at t_0 

        Args:
            images (torch.Tensor): `(batch_dim, H_portions * W_portions, encoder_dim)`
                The images coming from the encoder.

        Returns:
            (torch.Tensor, torch.Tensor): `[(batch_dim, hidden_dim), (batch_dim, hidden_dim)]`
                Hiddent state and cell state ready for the 1st input
        """
        images = images.mean(dim=1) # Dim=0 -> batch_dim, Dim=1 -> H_portions * W_portions, Dim=2 -> encoder_dim
        return self.h_0(images), self.c_0(images)
    
    
    def forward(self, images: torch.Tensor, captions: torch.Tensor, captions_length: List[int]) -> Tuple[torch.Tensor, List[int], torch.Tensor]:
        """Compute the forward operation of the RNN.
                input of the LSTM cell for each time step:
                    t_{-1}: NONE 
                    t_0: Deterministict <START> 
                    .
                    .
                    .
                    t_{N-1}: The embedding vector associated to the S_{N-1} id.

        Args:
            images (torch.Tensor): `(batch_dim, H_portions, W_portions, encoder_dim)`
                The features associated to each image of the batch. 
            
            captions (torch.Tensor):  `(batch_dim, max_captions_length, embedding_dim)`
                The caption associated to each element of the batch.
                    REMARK Each caption is in the full form: <START> + .... + <END>
                    REMARK The Tensor is padded with zeros
                    
            caption_length ([int]): 
                The length of each caption in the batch. 
               
        Returns:
            (torch.Tensor): `(batch_dim, max_captions_length, vocab_size)`
                The hidden state of each time step from t_1 to t_{MaxN}. 
                
            (List(int)): 
                The length of each decoded caption. 
                    REMARK The <START> is provided as input at t_0.
                    REMARK The <END> token will be removed from the input of the LSTM.
            
            (torch.Tensor): `(batch_dim, max_captions_length, alphas)`
                All the alphas evaluated over timestep t (from t_0 to t_{N-1}), for each image in the batch.
        """             
        
        # Retrieve batch size 
        batch_dim = images.shape[0] # images is of shape (batch_dim, H_portions, W_portions, encoder_dim)
        
        # Create embedded word vector for each word in the captions
        inputs = self.words_embedding(captions) # In:       Out: (batch_dim, captions length, embedding_dim)
        
        
        # Initialize the hidden state and the cell state at time t_{-1}
        images = images.reshape(batch_dim,-1, images.shape[3]) # Out: (batch_dim, H_portions * W_portions, encoder_dim)
        _h, _c = self.init_h_0_c_0(images) # _h : (batch_dim, hidden_dim), _c : (batch_dim, hidden_dim)
        
        # Deterministict <START> Output as first word of the caption t_{0}
        start = torch.zeros(self.vocab_size).unsqueeze(0)
        start[0][1] = 1
        start = start.to(self.device)  # Out: (1, vocab_size)
        
        # Bulk insert of <START> to all the elements of the batch 
        outputs = start.repeat(batch_dim,1,1).to(self.device) # Out: (batch_dim, 1, vocab_size)
        
        # Tensor for storing alphas at each timestep t, structure (batch_dim, MaxN, number_of_splits^2) -> number_of_splits intended for a single Measure like Heigth and assuming square images
        alphas_t = torch.zeros((batch_dim,inputs.shape[1],self.attention.number_of_splits**2)).to(self.device)
        
        # Feed LSTMCell with image features and retrieve the state
        
        # How it works the loop?
        # For each time step t \in {0, N-1}, where N is the caption length 
                
        for idx in range(0,inputs.shape[1]): 
            attention_encoding, alphas_t_i = self.attention(images, _h) # Out: attention_encoding->(batch_dim,encoder_dim), alphas_t_i->(batch_dim, number_of_splits)
            alphas_t[:,idx,:] = alphas_t_i
            _h, _c = self.lstm_unit(torch.cat([attention_encoding,inputs[:,idx,:]], dim=1), (_h,_c))  # inputs[:,idx,:]: for all the captions in the batch, pick the embedding vector of the idx-th word in all the captions
            _outputs = self.linear_1(_h) # In: (batch_dim, hidden_dim), Out: (batch_dim, vocab_size)
            outputs = torch.cat((outputs,_outputs.unsqueeze(1)),dim=1) # Append in dim `1` the output of the LSTMCell for all the elements in batch
        
        return outputs, list(map(lambda length: length-1, captions_length)),alphas_t
    
    def generate_caption(self, image: torch.Tensor, captions_length: int) -> torch.Tensor:
        """Given the features vector retrieved by the encoder, perform a decoding (Generate a caption)

        Args:
        
            image (torch.tensor):  `(batch_dim, H_portions, W_portions, encoder_dim)`
                The image.
                
            captions_length (int): 
                The length of the caption.

        Returns:
        
            torch.tensor: 
                The caption associated to the image given. 
                    It includes <START> at t_0 by default.
        """
        
        sampled_ids = [torch.Tensor([1]).type(torch.int64).to(self.device)] # Hardcoded <START>
        input = self.words_embedding(torch.LongTensor([1]).to(torch.device(self.device))).reshape((1,-1))
        with torch.no_grad(): 
            image = image.reshape(1,-1, image.shape[2]) # Out: (batch_dim, H_portions * W_portions, encoder_dim)
            _h, _c = self.init_h_0_c_0(image)
            for _ in range(captions_length-1):
                attention_encoding, alpha = self.attention(image, _h)
                _h, _c = self.lstm_unit(torch.cat([attention_encoding, input], dim=1), (_h ,_c))           # _h: (1, 1, hidden_dim)
                outputs = self.linear_1(_h)            # outputs:  (1, vocab_size)
                _ , predicted = F.softmax(outputs,dim=1).cuda().max(1)  if self.device.type == "cuda" else   F.softmax(outputs,dim=1).max(1)  # predicted: The predicted id
                sampled_ids.append(predicted)
                input = self.words_embedding(predicted)                       # In: (batch_dim, embedding_dim)
                input = input.to(torch.device(self.device))                       # In: (1, 1, embedding_dim)
                if predicted == 2:
                    break
            sampled_ids = torch.stack(sampled_ids, 1)                # sampled_ids: (1, captions_length)
        return sampled_ids