import torch
import torch.nn as nn
import torchvision.models as models

import torch.nn.functional as F

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = torch.hub.load('pytorch/vision:v0.6.0', 'resnet50', pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]   # remove last fc layer
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size) # attach a linear layer ()

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    
class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, padding_index, vocab_size, embeddings ):
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNN, self).__init__()
        # Keep track of hidden_size for initialization of hidden state
        self.hidden_size = hidden_size
        
        # Embedding layer that turns words into a vector of a specified size
        self.word_embeddings = nn.Embedding.from_pretrained(embeddings, freeze=True, padding_idx=padding_index)
        
        # The LSTM takes embedded word vectors (of a specified size) as input
        # and outputs hidden states of size hidden_dim
        self.lstm = nn.LSTM(input_size=50, \
                            hidden_size=hidden_size, # LSTM hidden units 
                            num_layers=1, # number of LSTM layer
                            batch_first=True,  # input & output will have batch size as 1st dimension
                            dropout=0, # Not applying dropout 
                            bidirectional=False, # unidirectional LSTM
                           )
        
        # The linear layer that maps the hidden state output dimension
        # to the number of words we want as output, vocab_size
        self.linear = nn.Linear(hidden_size, vocab_size)                     

    def init_hidden(self, batch_size):
        """ At the start of training, we need to initialize a hidden state;
        there will be none because the hidden state is formed based on previously seen data.
        So, this function defines a hidden state with all zeroes
        The axes semantics are (num_layers, batch_size, hidden_dim)
        """
        return (torch.zeros((1, batch_size, self.hidden_size)), \
                torch.zeros((1, batch_size, self.hidden_size)))
        
    
    def forward(self, features, captions):
        """ Define the feedforward behavior of the model """   
        
        # Discard the <end> word to avoid predicting when <end> is the input of the RNN
        captions = captions[:, :-1]    
        
        # Initialize the hidden state
        batch_size = features.shape[0] # features is of shape (batch_size, embed_size)
        self.hidden = self.init_hidden(batch_size) 
                
        # Create embedded word vectors for each word in the captions
        embeddings = self.word_embeddings(captions) # embeddings new shape : (batch_size, captions length -1, embed_size)
        
        # Stack the features and captions
        embeddings = torch.cat((features.unsqueeze(1), embeddings), dim=1) # embeddings new shape : (batch_size, caption length, embed_size)
        
        # Get the output and hidden state by passing the lstm over our word embeddings
        # the lstm takes in our embeddings and hidden state
        lstm_out, self.hidden = self.lstm(embeddings, self.hidden) # lstm_out shape : (batch_size, caption length, hidden_size)

        # Fully connected layer
        outputs = self.linear(lstm_out) # outputs shape : (batch_size, caption length + 1, vocab_size)

        return outputs
    
    def sample(self, features, states=None):
        """Generate captions for given image features using greedy search."""
        sampled_ids = []
        inputs = features.unsqueeze(1)
        inputs = inputs.reshape((1,1,inputs.shape[0]))
        for _ in range(30):
            hiddens, states = self.lstm(inputs, states)           # hiddens: (batch_size, 1, hidden_size)
            outputs = self.linear(hiddens.squeeze(1))            # outputs:  (batch_size, vocab_size)
            _, predicted = outputs.max(1)                        # predicted: (batch_size)
            sampled_ids.append(predicted)
            inputs = self.word_embeddings(predicted)                       # inputs: (batch_size, embed_size)
            inputs = inputs.unsqueeze(1)                         # inputs: (batch_size, 1, embed_size)
        sampled_ids = torch.stack(sampled_ids, 1)                # sampled_ids: (batch_size, max_seq_length)
        return sampled_ids



# Example of usage
if __name__ == "__main__":
    from Vocabulary import Vocabulary
    from PreProcess import PreProcess
    from Dataset import MyDataset
    from torch.utils.data import DataLoader
    ds = MyDataset("./dataset")
    df = ds.get_fraction_of_dataset(percentage=10)
    
    # use dataloader facilities which requires a preprocessed dataset
    v = Vocabulary(verbose=True)    
    df_pre_processed,v_enriched = PreProcess.DatasetForTraining.process(dataset=df,vocabulary=v)
    
    dataloader = DataLoader(df, batch_size=4,
                        shuffle=False, num_workers=0, collate_fn=df.pack_minibatch)
    
    encoder = EncoderCNN(50)
    decoder = DecoderRNN(100,0,len(v_enriched.word2id.keys()),v_enriched.embeddings)
    for images,captions in dataloader:
        features = encoder(images)
        caption = decoder.sample(features[0])
        print(v_enriched.rev_translate(caption))