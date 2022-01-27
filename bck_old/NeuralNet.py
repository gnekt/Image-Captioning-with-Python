import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.functional as F
import random

device = "cuda:0"
class EncoderCNN(nn.Module):
    def __init__(self, embedding_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]   # remove last fc layer
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embedding_size) 
        
    def forward(self, images):
        features = self.resnet(images) 
        features = features.reshape(features.size(0), -1) # (Batch Size, Embedding Dim.)
        features = self.linear(features)
        return features
    
class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, padding_index, vocab_size, embeddings, embedding_size):
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNN, self).__init__()
        
        # Embedding layer that turns words into a vector of a specified size
        self.word_embeddings = nn.Embedding(vocab_size, embedding_size, padding_idx=padding_index)
        
        # The LSTM takes embedded word vectors (of a specified size) as input
        # and outputs hidden states of size hidden_dim
        self.lstm_unit = torch.nn.LSTMCell(embedding_size, hidden_size)
        
        # The linear layer that maps the hidden state output dimension
        # to the number of words we want as output, vocab_size
        self.linear_1 = nn.Linear(hidden_size, vocab_size)

    def init_hidden_state(self, encoder_out):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :return: hidden state, cell state
        """
        
        h = encoder_out.reshape((1,encoder_out.shape[0],encoder_out.shape[1]))  # (batch_size, decoder_dim)
        c = encoder_out.reshape((1,encoder_out.shape[0],encoder_out.shape[1]))
        return h, c
        
    
    def forward(self, features, captions,caption_lengths):
        """ Define the feedforward behavior of the model """      
        
        # Initialize the hidden state
        batch_size = features.shape[0] # features is of shape (batch_size, embed_size)
        
        # Create embedded word vectors for each word in the captions
        inputs = self.word_embeddings(captions) # embeddings new shape : (batch_size, captions length, embed_size)
        
       
        # Get the output and hidden state by passing the lstm over our word embeddings
        # the lstm takes in our embeddings and hidden state
        #h, c = self.init_hidden_state(features) 
        inputs = torch.cat((features.unsqueeze(1), inputs), dim=1)
        lstm_out, self.hidden = self.lstm(inputs) # lstm_out shape : (batch_size, caption length, hidden_size), Defaults to zeros if (h_0, c_0) is not provided.
        
        lstm_out = lstm_out[:,1:,:]
        # Fully connected layers
        outputs = self.linear_1(lstm_out) # outputs shape : (batch_size, caption length, vocab_size)
        
        return outputs
    
    def sample(self, features):
        """Generate captions for given image features using greedy search."""
       
        sampled_ids = []
        input = self.word_embeddings(torch.LongTensor([1]).to(torch.device(device))).reshape((1,1,-1))
        with torch.no_grad(): 
            print(features.shape)
            _ ,state = self.lstm(features.reshape(1,1,-1))
            for _ in range(15):
                hiddens, state = self.lstm(input, state)           # hiddens: (batch_size, 1, hidden_size)
                outputs = self.linear_1(hiddens.squeeze(1))            # outputs:  (batch_size, vocab_size)
                _, predicted = F.softmax(outputs,dim=1).cuda.max(1)  if device == "cuda" else   F.softmax(outputs,dim=1).max(1)                # predicted: (batch_size)
                sampled_ids.append(predicted)
                inputs = self.word_embeddings(predicted)                       # inputs: (batch_size, embed_size)
                input = inputs.unsqueeze(1).to(torch.device(device))                       # inputs: (batch_size, 1, embed_size)
                if predicted == 2:
                    break
            sampled_ids = torch.stack(sampled_ids, 1)                # sampled_ids: (batch_size, max_seq_length)
        return sampled_ids

def save(self, file_name):
    """Save the classifier."""

    torch.save(self.net.state_dict(), file_name)

def load(self, file_name):
    """Load the classifier."""

    # since our classifier is a nn.Module, we can load it using pytorch facilities (mapping it to the right device)
    self.net.load_state_dict(torch.load(file_name, map_location=self.device))
        
def train(train_set, validation_set, lr, epochs, vocabulary):
        device_t = torch.device(device)
        criterion = nn.CrossEntropyLoss(ignore_index=0,reduction="sum").cuda() if device == "cuda" else nn.CrossEntropyLoss(ignore_index=0,reduction="sum")
        
        # initializing some elements
        best_val_acc = -1.  # the best accuracy computed on the validation data
        best_epoch = -1  # the epoch in which the best accuracy above was computed

        encoder = EncoderCNN(50)
        decoder = DecoderRNN(1024,0,len(vocabulary.word2id.keys()),vocabulary.embeddings)
        
        encoder.to(device_t)
        decoder.to(device_t)
        
        # ensuring the classifier is in 'train' mode (pytorch)
        decoder.train()

        # creating the optimizer
        optimizer = torch.optim.Adam(list(decoder.parameters()) + list(encoder.linear.parameters()), lr)

        # loop on epochs!
        for e in range(0, epochs):

            # epoch-level stats (computed by accumulating mini-batch stats)
            epoch_train_acc = 0.
            epoch_train_loss = 0.
            epoch_num_train_examples = 0

            for images,captions,captions_length,captions_training in train_set:
                optimizer.zero_grad() 
                
                # zeroing the memory areas that were storing previously computed gradients
                batch_num_train_examples = images.shape[0]  # mini-batch size (it might be different from 'batch_size')
                epoch_num_train_examples += batch_num_train_examples
                
                lengths = Variable(torch.LongTensor(captions_length))
                    
                lengths = lengths.to(device_t)
                images = images.to(device_t)
                captions = captions.to(device_t) # captions > (B, L)
                captions_training = captions_training.to(device_t) # captions > (B, |L|-1) without end token

                # computing the network output on the current mini-batch
                features = encoder(images)
                outputs = decoder(features, captions,lengths) # outputs > (B, L, |V|); 
                
                # (B, L, |V|) -> (B * L, |V|) and captions > (B * L)
                loss = criterion(outputs.reshape((-1,outputs.shape[2])), captions.reshape(-1))
                
                # computing gradients and updating the network weights
                loss.backward()  # computing gradients
                optimizer.step()  # updating weights

                print(f"mini-batch:\tloss={loss.item():.4f}")
            with torch.no_grad():
                decoder.eval()
                encoder.eval()
                features = encoder(images)
                numb = random.randint(0,2)
                caption = decoder.sample(features[numb])
                print(vocabulary.rev_translate(captions[numb]))
                print(vocabulary.rev_translate(caption[0]))
                decoder.train()
                encoder.train()

# Example of usage
if __name__ == "__main__":
    from Vocabulary import Vocabulary
    from Dataset import MyDataset
    from torch.utils.data import DataLoader
    ds = MyDataset("./dataset", percentage=1)
    ds = ds.get_fraction_of_dataset(percentage=100)
    # use dataloader facilities which requires a preprocessed dataset
    v = Vocabulary(ds,reload=True)    
    
    dataloader = DataLoader(ds, batch_size=30,
                        shuffle=True, num_workers=4, collate_fn = lambda data: ds.pack_minibatch_training(data,v))
    
    train(dataloader, dataloader, 1e-3, 400, v)