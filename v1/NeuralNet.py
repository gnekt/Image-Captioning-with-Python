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
    def __init__(self, projection_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]   # remove last fc layer
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, projection_size) 
        
    def forward(self, images):
        features = self.resnet(images) 
        features = features.reshape(features.size(0), -1) # (Batch Size, Embedding Dim.)
        features = self.linear(features)
        return features
    
class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, padding_index, vocab_size, embedding_size):
        """[summary]

        Args:
            hidden_size ([type]): [description]
            padding_index ([type]): [description]
            vocab_size ([type]): [description]
            embedding_size ([type]): [description]
        """
        super(DecoderRNN, self).__init__()
    
        # Embedding layer that turns words into a vector of a specified size
        self.word_embeddings = nn.Embedding(vocab_size, embedding_size, padding_idx=padding_index)
        
        # The LSTM takes embedded word vectors (of a specified size) as input
        # and outputs hidden states of size hidden_dim
        self.lstm_unit = torch.nn.LSTMCell(embedding_size, hidden_size)
        
        # The linear layer that maps the hidden state output dimension
        # to the number of words we want as output, vocab_size
        self.linear_1 = nn.Linear(hidden_size, vocab_size)
                

    def forward(self, features, captions):
        """[summary]

        Args:
            features (torch.tensor(batch_size, hidden_size)): [description]
            captions (torch.tensor(batch_size, max_captions_length, word_embedding)): [description]

        Returns:
            [torch.tensor(batch_size, max_captions_length, vocab_size)]: [description]
        """             
        
        # Initialize the hidden state
        batch_size = features.shape[0] # features is of shape (batch_size, embed_size)
        
        embedding_size = features.shape[1] 
        # Create embedded word vectors for each word in the captions
        inputs = self.word_embeddings(captions) # embeddings new shape : (batch_size, captions length, embed_size)
        
        # Feed LSTMCell with image features and retrieve the state
        
        _h, _c = self.lstm_unit(features) # _h : (Batch size, Hidden size)
        
        # Deterministict <SOS> Output as first word of the caption :)
        start = torch.zeros(self.word_embeddings.num_embeddings)
        start[1] = 1
        outputs = start.repeat(batch_size,1,1).to(torch.device(device)) # Bulk insert of <SOS> embeddings to all the elements of the batch 
          
        
        
        # How it works the loop?
        # For each time step t \in {0, N-1}, where N is the caption length 
        
        # Since the sequences are padded, how the forward is performed? Since the <EOS> don't need to be feeded as input?
        # The assumption is that the captions are of lenght N-1, so the captions provide by external as input are without <EOS> token
        
        for idx in range(0,inputs.shape[1]): 
            _h, _c = self.lstm_unit(inputs[:,idx,:], (_h,_c))
            _outputs = self.linear_1(_h)
            outputs = torch.cat((outputs,_outputs.unsqueeze(1)),dim=1)
        
        return outputs # (Batch Size, N, |Vocabulary|)
    
    def sample(self, features):
        """Generate captions for given image features using greedy search."""
       
        sampled_ids = []
        input = self.word_embeddings(torch.LongTensor([1]).to(torch.device(device))).reshape((1,-1))
        with torch.no_grad(): 
            _h ,_c = self.lstm_unit(features.unsqueeze(0))
            for _ in range(15):
                _h, _c = self.lstm_unit(input, (_h ,_c))           # _h: (1, 1, hidden_size)
                outputs = self.linear_1(_h)            # outputs:  (1, vocab_size)
                _ , predicted = F.softmax(outputs,dim=1).cuda().max(1)  if device == "cuda" else   F.softmax(outputs,dim=1).max(1)                # predicted: (batch_size)
                sampled_ids.append(predicted)
                input = self.word_embeddings(predicted)                       # inputs: (batch_size, embed_size)
                input = input.to(torch.device(device))                       # inputs: (batch_size, 1, embed_size)
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

        encoder = EncoderCNN(512)
        decoder = DecoderRNN(512,0,len(vocabulary.word2id.keys()),512)
        
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

            for images,captions_training_ids,captions_target_ids in train_set:
                optimizer.zero_grad() 
                
                # zeroing the memory areas that were storing previously computed gradients
                batch_num_train_examples = images.shape[0]  # mini-batch size (it might be different from 'batch_size')
                epoch_num_train_examples += batch_num_train_examples
                
                
                images = images.to(device_t)
                captions_training_ids = captions_training_ids.to(device_t) # captions > (B, L)
                captions_target_ids = captions_target_ids.to(device_t) # captions > (B, |L|-1) without end token

                # computing the network output on the current mini-batch
                features = encoder(images)
                outputs = decoder(features, captions_training_ids) # outputs > (B, L, |V|); 
                
                # (B, L, |V|) -> (B * L, |V|) and captions > (B * L)
                loss = criterion(outputs.reshape((-1,outputs.shape[2])), captions_target_ids.reshape(-1))
                
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
                    print(vocabulary.rev_translate(captions_target_ids[numb]))
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