import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.functional as F

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]   # remove last fc layer
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size) 
        
    def init_weights(self):
        # weight init, inspired by tutorial
        self.embed.weight.data.normal_(0,0.02)
        self.embed.bias.data.fill_(0)
        
    def forward(self, images):
        
        features = self.resnet(images) 
        features = features.reshape(features.size(0), -1)
        features = self.linear(features)
        return features
    
class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, padding_index, vocab_size, embeddings ):
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNN, self).__init__()
        # Keep track of hidden_size for initialization of hidden state
        self.hidden_size = hidden_size
        
        # Embedding layer that turns words into a vector of a specified size
        self.word_embeddings = nn.Embedding.from_pretrained(embeddings, freeze=True, padding_idx = 0)
        
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
        
    
    def forward(self, features, captions,caption_lengths):
        """ Define the feedforward behavior of the model """      
        
        # Initialize the hidden state
        batch_size = features.shape[0] # features is of shape (batch_size, embed_size)
        self.hidden = self.init_hidden(batch_size)
        
        # Create embedded word vectors for each word in the captions
        embeddings = self.word_embeddings(captions) # embeddings new shape : (batch_size, captions length -1, embed_size)
        
        # Stack the features and captions
        inputs = torch.cat((features.unsqueeze(1), embeddings), dim=1) # embeddings new shape : (batch_size, caption length, embed_size)
        
        # Get the output and hidden state by passing the lstm over our word embeddings
        # the lstm takes in our embeddings and hidden state
        lstm_out, self.hidden = self.lstm(inputs) # lstm_out shape : (batch_size, caption length, hidden_size)

        # Fully connected layer
        outputs = self.linear(lstm_out) # outputs shape : (batch_size, caption length, vocab_size)

        return outputs
    
    def sample(self, features, states=None):
        """Generate captions for given image features using greedy search."""
        sampled_ids = []
        inputs = features.unsqueeze(1)
        inputs = inputs.reshape((1,1,inputs.shape[0]))
        self.init_hidden(1)
        with torch.no_grad(): 
            for _ in range(30):
                hiddens, states = self.lstm(inputs, states)           # hiddens: (batch_size, 1, hidden_size)
                outputs = self.linear(hiddens.squeeze(1))            # outputs:  (batch_size, vocab_size)
                _, predicted = outputs.max(1)                     # predicted: (batch_size)
                sampled_ids.append(predicted)
                inputs = self.word_embeddings(predicted)                       # inputs: (batch_size, embed_size)
                inputs = inputs.unsqueeze(1)                         # inputs: (batch_size, 1, embed_size)
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
        device = torch.device("cuda:0")
        criterion = nn.CrossEntropyLoss(ignore_index=0).cuda()
        
        # initializing some elements
        best_val_acc = -1.  # the best accuracy computed on the validation data
        best_epoch = -1  # the epoch in which the best accuracy above was computed

        encoder = EncoderCNN(50)
        decoder = DecoderRNN(2048,0,len(vocabulary.word2id.keys()),vocabulary.embeddings)
        
        encoder.to(device)
        decoder.to(device)
        
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

            for images,captions,captions_length in train_set:
                optimizer.zero_grad() 
                
                # zeroing the memory areas that were storing previously computed gradients
                batch_num_train_examples = images.shape[0]  # mini-batch size (it might be different from 'batch_size')
                epoch_num_train_examples += batch_num_train_examples
                
                lengths = Variable(torch.LongTensor(captions_length))
                    
                lengths = lengths.to(device)
                images = images.to(device)
                captions = captions.to(device)

                # computing the network output on the current mini-batch
                features = encoder(images)
                outputs = decoder(features, captions,lengths)[:,:-1,:]
                
                
               
                loss = criterion(outputs.reshape((-1,outputs.shape[2])), captions.reshape(-1))
                
                # computing gradients and updating the network weights
                loss.backward()  # computing gradients
                optimizer.step()  # updating weights

                print(f"mini-batch:\tloss={loss.item():.4f}")
                torch.save(decoder.state_dict(),".saved/decoder.pt")
                with torch.no_grad():
                    decoder.eval()
                    encoder.eval()
                    features = encoder(images)
                    caption = decoder.sample(features[0])
                    print(vocabulary.rev_translate(captions))
                    print(vocabulary.rev_translate(caption))
                    decoder.train()
                    encoder.train()
                    
                # computing the performance of the net on the current training mini-batch
                # with torch.no_grad():  # keeping these operations out of those for which we will compute the gradient
                #     self.net.eval()  # switching to eval mode

                #     # computing performance
                #     batch_train_acc = self.__performance(outputs, y)

                #     # accumulating performance measures to get a final estimate on the whole training set
                #     epoch_train_acc += batch_train_acc * batch_num_train_examples

                #     # accumulating other stats
                #     epoch_train_loss += loss.item() * batch_num_train_examples

                #     self.net.train()  # going back to train mode

                #     # printing (mini-batch related) stats on screen
                #     print("  mini-batch:\tloss={0:.4f}, tr_acc={1:.2f}".format(loss.item(), batch_train_acc))

            # val_acc = self.eval_classifier(validation_set)

            # # saving the model if the validation accuracy increases
            # if val_acc > best_val_acc:
            #     best_val_acc = val_acc
            #     best_epoch = e + 1
            #     self.save("classifier.pth")

            # epoch_train_loss /= epoch_num_train_examples

            # # printing (epoch related) stats on screen
            # print(("epoch={0}/{1}:\tloss={2:.4f}, tr_acc={3:.2f}, val_acc={4:.2f}"
            #        + (", BEST!" if best_epoch == e + 1 else ""))
            #       .format(e + 1, epochs, epoch_train_loss,
            #               epoch_train_acc / epoch_num_train_examples, val_acc))

# Example of usage
if __name__ == "__main__":
    from Vocabulary import Vocabulary
    from Dataset import MyDataset
    from torch.utils.data import DataLoader
    ds = MyDataset("./dataset", percentage=10)
    
    # use dataloader facilities which requires a preprocessed dataset
    v = Vocabulary(ds,reload=True)    
    
    dataloader = DataLoader(ds, batch_size=15,
                        shuffle=True, num_workers=0, collate_fn = lambda data: ds.pack_minibatch_training(data,v))
    
    train(dataloader, dataloader, 1e-3, 10, v)