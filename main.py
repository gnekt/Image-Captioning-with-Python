

from torch.utils.data import DataLoader
from NeuralModels.FactoryModels import *
from NeuralModels.Dataset import MyDataset
from NeuralModels.Vocabulary import Vocabulary
import argparse

parameters = {
        0: {
            "question": "What neural net do you want to use? \n 1) CaRNet (ConvolutionalandRecurrentNet) \n 2) CARNet (ConvolutionalAttentionRecurrentNet)",
            "bound": (lambda value: 1<=value<=2),
            "value": -1
        },
        
    }

def parse_command_line_arguments():

    parser = argparse.ArgumentParser(description='CLI for C[aA]RNet, some static definition are placed in the VARIABLE.py file')
    
    parser.add_argument('decoder', choices=['vI','vH','vHC'],
                        help="What type of decoder do you want use?")
    
    parser.add_argument('mode', choices=['train', 'eval'],
                        help='train or evaluate C[aA]RNet.')
    
    parser.add_argument('attention', choices=[False, True], default=False, type=bool,
                        help='Use attention model. IF True, vHCAttention decoder and CResNet50Attention encoder are mandatories. (default: False)')
    
    parser.add_argument('encoder_dim', type=int,
                        help = 'Size of the encoder output. IF Attention is True fixed at 2048.')
    
    parser.add_argument('hidden_dim', type=int,
                        help = 'Capacity of the LSTM Cell.')
    
    parser.add_argument('--attention_dim', type=int, default=0, 
                        help="The attention capacity. Valid only if attention is true. (default: 0)")
    
    parser.add_argument('--dataset_folder', type=str, default="./dataset",
                        help='Data set folder. Used only if mode = train (Default: "./dataset")')
    
    parser.add_argument('--image_path', type=str, default="",
                        help = "The absolute path of the image that we want to retrieve the caption. Used only if mode = eval (Default: ''")
    
    parser.add_argument('--splits', type=(float), default=[.6,.3,.1],
                        help='Fraction of data to be used in train set, val set and test set (default: [.6,.3,.1])')
    
    parser.add_argument('--batch_size', type=int, default=32,
                        help='mini-batch size (default: 32)')
    
    parser.add_argument('--epochs', type=int, default=500,
                        help='number of training epochs (default: 500)')
    
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate (Adam) (default: 1e-3)')
    
    parser.add_argument('--workers', type=int, default=4,
                        help='number of working units used to load the data (default: 4)')
    
    parser.add_argument('--device', default='cpu', type=str,
                        help='device to be used for computations (in {cpu, cuda:0, cuda:1, ...}, default: cpu)')

    parsed_arguments = parser.parse_args()

    return parsed_arguments



if __name__ == "__main__":
    args = parse_command_line_arguments()
    print("Coded with love by christiandimaio aka gnekt :* ")
    
    for k, v in args.__dict__.items():
        print(k + '=' + str(v))
        
    dataset = MyDataset("./dataset/flickr30k_images", percentage=8)
    vocabulary = Vocabulary(dataset) 
    
    # Load Encoder and Decoder models
    attention = FactoryAttention(Attention.Attention)
    decoder = FactoryDecoder(Decoder.RNetvI)
    encoder = FactoryEncoder(Encoder.CResNet50)
    
    # # # Load the NeuralNet
    # net = FactoryNeuralNet(NeuralNet.CaRNet)(
    #                                             encoder=encoder,
    #                                             decoder=decoder,
    #                                             attention=attention,
    #                                             net_name="CaRNetvHCAttention",
    #                                             encoder_dim= 1024,
    #                                             hidden_dim= 512,
    #                                             padding_index= vocabulary.predefined_token_idx()["<PAD>"],
    #                                             vocab_size= len(vocabulary.word2id.keys()),
    #                                             embedding_dim= vocabulary.embeddings.shape[1],
    #                                             device="cuda:0"
    #                                         )
    # Load the NeuralNet
    
    net = FactoryNeuralNet(NeuralNet.CaRNet)(
                                                encoder=encoder,
                                                decoder=decoder,
                                                net_name="CaRNetvI",
                                                encoder_dim = vocabulary.embeddings.shape[1],
                                                hidden_dim= 1024,
                                                padding_index= vocabulary.predefined_token_idx()["<PAD>"],
                                                vocab_size= len(vocabulary.word2id.keys()),
                                                embedding_dim = vocabulary.embeddings.shape[1],
                                                device="cuda:0"
                                            )
    
    dc = dataset.get_fraction_of_dataset(percentage=1, delete_transfered_from_source=True)
    df = dataset.get_fraction_of_dataset(percentage=1, delete_transfered_from_source=True)
    # use dataloader facilities which requires a preprocessed dataset
       
    
    dataloader_training = DataLoader(dc, batch_size=32,
                        shuffle=True, num_workers=2, collate_fn = lambda data: dataset.pack_minibatch_training(data,vocabulary))
    
    dataloader_evaluation = DataLoader(df, batch_size=32,
                        shuffle=True, num_workers=2, collate_fn = lambda data: dataset.pack_minibatch_evaluation(data,vocabulary))
    
    
    net.train(
                train_set=dataloader_training,
                validation_set=dataloader_evaluation,
                lr=1e-3,
                epochs=500,
                vocabulary=vocabulary
            )