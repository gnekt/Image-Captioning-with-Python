from torch.utils.data import DataLoader
from NeuralModels.FactoryModels import *
from NeuralModels.Dataset import MyDataset
from NeuralModels.Vocabulary import Vocabulary
import argparse
import sys, os
from PIL import Image

def parse_command_line_arguments():

    parser = argparse.ArgumentParser(description='CLI for C[aA]RNet, some static definition are placed in the VARIABLE.py file')
    
    parser.add_argument('decoder', type=Decoder.argparse, choices=list(Decoder),
                        help="What type of decoder do you want use?")
    
    parser.add_argument('mode', choices=['train', 'eval'],
                        help='train or evaluate C[aA]RNet.')
    
    parser.add_argument('encoder_dim', type=int,
                        help = 'Size of the encoder output. IF Attention is True, fixed at 2048. IF CaRNetvI as net, encoder_dim == |vocabulary|.')
    
    parser.add_argument('hidden_dim', type=int,
                        help = 'Capacity of the LSTM Cell.')
    
    parser.add_argument('--attention', default=False, type=bool,
                        help='Use attention model. IF True, vHCAttention decoder and CResNet50Attention encoder are mandatories. (default: False)')
    
    
    parser.add_argument('--attention_dim', type=int, default=0, 
                        help="The attention capacity. Valid only if attention is true. (default: 0)")
    
    parser.add_argument('--dataset_folder', type=str, default="./dataset",
                        help='Data set folder. Used only if mode = train (Default: "./dataset")')
    
    parser.add_argument('--image_path', type=str, default="",
                        help = "The absolute path of the image that we want to retrieve the caption. Used only if mode = eval (Default: ''")
    
    parser.add_argument('--splits', type=int, nargs="+", default=[60,30,10],
                        help='Fraction of data to be used in train set, val set and test set (default: 60 30,10)')
    
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
    print("Coded with love by christiandimaio aka gnekt :* \n ")
    args = parse_command_line_arguments()
    
    for k, v in args.__dict__.items():
        print(k + '=' + str(v))
    
    #################################### Define Encoder/Decoder
    encoder = None 
    decoder = None
    attention = None
    if args.attention == True:
        # Attention is true, encoder = CResNet50Attention, decoder = RNetvHCAttention
        encoder = FactoryEncoder(Encoder.CResNet50Attention)
        decoder = FactoryDecoder(Decoder.RNetvHCAttention)
        attention = FactoryAttention(Attention.Attention)
        args.net_name = "CARNetvHCAttention"
        
    if args.attention == False:
        args.net_name = f"Ca{args.decoder.name}"
        encoder = FactoryEncoder(Encoder.CResNet50)
        decoder = FactoryDecoder(args.decoder)
    ####################################
    
    #################################### Construct Data
    print("Construct data..")
    
    if args.mode == "train":
        print("Define dataset..")
        dataset = MyDataset(args.dataset_folder, percentage=8) # Percentage is fixed cause the dataset is HUGE, 8% is enough for sperimental test.
        print("OK.")
        
        print("Define vocabulary..")
        vocabulary = Vocabulary(dataset)
        print("OK.")
        
        # Obtain train, validation and test set
        print("Obtain train, validation and test set..")
        train_set = dataset.get_fraction_of_dataset(percentage=args.splits[0], delete_transfered_from_source=True)
        validation_set = dataset.get_fraction_of_dataset(percentage=args.splits[1], delete_transfered_from_source=True)
        test_set  = dataset.get_fraction_of_dataset(percentage=args.splits[2], delete_transfered_from_source=True)
        print("OK.")
        
        # Define the associate dataloader
        print("Define the associate dataloader")
        dataloader_training = DataLoader(train_set, batch_size=args.batch_size,
                        shuffle=True, num_workers=args.workers, collate_fn = lambda data: dataset.pack_minibatch_training(data,vocabulary))
        dataloader_validation = DataLoader(validation_set, batch_size=args.batch_size,
                        shuffle=True, num_workers=args.workers, collate_fn = lambda data: dataset.pack_minibatch_evaluation(data,vocabulary))
        dataloader_test = DataLoader(test_set, batch_size=args.batch_size,
                        shuffle=True, num_workers=args.workers, collate_fn = lambda data: dataset.pack_minibatch_evaluation(data,vocabulary))
        print("OK.")
        
    if args.mode == "eval":
        print("Define vocabulary..")
        vocabulary = Vocabulary()
        print("Ok.")

        print("Load the image..")
        if not os.path.exists(args.image_path) or os.path.isdir(args.image_path):
            raise ValueError(f"Got {args.image_path} as file path, error!")
        image: Image.Image = Image.open(args.image_path).convert('RGB') 
        print("Ok.")
    ####################################
    
    #################################### Define Net
    print("Create the net..")
    net = FactoryNeuralNet(NeuralNet.CaRNet)(
        encoder=encoder,
        decoder=decoder,
        attention=attention, # != None only if Attention is requested
        attention_dim = args.attention_dim, # != 0 only if Attention is True
        net_name=args.net_name,
        encoder_dim = args.encoder_dim if args.decoder is not Decoder.RNetvI else vocabulary.embeddings.shape[1], # if Attention is True encoder_dim hasn't any meaning, cause it is 2048 internally by construction.
        hidden_dim= args.hidden_dim,
        padding_index= vocabulary.predefined_token_idx()["<PAD>"],
        vocab_size= len(vocabulary.word2id.keys()),
        embedding_dim = vocabulary.embeddings.shape[1],
        device=args.device
    )
    print("OK.")
    #################################### Load a previous trained net, if exist
    
    print("Check if it is present a previous version of the Net..")
    try:
        net.load("./.saved")
        print("Found.")
    except Exception as ex:
        print("An exception has occurred.")
        print(ex)
        if args.mode == "eval": # If the mode is eval the script cannot continue
            print("Since you want an evaluation, the script cannot continue, please retrain the network.")
            sys.exit(0)
        # In training it creates new files.
        print("Not Found.")
        print("Since the selected mode is training, a new instance of the net will saved during the training activity.")
    
    #################################### Training or Evaluate
    
    if args.mode == "train":
        print("Start training..")
        net.train(
                train_set=dataloader_training,
                validation_set=dataloader_validation,
                lr=args.lr,
                epochs=args.epochs,
                vocabulary=vocabulary
            )
        # Evaluate Test set
        print("Done")
        print(f"Test set Accuracy: {net.eval_net(dataloader_test, vocabulary):.4f}")
        
    if args.mode == "eval":
        print("Start evaluation..")
        net.eval(image, vocabulary)
        print("OK.")
    ####################################