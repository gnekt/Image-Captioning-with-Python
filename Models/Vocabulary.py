import os

class Vocabulary():
    # The vocabulary implementation is done with a pre-trained word embedding matrix GLOVE50d
    # each word is represented by a record in a dataframe with this structure
    # id | word | dim1 | dim2 | ... | dim_n 
    
    def __init__(self, verbose: bool = False):
        
        self.vocabulary = {}
        
        self.enriched = False
        
        # Check if the enriched vocabulary already exists
        if os.path.exists(".saved/enriched_dataset_vocabulary.pkl"):
            self.vocabulary = pd.read_pickle(".saved/enriched_dataset_vocabulary.pkl")
            self.enriched = True
            return
        
        # Check if the base vocabulary(glove only + SOS + EOS + UNK) already exists
        if os.path.exists(".saved/base_vocabulary.pkl"):
            self.vocabulary = pd.read_pickle(".saved/base_vocabulary.pkl")
            return
        
        # Since the constructor arrived here, we need to load for the 1st time the glove word embeddings
        
        self.word2id = {}
        self.embeddings = {}
        
        # Initialize the token:
        # <PAD>, <SOS>, <EOS>, <UNK>
        self.word2id["<PAD>"] = 0
        self.word2id["<SOS>"] = 1
        self.word2id["<EOS>"] = 2
        self.word2id["<UNK>"] = 3
        
        self.embeddings["<PAD>"]
        counter = 0
        with open('.saved/glove.6B.50d.txt', 'r', encoding='utf-8') as _vocabulary_file:
            for line in _vocabulary_file:
                line = line.strip().split()
                self.word2id[line[0]] = counter
        
        

# ----------------------------------------------------------------
# Usage example

if __name__ == '__main__':
    v = Vocabulary(verbose=True)
    
        
        
        
        
            
        
        
    
    def translate(self, word_sequence : str):
        """Given a sequence of word, translate them according to the vocabulary.

        Args:
            word_sequence (str): [description]
        """
        
    