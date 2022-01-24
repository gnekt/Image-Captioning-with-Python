import os
import torch
import warnings
from Dataset import MyDataset
class Vocabulary():
    # The vocabulary implementation is done with a pre-trained word embedding GLOVE50d
    # each word is represented by a record in a dataframe with this structure
    
    
    def __init__(self, source_dataset: MyDataset, verbose: bool = False, reload: bool = False):
        
        self.enriched = False       # Tell that all the word coming from the dataset are in the vocabulary if it is set to True
        self._make_enrich = False         # Allow the user to enrich the vocabulary if it is set to True
        # Check if the enriched vocabulary(glove + PAD + SOS + EOS + UNK + dataset vocabulary) already exists
        if os.path.exists(".saved/rich_embeddings.pt") and os.path.exists(".saved/rich_word2id.pt") and not reload:
            self.embeddings = torch.load(".saved/rich_embeddings.pt")
            self.word2id = torch.load(".saved/rich_word2id.pt")
            self.enriched = True
            return
        
        # Since the constructor arrived here, we need to load for the 1st time the glove word embeddings
        dataset_words = source_dataset.get_all_distinct_words_in_dataset()
        
        self.word2id = {}
        self.embeddings = torch.zeros((len(dataset_words)+4, 50))  # DIM1: Glove50 rows + 4 flavored token (PAD + SOS + EOS + UNK) | DIM2: Embedding Size 50d
        
        # Initialize the token:
        # <PAD>, <SOS>, <EOS>, <UNK>
        self.word2id["<PAD>"] = 0
        self.word2id["<SOS>"] = 1
        self.word2id["<EOS>"] = 2
        self.word2id["<UNK>"] = 3
        
        self.embeddings[self.word2id["<PAD>"]] = torch.zeros(50, dtype=torch.float32)
        self.embeddings[self.word2id["<SOS>"]] = torch.rand(50, dtype=torch.float32)
        self.embeddings[self.word2id["<EOS>"]] = torch.rand(50, dtype=torch.float32)
        self.embeddings[self.word2id["<UNK>"]] = torch.rand(50, dtype=torch.float32)
        
        counter = 4
        _glove_embeddings = {}
        
        with open('.saved/glove.6B.50d.txt', 'r', encoding='utf-8') as _vocabulary_file:
            for line in _vocabulary_file:
                line = line.strip().split()
                _glove_embeddings[line[0]] = torch.tensor([float(dimension) for dimension in line[1:]], dtype=torch.float32)
                
        for word in dataset_words:
            self.word2id[word] = counter
            counter += 1
            if word in _glove_embeddings.keys():
                self.embeddings[self.word2id[word]] = _glove_embeddings[word] 
                continue
            self.embeddings[self.word2id[word]] = torch.rand(50, dtype=torch.float32)
        
        torch.save(self.embeddings,".saved/rich_embeddings.pt")
        torch.save(self.word2id,".saved/rich_word2id.pt")
            

                
            
        torch.save(self.embeddings,".saved/base_embeddings.pt")
        torch.save(self.word2id,".saved/base_word2id.pt")
        print("break")
    
    def predefined_token_idx(self) -> dict:
        return {
            "<PAD>":0,
            "<SOS>":1,
            "<EOS>":2,
            "<UNK>":3
        }
    
    def translate(self, word_sequence : list[str]) -> torch.tensor:
        """Given a sequence of word, translate into id list according to the vocabulary.

        Args:
            word_sequence (str): [description]
        """
        # Check if the Vocabulary is enriched with all the possible word outside glove, taken from the dataset.
        # if not self.enriched:
        #     warnings.warn("The vocabulary is not enriched with dataset words that could be not in glove, pay attention to what you want to do with this representation.")
        
        # Initialize the translator
        _sequence = torch.zeros(len(word_sequence)+2, dtype=torch.int32) # +2 because of <SOS> and <EOS> token
        _sequence[0] = self.word2id["<SOS>"]
        _sequence[-1] = self.word2id["<EOS>"]
        
        counter = 1 # SKIP THE <SOS> TOKEN 
        for word in word_sequence:
            if word.lower() in self.word2id.keys():
                _sequence[counter] = self.word2id[word.lower()]
            else:
                _sequence[counter] = self.word2id["<UNK>"]
            counter += 1
        return _sequence
    
    def rev_translate(self, words_id : torch.tensor) -> list[str]:
        """Given a sequence of word, translate into id list according to the vocabulary.

        Args:
            word_sequence (str): [description]
        """
        # Check if the Vocabulary is enriched with all the possible word outside glove, taken from the dataset.
        return [list(self.word2id.keys())[idx] for idx in words_id[:].tolist()]   # word_id (1,caption_length)
    
    
    def __len__(self):
        """The total number of words in this Vocabulary."""

        return len(self.word2id.keys())
    
    
# ----------------------------------------------------------------
# Usage example

if __name__ == '__main__':
    #Load the vocabulary
    v = Vocabulary(verbose=True)
    # Make a translation
    print(v.translate(["I","like","PLay","piano","."]))
    # Enrich the vocabulary
    v.make_enrich = True
    dataset = ["I","Like","PLay","PIPPOplutopaperino"]
    v.enrich(dataset)
    v.make_enrich = False
    # Enrich the vocabulary with a bulk insert 
    v.make_enrich = True
    dataset = [["I","Like","PLay","PIPPOplutopaperino"],["I","Like","PLay","pizza"]]
    v.bulk_enrich(dataset)
    v.make_enrich = False
    
    
    
        
        
        
        
            
        
        
    
    
        
    