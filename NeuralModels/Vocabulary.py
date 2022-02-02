# Typing trick for avoid circular import dependencies
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .Dataset import MyDataset
    
import os
import torch
from typing import List

class Vocabulary():
    """
        Implementation of the vocabulary.
        
        Assumption: 
        
            1) The vocabulary is enriched with 4 special words:\n
                <PAD>: Padding ------> ID: 0\n
                <SOS>: Start Of String ------> ID: 1\n
                <EOS>: End Of String ------> ID: 2\n
                <UNK>: Out of vocabulary word ------> ID: 3\n

                Example: <SOS> I Love Pizza <EOS> <PAD> <PAD> -> Translate into ids -> 1 243 5343 645655 2 0 0 
    """
    
    
    def __init__(self, source_dataset: MyDataset):
        """[summary]

        Args:
            source_dataset (MyDataset): [description]
        """
        
        self.ready = False       # Tell that all the word coming from the dataset are in the vocabulary if it is set to True
        
        # Since the constructor arrived here, we need to load for the 1st time all the possible words from the dataset
        dataset_words = source_dataset.get_all_distinct_words_in_dataset()
        
        # Dictionary length 
        self.dictionary_length = len(dataset_words)+4 # Dictionary word + 4 Flavored Token (PAD + SOS + EOS + UNK)
        
        self.word2id = {}
        self.embeddings = torch.zeros((self.dictionary_length, self.dictionary_length))  # DIM1: dict rows + 4 flavored token (PAD + SOS + EOS + UNK) | DIM2: Dict Rows +4 flavored token (PAD + SOS + EOS + UNK) as 1-hot
        
        # Initialize the token:
        # <PAD>, <SOS>, <EOS>, <UNK>
        self.word2id["<PAD>"] = 0
        self.word2id["<SOS>"] = 1
        self.word2id["<EOS>"] = 2
        self.word2id["<UNK>"] = 3
        
        counter = 4 
        for word in dataset_words:
            self.word2id[word] = counter
            counter += 1
            
        self.embeddings = torch.eye(self.dictionary_length)
    
    def predefined_token_idx(self) -> dict:
        return {
            "<PAD>":0,
            "<SOS>":1,
            "<EOS>":2,
            "<UNK>":3
        }
    
    def translate(self, word_sequence : List[str], type : str = "complete") -> torch.tensor:
        """Given a sequence of word, translate into id list according to the vocabulary.

        Args:
            word_sequence (str): [description]
        """
        
        # Initialize the translator
        
        if type == "uncomplete":
            _sequence = torch.zeros(len(word_sequence)+1, dtype=torch.int32) # <SOS> + ...Caption...
            
        if type == "complete":
            _sequence = torch.zeros(len(word_sequence)+2, dtype=torch.int32) # <SOS> + ...Caption... + <EOS> 
            _sequence[-1] = self.word2id["<EOS>"]
            
        _sequence[0] = self.word2id["<SOS>"]
        
        counter = 1 # Always skip <SOS> 
        
        # Evaluate all the word into the caption and translate it to an embeddings
        for word in word_sequence:
            if word.lower() in self.word2id.keys():
                _sequence[counter] = self.word2id[word.lower()]
            else:
                _sequence[counter] = self.word2id["<UNK>"]
            counter += 1
        
        return _sequence
    
    def rev_translate(self, words_id : torch.tensor) -> List[str]:
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
    
    
    
        
        
        
        
            
        
        
    
    
        
    