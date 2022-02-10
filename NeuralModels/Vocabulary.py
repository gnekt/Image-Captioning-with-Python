# MIT License

# Copyright (c) 2022 christiandimaio

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Typing trick for avoid circular import dependencies valid for python > 3.9
# from __future__ import annotations
# from typing import TYPE_CHECKING
# if TYPE_CHECKING:
#     from .Dataset import MyDataset
    
import torch
from typing import List
import os
import pickle

class Vocabulary():
    """
        Implementation of the vocabulary.
        
        Assumption: 
        
            1) The vocabulary is enriched with 4 special words:\n
                <PAD>: Padding ------> ID: 0\n
                <START>: Start Of String ------> ID: 1\n
                <END>: End Of String ------> ID: 2\n
                <UNK>: Out of vocabulary word ------> ID: 3\n

                Example: <START> I Love Pizza <END> <PAD> <PAD> -> Translate into ids -> 1 243 5343 645655 2 0 0 
    """
    
    
    def __init__(self, source_dataset = None): # for python > 3.9 -> def __init__(self, source_dataset: MyDataset):
        """Vocabulary constructor

        Args:
            source_dataset (MyDataset): 
                The source Dataset, if None try to load a vocabulary from the hidden .saved folder
        """
        
        if source_dataset is None:
            print("Try to load the vocabulary from file..")
            
            if not os.path.exists(".saved/word2id.pickle") or not os.path.exists(".saved/embeddings.pickle"):
                raise FileNotFoundError("You request a loading from file but the file doesn't exist, first generate the vocabulary!")
            
            with open('.saved/word2id.pickle', 'rb') as word2id, open('.saved/embeddings.pickle', 'rb') as embeddings:
                self.word2id = pickle.load(word2id)
                self.embeddings = pickle.load(embeddings)
                self.dictionary_length = len(self.word2id.keys())
                return 
        
        # Load for the 1st time all the possible words from the dataset
        dataset_words = source_dataset.get_all_distinct_words_in_dataset()
        
        # Dictionary length 
        self.dictionary_length = len(dataset_words)+4 # Dictionary word + 4 Flavored Token (PAD + START + END + UNK)
        
        self.word2id = {}
        self.embeddings = torch.zeros((self.dictionary_length, self.dictionary_length))  # DIM1: dict rows + 4 flavored token (PAD + START + END + UNK) | DIM2: Dict Rows +4 flavored token (PAD + START + END + UNK) as 1-hot
        
        # Initialize the token:
        # <PAD>, <START>, <END>, <UNK>
        self.word2id["<PAD>"] = 0
        self.word2id["<START>"] = 1
        self.word2id["<END>"] = 2
        self.word2id["<UNK>"] = 3
        
        counter = 4 
        for word in dataset_words:
            self.word2id[word] = counter
            counter += 1
        
        # Identiry matrix == 1-hot vector :)
        self.embeddings = torch.eye(self.dictionary_length)

        with open('.saved/word2id.pickle', 'wb') as word2id, open('.saved/embeddings.pickle', 'wb') as embeddings:
            pickle.dump(self.word2id, word2id, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.embeddings, embeddings, protocol=pickle.HIGHEST_PROTOCOL)
            
    def predefined_token_idx(self) -> dict:
        """Return the predefined token indexes.

        Returns:
            dict: The token dictionary
        """
        return {
            "<PAD>":0,
            "<START>":1,
            "<END>":2,
            "<UNK>":3
        }
    
    def translate(self, word_sequence : List[str], type : str = "complete") -> torch.tensor:
        """Given a sequence of word, translate into id list according to the vocabulary.

        Args:
            word_sequence (list(str)): 
                The sequence of words to translate
            
            type (str, optional): Default is complete
                The type of translation.
        
        Returns:
            (torch.Tensor): `(1,caption_length)`
                The caption in IDs form. 
                `if` complete: <1> + ...Caption... + <2> 
                `else`: <1> + ...Caption...
        """
        
        # Initialize the translator
        
        if type == "uncomplete":
            _sequence = torch.zeros(len(word_sequence)+1, dtype=torch.int32) # <START> + ...Caption...
            
        if type == "complete":
            _sequence = torch.zeros(len(word_sequence)+2, dtype=torch.int32) # <START> + ...Caption... + <END> 
            _sequence[-1] = self.word2id["<END>"]
            
        _sequence[0] = self.word2id["<START>"]
        
        counter = 1 # Always skip <START> 
        
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
            words_id (torch.Tensor): `(1,caption_length)`
                The sequence of IDs.
        Returns:
            (List(str)):
                The caption in words form.
        """
        return [list(self.word2id.keys())[idx] for idx in words_id[:].tolist()]   # word_id (1,caption_length)
    
    
    def __len__(self):
        """The total of words in this Vocabulary."""

        return len(self.word2id.keys())
    
    
    
    
    
        
        
        
        
            
        
        
    
    
        
    