import os
import torch
import warnings

class Vocabulary():
    # The vocabulary implementation is done with a pre-trained word embedding GLOVE50d
    # each word is represented by a record in a dataframe with this structure
    
    
    def __init__(self, verbose: bool = False):
        
        self.enriched = False       # Tell that all the word coming from the dataset are in the vocabulary if it is set to True
        self._make_enrich = False         # Allow the user to enrich the vocabulary if it is set to True
        # Check if the enriched vocabulary(glove + PAD + SOS + EOS + UNK + dataset vocabulary) already exists
        if os.path.exists(".saved/rich_embeddings.pt") and os.path.exists(".saved/rich_word2id.pt"):
            self.embeddings = torch.load(".saved/rich_embeddings.pt")
            self.word2id = torch.load(".saved/rich_word2id.pt")
            self.enriched = True
            return
        
        # Check if the base vocabulary(glove + PAD + SOS + EOS + UNK) already exists
        if os.path.exists(".saved/base_embeddings.pt") and os.path.exists(".saved/base_word2id.pt"):
            self.embeddings = torch.load(".saved/base_embeddings.pt")
            self.word2id = torch.load(".saved/base_word2id.pt")
            return
        
        # Since the constructor arrived here, we need to load for the 1st time the glove word embeddings
        
        self.word2id = {}
        self.embeddings = torch.zeros((400004, 50)) # DIM1: Glove50 rows + 4 flavored token (PAD + SOS + EOS + UNK) | DIM2: Embedding Size 50d
        
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
        with open('.saved/glove.6B.50d.txt', 'r', encoding='utf-8') as _vocabulary_file:
            for line in _vocabulary_file:
                line = line.strip().split()
                self.word2id[line[0]] = counter
                self.embeddings[counter] = torch.tensor([float(dimension) for dimension in line[1:]], dtype=torch.float32)
                counter += 1
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
        if not self.enriched:
            warnings.warn("The vocabulary is not enriched with dataset words that could be not in glove, pay attention to what you want to do with this representation.")
        
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
    
    
    @property
    def make_enrich(self):
        return self._make_enrich
    
    @make_enrich.setter
    def make_enrich(self, value: bool):
        if not isinstance(value,bool):
            raise TypeError("The value that you want to put on make_enrich is not a boolean. Pay attention!")
        
        if value is False:
            if self.make_enrich and not self.enriched: # If before the setter call make_enrich was True, probably the vocabulary was enriched by somebody, so the vocabulary now is in the state Enriched
                self.enriched = True
            # The enriched version of the vocabulary need to be dumped in memory
            torch.save(self.embeddings,".saved/rich_embeddings.pt")
            torch.save(self.word2id,".saved/rich_word2id.pt")
            self._make_enrich = False
        else:
            self._make_enrich = value    
        
    def enrich(self, words: list[str]) -> bool:
        
        if not self.make_enrich:
            raise ValueError(f"The vocabulary is not set to be enriched, before the enrichment set the flag 'make_enrich' to True.")
        
        _new_word = []
        for word in words:
            if word.lower() in self.word2id.keys():
                continue
            _new_word.append(word)
        
        if len(_new_word) == 0:
            return False
        
        _enrichment_of_embedding = torch.zeros((len(_new_word),50))
        
        _id_carry = len(self.word2id.keys()) # The new ids start from len(.) cause the ids start from 0 and not from 1
        
        for number,word in enumerate(_new_word):
            self.word2id[word.lower()] = _id_carry
            _enrichment_of_embedding[number] = torch.rand(50, dtype=torch.float32)
            _id_carry += 1
             
        # Append the enrichment at the end of the embeddings_matrix
        self.embeddings = torch.cat([self.embeddings,_enrichment_of_embedding], dim=0)
        return True
    
    def bulk_enrich(self, sequences: list[list[str]]) -> bool:
        _words_flatten = [word for sequence in sequences for word in sequence] # flatten a list of list,  credits to wjandrea on stackoverflow <3
        return self.enrich(_words_flatten)
    
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
    
    
    
        
        
        
        
            
        
        
    
    
        
    