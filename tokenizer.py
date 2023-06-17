from typing import List


class Tokenizer:
    ''' Class for character-wise tokenization'''

    def __init__(self, text: str):
        assert len(text) > 0, "Text used for creating tokenizer cannot be empty"

        self.text = text
        self.symbols = sorted(list(set(self.text)))
        self.vocab_size = len(self.symbols)
        self.stoi = { ch:i for i, ch in enumerate(self.symbols)}
        self.itos = { i:ch for i, ch in enumerate(self.symbols)}

    def encode(self, text: str) -> List[int]:
        ''' Encodes string to list of ints '''

        return [self.stoi[ch] for ch in text]
    
    def decode(self, tokens: List[int]) -> str:
        ''' Decodes list of ints to string '''
        
        return ''.join([self.itos[token] for token in tokens])
