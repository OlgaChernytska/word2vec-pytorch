import torch.nn as nn

from utils.constants import EMBED_DIMENSION


class CBOW_Model(nn.Module):
    def __init__(self, vocab_size: int):
        super(CBOW_Model, self).__init__()
        self.linear1 = nn.Linear(in_features=vocab_size, out_features=EMBED_DIMENSION)
        self.linear2 = nn.Linear(in_features=EMBED_DIMENSION, out_features=vocab_size)
        #self.sigmoid = nn.Sigmoid() #

    def forward(self, inputs_):
        x = self.linear1(inputs_)
        #x = self.sigmoid(x) #
        x = x.mean(axis=1)
        x = self.linear2(x)
        return x
    
    
class SkipGram_Model(nn.Module):
    def __init__(self, vocab_size: int):
        super(SkipGram_Model, self).__init__()
        self.linear1 = nn.Linear(in_features=vocab_size, out_features=EMBED_DIMENSION)
        self.linear2 = nn.Linear(in_features=EMBED_DIMENSION, out_features=vocab_size)
        #self.sigmoid = nn.Sigmoid() #

    def forward(self, inputs_):
        x = self.linear1(inputs_)
        #x = self.sigmoid(x) #
        x = self.linear2(x)
        return x
