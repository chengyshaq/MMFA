import torch
import torch.nn as nn
import numpy as np


class ScaledDotProductAttention(nn.Module):
    def __init__(self, attention_dropout=0.0):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, q, k, v, scale=None, attn_mask=None):

        similarity = torch.mm(q, k.transpose(0, 1))
        if scale:
        	similarity = similarity * scale
        if attn_mask:      	
        	similarity = similarity.masked_fill_(attn_mask, -np.inf)
        similarity = self.softmax(similarity)
		#add dropout
        similarity = self.dropout(similarity)

        attention = torch.mm(similarity, v)
        return attention