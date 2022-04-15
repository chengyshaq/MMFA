import torch.nn as nn
from model.dot_attention import ScaledDotProductAttention
class MultiHeadAttention(nn.Module):

    def __init__(self, model_dim, num_heads, dropout=0.0):
        super(MultiHeadAttention, self).__init__()

        self.dim_per_head = model_dim // num_heads
        self.num_heads = num_heads
        self.linear_k = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_v = nn.Linear((num_heads + 1) * model_dim, (num_heads + 1) * model_dim)
        self.linear_q = nn.Linear(model_dim, self.dim_per_head * num_heads)

        self.dot_product_attention = ScaledDotProductAttention(dropout)
        self.linear_final = nn.Linear((num_heads + 1) * model_dim, (num_heads + 1) * model_dim)
        self.dropout = nn.Dropout(dropout)
		# add layer norm
        self.layer_norm = nn.LayerNorm((num_heads + 1) * model_dim, (num_heads + 1) * model_dim)

    def forward(self, key, value, query, attn_mask=None):

        residual = value
        num_views = self.num_heads
        batch_size = key.size(0)
        # linear projection
        key = self.linear_k(key)
        value = self.linear_v(value)
        query = self.linear_q(query)

        # # split by heads
        key = key.view(batch_size, -1)
        value = value.view(batch_size, -1)
        query = query.view(batch_size, -1)

        if attn_mask:
            attn_mask = attn_mask.repeat(num_views, 1, 1)
        # scaled dot product attention
        scale = (key.size(-1) // num_views) ** -0.5
        attention = self.dot_product_attention(query, key, value, scale, attn_mask)
        # final linear projection
        output = self.linear_final(attention)
        # dropout
        output = self.dropout(output)
        # add residual and norm layer
        output = self.layer_norm(residual + output)

        return output