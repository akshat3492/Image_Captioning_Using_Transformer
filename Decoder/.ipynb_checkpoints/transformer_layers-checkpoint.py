import torch
import torch.nn as nn
from torch.nn import functional as F
import math

class PositionalEncoding(nn.Module):
    """
    Adds positional encodings to the input embeddings to provide positional information to the model.
    The encodings are computed using sine and cosine functions.
    """
    def __init__(self, embed_dim, dropout=0.3, max_len=5000):
        """
        Initialize the PositionalEncoding layer.

        Arguments:
            embed_dim (int): Dimension of the embedding space.
            dropout (float): Dropout rate to apply after adding positional encodings.
            max_len (int): Maximum length of the input sequences.
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create a positional encoding matrix with max_len rows and embed_dim columns
        pe = torch.zeros(1, max_len, embed_dim)

        # Compute positional encodings using sine and cosine functions
        update_tensor = -torch.tensor([i for i in range(0, embed_dim, 2)], dtype=torch.float32)/embed_dim
        update_tensor = torch.pow(10000, update_tensor)[None, :]

        for i in range(max_len):
          pe[:, i, 0::2] = torch.sin(i * update_tensor)
          pe[:, i, 1::2] = torch.cos(i * update_tensor)

        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Add positional encodings to the input embeddings.

        Arguments:
            x (torch.Tensor): Input tensor of shape (N, S, D), where N is batch size,
                              S is sequence length, and D is embedding dimension.

        Returns:
            torch.Tensor: Tensor of the same shape as input with positional encodings added.
        """
        N, S, D = x.shape
        
        x = x + self.pe[:, :S, :]
        x = self.dropout(x)
        
        return x


class MultiHeadAttention(nn.Module):
    """
    Implements multi-head attention mechanism.
    """

    def __init__(self, embed_dim, num_heads, dropout=0.4):
        """
        Initialize the MultiHeadAttention layer.

        Arguments:
            embed_dim (int): Dimension of the token embeddings.
            num_heads (int): Number of attention heads.
            dropout (float): Dropout probability for attention weights.
        """
        super().__init__()
        assert embed_dim % num_heads == 0 #Embedding dimension must be divisible by number of heads.

        self.n_head = num_heads
        self.emd_dim = embed_dim
        self.head_dim = self.emd_dim // self.n_head

        # Define linear layers for query, key, value, and output projection
        self.key = nn.Linear(embed_dim, embed_dim)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        
        self.attn_drop = nn.Dropout(dropout)

        

    def forward(self, query, key, value, attn_mask=None):
        """
        Perform multi-head attention computation.

        Arguments:
            query (torch.Tensor): Query tensor of shape (N, S, E).
            key (torch.Tensor): Key tensor of shape (N, T, E).
            value (torch.Tensor): Value tensor of shape (N, T, E).
            attn_mask (torch.Tensor, optional): Mask tensor of shape (S, T) to prevent attention to certain positions.

        Returns:
            torch.Tensor: Output tensor of shape (N, S, E) after applying attention.
        """
        N, S, E = query.shape
        N, T, E = value.shape

        # Linear transformations and split into multiple heads
        #N, S, H, E/H
        query_output = self.query(query).view(N, S, self.n_head, E//self.n_head)
        #N, T, H, E/H
        key_output = self.key(key).view(N, T, self.n_head, E//self.n_head)
        #N, T, H, E/H
        value_output = self.value(value).view(N, T, self.n_head, E//self.n_head)

        query_output = torch.transpose(query_output, 1, 2)#N, H, S, E/H
        key_output = torch.transpose(key_output, 1, 2)#N, H, T, E/H
        value_output = torch.transpose(value_output, 1, 2)#N, H, T, E/H

        # Scaled dot-product attention
        #numerator will be shape of N, H, S, T
        attention_weights = torch.matmul(query_output, torch.transpose(key_output,2,3))
        attention_weights = attention_weights/(E // self.n_head) ** 0.5

        #replace upper triangular matrix with negative infinity
        if attn_mask is not None:
          attention_weights = attention_weights.masked_fill(attn_mask == 0, float('-inf'))# Apply mask to attention scores

        #apply softmax row wise for every batch N, every multi-head H, every query sequence
        attention_weights = torch.softmax(attention_weights, dim=-1)

        #apply dropout N, H, S, T
        attention_weights = self.attn_drop(attention_weights)

        #multiply with value matrix N, H, S, E/H
        attention_weights = torch.matmul(attention_weights, value_output)

        #reshape back to N, S, E
        attention_weights = attention_weights.transpose(1, 2).contiguous().view(N, S, E)

        # Final linear projection
        output = self.proj(attention_weights)

        return output


