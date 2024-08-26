import numpy as np
import copy

import torch
import torch.nn as nn

from Decoder.transformer_layers import *


class Transformer(nn.Module):
    """
    A Transformer-based model for generating captions from image features.
    
    This model uses a Transformer decoder to generate captions, given input image features.
    """
    def __init__(self, word_to_idx, input_dim, wordvec_dim, num_heads=4,
                 num_layers=4, max_length=17):
        """
        Initialize the CaptioningTransformer.

        Arguments:
        - word_to_idx (dict): Vocabulary mapping words to unique indices.
        - input_dim (int): Dimension of input image feature vectors.
        - wordvec_dim (int): Dimension of word embeddings.
        - num_heads (int): Number of attention heads.
        - num_layers (int): Number of layers in the Transformer decoder.
        - max_length (int): Maximum length of the generated caption sequences.
        """
        super().__init__()

        vocab_size = len(word_to_idx)
        self.vocab_size = vocab_size
        self._null = word_to_idx["<TAB>"]
        self._start = word_to_idx.get("<START>", None)
        self._end = word_to_idx.get("<END>", None)

        # Define layers for projecting image features, embedding words, and applying positional encoding
        self.visual_projection = nn.Linear(input_dim, wordvec_dim)
        self.embedding = nn.Embedding(vocab_size, wordvec_dim, padding_idx=self._null)
        self.positional_encoding = PositionalEncoding(wordvec_dim, max_len=max_length)

        # Define the Transformer decoder
        decoder_layer = DecoderLayers(input_dim=wordvec_dim, num_heads=num_heads)
        self.transformer = Decoder(decoder_layer, num_layers=num_layers)
        self.apply(self._init_weights)

        # Output layer to project decoder outputs to vocabulary size
        self.output = nn.Linear(wordvec_dim, vocab_size)

    def _init_weights(self, module):
        """
        Initialize weights of the model layers.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, features, captions):
        """
        Forward pass of the model. Generates scores over the vocabulary for each token in the captions.

        Arguments:
         - features (torch.Tensor): Image features of shape (N, D).
         - captions (torch.Tensor): Ground truth captions of shape (N, T).

        Returns:
         - scores (torch.Tensor): Scores for each token at each timestep, of shape (N, T, V).
        """
        N, T = captions.shape

        # Ensure all inputs are on the same device
        device = features.device
        captions = captions.to(device)

        # Project image features and embed the captions
        embed_caption = self.embedding(captions) #N, T, W
        embed_caption_output = self.positional_encoding(embed_caption) #N,T,W
        image_feature_output = self.visual_projection(features).unsqueeze(1) # (N, 1, W)
        
        # Create a mask to prevent attention to future positions
        tgt_mask = torch.tril(torch.ones(T, T, device=device))

        # Pass through Transformer decoder
        transfomer_output = self.transformer(embed_caption_output, image_feature_output, tgt_mask)
        
        # Project to vocabulary size to get scores
        scores = self.output(transfomer_output)
        return scores

class DecoderLayers(nn.Module):
    """
    A single layer of a Transformer decoder, using multi-head self-attention and a feedforward network.
    """
    def __init__(self, input_dim, num_heads, dim_feedforward=2048, dropout=0.5):
        """
        Initialize the TransformerDecoderLayer.

        Arguments:
         - input_dim (int): Number of expected features in the input.
         - num_heads (int): Number of attention heads.
         - dim_feedforward (int): Dimension of the feedforward network.
         - dropout (float): Dropout rate.
        """
        super().__init__()
        self.self_attn = MultiHeadAttention(input_dim, num_heads, dropout)
        self.multihead_attn = MultiHeadAttention(input_dim, num_heads, dropout)
        self.linear1 = nn.Linear(input_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, input_dim)

        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.norm3 = nn.LayerNorm(input_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = nn.ReLU()


    def forward(self, tgt, memory, tgt_mask=None):
        """
        Forward pass through the decoder layer.

        Arguments:
        - tgt (torch.Tensor): Target sequence of shape (N, T, W).
        - memory (torch.Tensor): Output of the encoder (image features), of shape (N, S, D).
        - tgt_mask (torch.Tensor, optional): Mask for the target sequence.

        Returns:
        - out (torch.Tensor): Output features, of shape (N, T, W).
        """
        
        # Self-attention and residual connection
        tgt2 = self.self_attn(query=tgt, key=tgt, value=tgt, attn_mask=tgt_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # Cross-attention with encoder output and residual connection
        tgt2 = self.multihead_attn(query=tgt, key=memory, value=memory)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # Feedforward network and residual connection
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

def clones(module, N):
    """
    Create N identical layers.

    Arguments:
    - module (nn.Module): The module to clone.
    - N (int): The number of copies to create.

    Returns:
    - nn.ModuleList: A list containing N identical copies of the module.
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Decoder(nn.Module):
    """
    A Transformer decoder consisting of multiple layers of DecoderLayers.
    """
    def __init__(self, decoder_layer, num_layers):
        """
        Initialize the TransformerDecoder.

        Arguments:
        - decoder_layer (nn.Module): The TransformerDecoderLayer to use.
        - num_layers (int): Number of layers in the decoder.
        """
        
        super().__init__()
        self.layers = clones(decoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, tgt, memory, tgt_mask=None):
        """
        Forward pass through the decoder.

        Arguments:
        - tgt (torch.Tensor): Target sequence of shape (N, T, W).
        - memory (torch.Tensor): Output of the encoder (image features), of shape (N, S, D).
        - tgt_mask (torch.Tensor, optional): Mask for the target sequence.

        Returns:
        - output (torch.Tensor): Output features, of shape (N, T, W).
        """
        
        output = tgt

        for mod in self.layers:
            output = mod(output, memory, tgt_mask=tgt_mask)

        return output
