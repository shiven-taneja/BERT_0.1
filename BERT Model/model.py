import torch
from torch import nn
import torch.nn.functional as f
from transformers import BertConfig, PreTrainedModel


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
Module to create the model for BERT

We will create the JointEmbedding, the AttentionHead, the Encoder and the BERT Model.

Authors: Shiven Taneja & Priyanka Shakira Raj
Version: 0.1
"""

class JointEmbedding(nn.Module):
    """
    Container for the embeddings. 
    There are 3 embedding layers: 
    * Token Embedding 
        Encode word tokens with pre-trained embeddings for the different words 
    * Segment Embedding 
        Which sentence it belongs to as a vector. Input (sentence) 1 has an embedding of 0 and input (sentence) 2 has an embedding of 1
    * Position Embedding 
        Position of the word in the sentence encoded to a vector. Using a periodic funciton to encode embedding instead of learnable positional embeddings 

    Args: 
        vocab_size (int): Size of the vocabulary 
        size (int): Size of the embedding 

    Attributes: 
        size (int): Where we store size
        tok_embed (Tensor): Token Embedding 
        seg_embed (Tensor): Segment Embedding 
        norm (Tensor): Applied Layer Normalization
    """
    def __init__ (self, vocab_size, size):
        super(JointEmbedding, self).__init__()

        self.size = size

        #Token Embedding 
        self.tok_embed = nn.Embedding(vocab_size, size)
        #Segment Embedding
        self.seg_embed = nn.Embedding(vocab_size, size)

        self.norm = nn.LayerNorm(size)

    def forward(self, input_tensor):
        sentence_size = input_tensor.size(-1)
        #Get positional encoding tensor
        pos_tensor = self.attention_position(self.size, input_tensor)

        #Get segment encoding tensor
        seg_tensor = torch.zeros_like(input_tensor).to(device)
        seg_tensor[:, sentence_size // 2 + 1:] = seg_tensor[:, sentence_size // 2 + 1:].fill_(1)

        #Sum all the encoding then normalize them
        output = self.tok_embed(input_tensor) + self.seg_embed(seg_tensor) + pos_tensor
        return self.norm(output)
    
    def attention_position(self, dim, input_tensor):
        """
        Returns positional encoding 

        Args: 
            dim (int): Embedding size
            input_tensor (Tensor): Word token tensor 

        Returns: 
            pos (Tensor): Positional embedding
        """
        batch_size = input_tensor.size(0)
        sentence_size = input_tensor.size(-1)

        pos = torch.arange(sentence_size, dtype=torch.long).to(device)
        #Positional encoding by the embedding axis 
        d = torch.arange(dim, dtype=torch.long).to(device)
        d = (2 * d / dim)

        #Broadcasting 
        pos = pos.unsqueeze(1)
        pos = pos / (1e4 ** d)

        #Applying periodic function to the pos tensor
        pos[:, ::2] = torch.sin(pos[:, ::2])
        pos[:, 1::2] = torch.cos(pos[:, 1::2])

        #Extend pos on every element in batch
        pos = pos.expand(batch_size, *pos.size())
        return pos

class AttentionHead(nn.Module):
    """
    The attention module using scaled dot-product attention
    
    Args:
        input_dim (int): The input dimensions 
        q (Tensor): Query 
        k (Tensor): Key
        v (Tensor): Value
    """
    def __init__(self, input_dim, output_dim):
        super(AttentionHead, self).__init__()

        self.input_dim = input_dim

        self.q = nn.Linear(input_dim, output_dim)  
        self.k = nn.Linear(input_dim, output_dim)  
        self.v = nn.Linear(input_dim, output_dim) 

    def forward(self, input_tensor: torch.Tensor, attention_mask: torch.Tensor = None):
        """
        A function that does Scaled Dot Product

        Args: 
            input_tensor (Tensor): The tensor of the output of JointEmbedding
            attention_mask (Tensor): Attention mask vector tgar masks [PAD] tokens
        
        Returns: 
            context (Tensor): Contextualized representaion of words
        """
        #Calculate query, key and value tensors
        query, key, value = self.q(input_tensor), self.k(input_tensor), self.v(input_tensor)

        #Scaled multiplication of query and key using batched matrix multiplication
        scores = torch.bmm(query, key.transpose(1, 2)) / (query.size(1) ** 0.5)  

        #Fill elements where attention mask is True to -1e9
        scores = scores.masked_fill_(attention_mask, -1e9)  
        #Calculate attention score for the rest of the elements
        attn = f.softmax(scores, dim=-1)  
        context = torch.bmm(attn, value)  
  
        return context
    
class MultiHeadAttention(nn.Module):
    """
    Parallel AttentionHeads which retreives information from multiple representations 

    Args: 
        heads (ModuleList): A list of all the Attention Heads
        Linear (Tensor): Linear tranformation of data
        norm (Tensor): Layer normalization
    """
    def __init__(self, num_heads, dim_inp, dim_out) -> None:
        super(MultiHeadAttention, self).__init__()
        
        self.heads = nn.ModuleList(AttentionHead(dim_inp, dim_out) for _ in range(num_heads))
        self.linear = nn.Linear(dim_out * num_heads, dim_inp)
        self.norm = nn.LayerNorm(dim_inp)
    def forward(self, input_tensor: torch.Tensor, attention_mask: torch.Tensor):  
        #List of attention
        s = [head(input_tensor, attention_mask) for head in self.heads]  
        #Concatenate tensors by last axis
        scores = torch.cat(s, dim=-1)  
        #Linear transformation
        scores = self.linear(scores)  
        #Normalization
        norm_scores = self.norm(scores)
        return norm_scores
    

class Encoder(nn.Module):
    """
    An encoder layer that has a Multi-Head Attention Layer and a Feed Forward Network 

    Args: 
        attention (Tensor): Multi-head attention layer 
        norm1 (Tensor): Layer normalization 
        feed_forward (nn.Sequential): Position wise feed-forward network
        norm2 (Tensor): Layer normalization
    """
    def __init__(self, dim_inp, dim_out, attention_heads=4, dropout=0.1):
        super(Encoder, self).__init__()

        # Instantiate the multi-head attention layer
        self.attention = MultiHeadAttention(attention_heads, dim_inp, dim_out)
        # Instantiate the layer normalization for the attention layer
        self.norm1 = nn.LayerNorm(dim_inp)

         # Define the feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(dim_inp, dim_inp * 4),  # Linear layer to increase dimensionality
            nn.GELU(),  # GELU activation function
            nn.Dropout(dropout),  # Dropout layer for regularization
            nn.Linear(dim_inp * 4, dim_inp),  # Linear layer to decrease dimensionality back to original size
            nn.Dropout(dropout)  # Dropout layer for regularization
        )
        # Instantiate the layer normalization for the feed-forward layer
        self.norm2 = nn.LayerNorm(dim_inp)

    def forward(self, input_tensor: torch.Tensor, attention_mask: torch.Tensor):
        # Pass the input tensor through the multi-head attention layer
        attention_output = self.attention(input_tensor, attention_mask)
        # Apply residual connection and layer normalization
        attention_output = self.norm1(input_tensor + attention_output)

        # Pass the attention_output through the feed-forward network
        feed_forward_output = self.feed_forward(attention_output)
        # Apply residual connection and layer normalization
        return self.norm2(attention_output + feed_forward_output)


class BERT(nn.Module):
    """
    Container that combines all modules 

    Args: 
        embedding (Tensor): Embedding Layer
        encoders (List[Tensor]): List of Encoder Layers 
        token_predicition_layer (Tensor): Token predicition layer
        softmax (Tensor): Applies Softmax 
        classifiction_layer (Tensor): Classification layer
    """
    def __init__(self, vocab_size, dim_inp, dim_out, attention_heads=6, num_layers=4):
        super(BERT, self).__init__()
        print('VOCAB:', vocab_size, 'IN:', dim_inp, 'OUT:', dim_out)
        # Instantiate the joint embedding layer (token and positional embeddings)
        self.embedding = JointEmbedding(vocab_size, dim_inp)
        # Create a list of Encoder layers
        self.encoders = nn.ModuleList([
            Encoder(dim_inp, dim_out, attention_heads) for _ in range(num_layers)
        ])
        config = BertConfig(vocab_size=vocab_size, hidden_size=dim_out, num_attention_heads=attention_heads)
        self.config = config
        # Define the token prediction layer for masked language modeling
        self.token_prediction_layer = nn.Linear(dim_inp, vocab_size)
        # Define the softmax activation for token predictions
        self.softmax = nn.LogSoftmax(dim=-1)
        # Define the classification layer for next sentence prediction
        self.classification_layer = nn.Linear(dim_inp, 2)

    def forward(self, input_tensor: torch.Tensor, attention_mask: torch.Tensor):
        # Pass the input tensor through the joint embedding layer
        embedded = self.embedding(input_tensor)

        # Pass the embedded input through the encoder layers
        encoded = embedded
        for encoder in self.encoders:
            encoded = encoder(encoded, attention_mask)

        # Compute token predictions from the encoded output
        token_predictions = self.token_prediction_layer(encoded)

        # Extract the first token's representation for next sentence prediction
        first_word = encoded[:, 0, :]
        # Return the token predictions and next sentence predictions
        return self.softmax(token_predictions), self.classification_layer(first_word)