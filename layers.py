from torch import nn
import numpy as np
import torch

class DotProductAttention( nn.Module ):

    def __init__(self, embedding_dim, head_dim):
        super( DotProductAttention , self ).__init__()
        self.head_dims = head_dim
        self.keys = nn.Linear(embedding_dim, head_dim)
        self.queries = nn.Linear(embedding_dim, head_dim)
        self.values = nn.Linear(embedding_dim, head_dim)
        self.softmax = nn.Softmax( dim=-1 )

    def forward( self , inputs ):
        # inputs : ( B , seq_length , embedding_dim )

        K = self.keys(inputs)      # keys    : ( B , seq_length , head_dim )
        Q = self.queries( inputs ) # queries : ( B , seq_length , head_dim )
        V = self.values(inputs)    # values  : ( B , seq_length , head_dim )

        # SA : ( B , seq_length , seq_length )
        # Attention weights
        SA = self.softmax(torch.matmul( Q , K.transpose( -2 , -1 ) ) * self.head_dims ** (-0.5))

        return torch.matmul( SA , V ) # Output: ( B , seq_length , head_dim )

class MultiHeadAttention( nn.Module ):

    def __init__(self, embedding_dim, num_heads, head_dim, dropout=0.5):
        super( MultiHeadAttention , self ).__init__()
        self.heads = nn.ModuleList([DotProductAttention(embedding_dim, head_dim) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_dim, embedding_dim)
        self.dropout = nn.Dropout( dropout )

    def forward( self , inputs ):
        # inputs : ( B , seq_length , embedding_dim )
        concat_heads = torch.concat( [ self_attention( inputs ) for self_attention in self.heads ] , dim=-1 )
        # concat_heads : ( B , seq_length , num_heads * head_dim )
        output = self.dropout( self.proj( concat_heads ) )
        # output: ( B , seq_length , embedding_dim )
        return output

class FeedForward( nn.Module ):

    def __init__(self, embedding_dim, dropout_rate = 0.5):
        super( FeedForward , self ).__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim) ,
            nn.ReLU() ,
            nn.Linear(4 * embedding_dim, embedding_dim) ,
            nn.Dropout( dropout_rate )
        )

    def forward( self , inputs ):
        return self.net( inputs )

class Block( nn.Module ):

    def __init__(self, embedding_dim, num_heads):
        super( Block , self ).__init__()
        self.attention = MultiHeadAttention( embedding_dim , num_heads, head_dim=embedding_dim )
        self.feed_forward = FeedForward(embedding_dim)
        self.layer_norm_1 = nn.LayerNorm(embedding_dim)
        self.layer_norm_2 = nn.LayerNorm(embedding_dim)

    def forward( self , inputs ):
        x = inputs + self.attention( self.layer_norm_1( inputs ) )
        x = x + self.feed_forward( self.layer_norm_2( x ) )
        return x

class TokenEmbedding( nn.Module ):

    def __init__( self , vocab_size , embedding_dim ):
        super( TokenEmbedding , self ).__init__()
        self.embedding = nn.Embedding( vocab_size , embedding_dim )

    def forward( self , inputs ):
        return self.embedding( inputs )

class PositionalEncoding( nn.Module ):

    def __init__( self , seq_length , embedding_dim ):
        super( PositionalEncoding , self ).__init__()
        self.seq_length = seq_length
        self.positional_encoding = nn.Embedding( seq_length , embedding_dim )

    def forward( self , inputs ):
        # TODO: Change here to dynamically choose device
        return inputs + self.positional_encoding( torch.arange( self.seq_length , device="cuda" ) )

class Transformer( nn.Module ):

    def __init__( self , vocab_size , embedding_dim , seq_length , num_blocks , num_heads_in_block ):
        super( Transformer , self ).__init__()
        self.token_embedding = TokenEmbedding( vocab_size , embedding_dim )
        self.pos_encoding = PositionalEncoding( seq_length , embedding_dim )
        self.blocks = nn.Sequential( *[ Block( embedding_dim , num_heads_in_block ) for _ in range( num_blocks ) ] )
        self.layer_norm = nn.LayerNorm( embedding_dim )
        self.model_head = nn.Linear( embedding_dim , vocab_size )

    def forward( self , inputs ):
        # inputs : ( B , seq_length )
        token_embeddings = self.token_embedding( inputs )
        x = self.pos_encoding( token_embeddings )
        x = self.blocks( x )
        x = self.layer_norm( x )
        logits = self.model_head( x )
        return logits

if __name__ == "__main__":
    B = 32
    seq_length = 10
    embedding_dim = 1024
    head_dim = 24
    num_heads = 14
    vocab_size = 10000

    model = Transformer( vocab_size , embedding_dim , seq_length , num_blocks=5 , num_heads_in_block=num_heads )
    inputs = torch.randint( low=1 , high=vocab_size , size=( B , seq_length ) )
    outputs = model( inputs )
    print( outputs.shape )