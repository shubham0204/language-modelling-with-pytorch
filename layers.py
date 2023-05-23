from torch import nn
import torch

class DotProductAttention( nn.Module ):

    def __init__(self, embedding_dim, head_dim):
        super( DotProductAttention , self ).__init__()
        self.head_dims = head_dim
        self.keys = nn.Linear(embedding_dim, head_dim)
        self.queries = nn.Linear(embedding_dim, head_dim)
        self.values = nn.Linear(embedding_dim, head_dim)
        self.softmax = nn.Softmax( dim=-1 )

    def forward( self , inputs , mask ):
        # inputs : ( B , seq_length , embedding_dim )
        # mask : ( B , seq_length , embedding_dim )

        K = self.keys(inputs)      # keys    : ( B , seq_length , head_dim )
        Q = self.queries( inputs ) # queries : ( B , seq_length , head_dim )
        V = self.values(inputs)    # values  : ( B , seq_length , head_dim )

        # SA : ( B , seq_length , seq_length )
        # Attention weights
        SA = self.softmax(torch.matmul( Q , K.transpose( -2 , -1 ) ) * self.head_dims ** (-0.5))

        # mask : ( B , 1 , seq_length )
        mask = torch.reshape( mask , shape=( SA.shape[0] , 1 , SA.shape[2] ) )
        SA = torch.masked_fill( SA , mask == 0 , -1e9 )

        return torch.matmul( SA , V ) # Output: ( B , seq_length , head_dim )

class MultiHeadAttention( nn.Module ):

    def __init__(self, embedding_dim, num_heads, head_dim, dropout=0.5):
        super( MultiHeadAttention , self ).__init__()
        self.heads = nn.ModuleList([DotProductAttention(embedding_dim, head_dim) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_dim, embedding_dim)
        self.dropout = nn.Dropout( dropout )

    def forward( self , inputs , mask ):
        # inputs : ( B , seq_length , embedding_dim )
        # mask : ( B , seq_length )
        concat_heads = torch.concat( [ self_attention( inputs , mask ) for self_attention in self.heads ] , dim=-1 )
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

class Transformer( nn.Module ):

    def __init__( self , vocab_size , embedding_dims ):
        super( Transformer , self ).__init__()
        self.token_embedding_table = nn.Embedding( vocab_size , embedding_dims )
        self.position_embedding_table = nn.Embedding( vocab_size , embedding_dims )
        self.blocks = nn.Sequential( *[ ] )

B = 32
seq_length = 10
embedding_dim = 1024
head_dim = 24
num_heads = 14

layer = Block( 1024 , 4 )
inputs = torch.randn( size=( 10 , 1024 ) )

outputs = layer( inputs )
print( outputs.shape )

layer2 = MultiHeadAttention( 1024 , num_heads , head_dim )
inputs2 = torch.randn( size=( B , seq_length , embedding_dim ) )
print( layer2( inputs2 ).shape )
print( layer2( inputs2 ).shape )