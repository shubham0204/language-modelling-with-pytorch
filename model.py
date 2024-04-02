import math

import torch
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DotProductAttention(nn.Module):
    def __init__(self, embedding_dim, head_dim, dropout):
        super(DotProductAttention, self).__init__()
        self.head_dims = head_dim
        self.keys = nn.Linear(embedding_dim, head_dim, bias=False)
        self.queries = nn.Linear(embedding_dim, head_dim, bias=False)
        self.values = nn.Linear(embedding_dim, head_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)
        self.register_buffer("tril", torch.tril(torch.ones(head_dim, head_dim)))

    def forward(self, inputs):
        # inputs : ( B , seq_length , embedding_dim )
        T = inputs.shape[1]

        K = self.keys(inputs)  # keys    : ( B , seq_length , head_dim )
        Q = self.queries(inputs)  # queries : ( B , seq_length , head_dim )
        V = self.values(inputs)  # values  : ( B , seq_length , head_dim )

        # SA : ( B , seq_length , seq_length )
        # Attention weights
        SA = torch.matmul(Q, K.transpose(-2, -1)) * self.head_dims ** (-0.5)
        SA = SA.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        SA = self.softmax(SA)
        SA = self.dropout(SA)

        return torch.matmul(SA, V)  # Output: ( B , seq_length , head_dim )


class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads, head_dim, dropout):
        super(MultiHeadAttention, self).__init__()
        self.heads = nn.ModuleList(
            [
                DotProductAttention(embedding_dim, head_dim, dropout)
                for _ in range(num_heads)
            ]
        )
        self.proj = nn.Linear(num_heads * head_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        # inputs : ( B , seq_length , embedding_dim )
        concat_heads = torch.concat(
            [self_attention(inputs) for self_attention in self.heads], dim=-1
        )
        # concat_heads : ( B , seq_length , num_heads * head_dim )
        output = self.dropout(self.proj(concat_heads))
        # output: ( B , seq_length , embedding_dim )
        return output


class FeedForward(nn.Module):
    def __init__(self, embedding_dim, dropout_rate=0.2):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            nn.ReLU(),
            nn.Linear(4 * embedding_dim, embedding_dim),
            nn.Dropout(dropout_rate),
        )

    def forward(self, inputs):
        return self.net(inputs)


class Block(nn.Module):
    def __init__(self, embedding_dim, num_heads, dropout):
        super(Block, self).__init__()
        head_dim = embedding_dim // num_heads
        self.attention = MultiHeadAttention(embedding_dim, num_heads, head_dim, dropout)
        self.feed_forward = FeedForward(embedding_dim)
        self.layer_norm_1 = nn.LayerNorm(embedding_dim)
        self.layer_norm_2 = nn.LayerNorm(embedding_dim)

    def forward(self, inputs):
        x = inputs + self.attention(self.layer_norm_1(inputs))
        x = x + self.feed_forward(self.layer_norm_2(x))
        return x


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, inputs):
        return self.embedding(inputs)


class PositionalEncoding(nn.Module):
    def __init__(self, seq_length, embedding_dim, dropout):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.embedding_dim = embedding_dim
        position = torch.arange(seq_length).unsqueeze(1)
        div_term = torch.pow(10000.0, torch.arange(0, embedding_dim, 2) / embedding_dim)
        positional_embeddings = torch.zeros((seq_length, embedding_dim))
        positional_embeddings[:, 0::2] = torch.sin(position / div_term)
        positional_embeddings[:, 1::2] = torch.cos(position / div_term)
        self.register_buffer("pe", positional_embeddings)

    def forward(self, inputs):
        x = inputs * math.sqrt(self.embedding_dim)
        x = x + self.pe
        return self.dropout(x)


class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        seq_length,
        num_blocks,
        num_heads_in_block,
        dropout,
    ):
        super(Transformer, self).__init__()
        self.token_embedding = TokenEmbedding(vocab_size, embedding_dim)
        self.pos_encoding = PositionalEncoding(seq_length, embedding_dim, dropout)
        self.blocks = nn.Sequential(
            *[
                Block(embedding_dim, num_heads_in_block, dropout)
                for _ in range(num_blocks)
            ]
        )
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.model_head = nn.Linear(embedding_dim, vocab_size)
        self.apply(self._init_weight)

    def _init_weight(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, 0.0, 0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, 0.0, 0.02)

    def forward(self, inputs):
        # inputs : ( B , seq_length )
        token_embeddings = self.token_embedding(inputs)
        x = self.pos_encoding(token_embeddings)
        x = self.blocks(x)
        x = self.layer_norm(x)
        logits = self.model_head(x)
        return logits


if __name__ == "__main__":
    B = 32
    seq_length = 10
    embedding_dim = 1024
    head_dim = 24
    num_heads = 14
    vocab_size = 10000

    model = Transformer(
        vocab_size,
        embedding_dim,
        seq_length,
        num_blocks=5,
        num_heads_in_block=num_heads,
        dropout=0.1,
    )
    inputs = torch.randint(low=1, high=vocab_size, size=(B, seq_length))
    outputs = model(inputs)
    print(outputs.shape)

    enc = PositionalEncoding(seq_length, embedding_dim, 0.1)
    emb = torch.randn((B, seq_length, embedding_dim))
    print(enc(emb).shape)
