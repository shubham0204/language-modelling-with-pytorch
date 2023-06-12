import os
import random
import re

import torch

from config import load_global_config, save_global_config
from utils import save_dict_as_pickle

config = load_global_config()
data_config = config.data
"""
TODO:
1. Use SGD
2. Increase context length
3. Change dataset
4. Add free text generation in predict.py
"""
token_linebreak = "[SEP]"
token_seq_start = "[START]"
token_seq_end = "[END]"

def filter_token( text_token : str ):
    return re.sub(r"\W+", "", text_token)

def get_tokens( poem_text : str ):
    poem_text = poem_text.lower()
    poem_text = poem_text.replace( "\n" , " " + token_linebreak + " " )
    tokens = poem_text.split()
    tokens = [token for token in tokens if len(token.strip()) != 0]
    return [ token_seq_start ] + tokens + [ token_seq_end ]

def pad_sequence( sequence , max_length ):
    if len( sequence ) < max_length:
        return sequence + [ 0 for _ in range( max_length - len( sequence ) ) ]
    else:
        return sequence[ 0 : max_length ]

def make_sequences( sequence , input_length ):
    sequences = []
    for i in range( len( sequence ) - input_length - 1 ):
        sequences.append( [sequence[i: i + input_length] , sequence[i + 1: i + input_length + 1] ] )
    return sequences

if __name__ == "__main__":
    data_dir = data_config.data_path
    output_dir = data_config.data_tensors_path
    txt_files = os.listdir( data_dir )

    if not os.path.exists( output_dir ):
        os.mkdir( output_dir )

    vocab = []
    tokenized_sentences = []
    for file in txt_files:
        with open( os.path.join( data_dir , file ) , "r" , encoding="utf-8" ) as file_buffer:
            tokens = get_tokens( file_buffer.read() )
            vocab += tokens
            tokenized_sentences.append( tokens )
    print( f"{len(tokenized_sentences)} sentences read." )
    vocab = list( set( vocab ) )

    index_to_word = dict( zip( range( len(vocab ) ) , vocab ) )
    word_to_index = dict( zip( vocab , range( len(vocab) ) ) )
    save_dict_as_pickle( index_to_word , os.path.join( output_dir , "idx_to_word.pkl" ) )
    save_dict_as_pickle( word_to_index , os.path.join( output_dir , "word_to_idx.pkl" ) )
    config.data.vocab_size = len( index_to_word )
    print( f"{config.data.vocab_size} words in vocabulary" )
    save_global_config( config )

    idx_tokenized_sentences = [ [ word_to_index[ word ] for word in sentence ] for sentence in tokenized_sentences ]
    n_gram_sequences = []
    for i in range( len(idx_tokenized_sentences) ):
        n_gram_sequences += make_sequences(idx_tokenized_sentences[i], input_length=data_config.seq_length)
    print( f"{len(n_gram_sequences)} sequences produced" )

    split = data_config.test_split
    test_size = int( split * len(n_gram_sequences) )
    train_size = len( n_gram_sequences ) - test_size
    print( f"Number of training samples are {train_size} and validation samples are {test_size}" )

    random.shuffle( n_gram_sequences )
    inputs = torch.tensor( [ sequence[0] for sequence in n_gram_sequences ] )
    outputs = torch.tensor( [ sequence[1] for sequence in n_gram_sequences ] )
    outputs = torch.unsqueeze( outputs , 1 )
    torch.save( inputs , os.path.join( output_dir , "inputs.pt" ) )
    torch.save( outputs , os.path.join( output_dir , "outputs.pt" ) )
