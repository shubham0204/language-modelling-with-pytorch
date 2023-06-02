from utils import save_dict_as_pickle
import fire
import torch
import os
import re

def filter_token( text_token : str ):
    return re.sub(r"\W+", "", text_token)

def get_tokens( poem_text : str ):
    lines = [line.lower() for line in poem_text.split("\n")]
    lines = [line for line in lines if len(line.strip()) != 0]
    tokens = []
    for line in lines:
        tokens += line.split()
    tokens = [token for token in tokens if len(token.strip()) != 0]
    tokens = [filter_token(token) for token in tokens]
    return tokens

def pad_sequence( sequence , max_length ):
    if len( sequence ) < max_length:
        return sequence + [ 0 for _ in range( max_length - len( sequence ) ) ]
    else:
        return sequence[ 0 : max_length ]

def make_sequences( sequence , input_length ):
    sequences = []
    for i in range( len( sequence ) - input_length - 1 ):
        sequences.append( [sequence[i: i + input_length] , sequence[i + input_length]] )
    return sequences

def process_data( data_dir : str , output_dir : str ):
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
    vocab = list( set( vocab ) )
    
    index_to_word = dict( zip( range( len(vocab ) ) , vocab ) )
    word_to_index = dict( zip( vocab , range( len(vocab) ) ) )
    save_dict_as_pickle( index_to_word , os.path.join( output_dir , "idx_to_word.pkl" ) )
    save_dict_as_pickle( word_to_index , os.path.join( output_dir , "word_to_idx.pkl" ) )
    print( "Vocab Size:" , len( word_to_index ) )

    idx_tokenized_sentences = [ [ word_to_index[ word ] for word in sentence ] for sentence in tokenized_sentences ]
    n_gram_sequences = []
    for i in range( len(idx_tokenized_sentences) ):
        n_gram_sequences += make_sequences(idx_tokenized_sentences[i], input_length=10)

    inputs = torch.tensor( [ sequence[0] for sequence in n_gram_sequences ] )
    outputs = torch.tensor( [ sequence[1] for sequence in n_gram_sequences ] )
    outputs = torch.unsqueeze( outputs , 1 )
    torch.save( inputs , os.path.join( output_dir , "inputs.pt" ) )
    torch.save( outputs , os.path.join( output_dir , "outputs.pt" ) )


if __name__ == "__main__":
    fire.Fire( process_data )