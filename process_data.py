from utils import save_dict_as_pickle
import fire
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

def process_data( data_dir : str ):
    txt_files = os.listdir( data_dir )

    vocab = []
    for file in txt_files:
        with open( os.path.join( data_dir , file ) , "r" , encoding="utf-8" ) as file_buffer:
            vocab += get_tokens( file_buffer.read() )
    vocab = list( set( vocab ) )
    
    index_to_word = dict( zip( range( len(vocab ) ) , vocab ) )
    word_to_index = dict( zip( vocab , range( len(vocab) ) ) )
    save_dict_as_pickle( index_to_word , "idx_to_word.pkl" )
    save_dict_as_pickle( word_to_index , "word_to_idx.pkl" )

if __name__ == "__main__":
    fire.Fire( process_data )