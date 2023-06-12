import os

import fire
import torch

from config import load_global_config
from process_data import get_tokens
from utils import Predictor
from utils import load_dict_from_pickle

config = load_global_config()

def predict( model_path ,
             num_tokens : int,
             generate : bool = False ,
             data_tensors = config.data.data_tensors_path ,
             temperature = 1.0 ,
             device="cuda" ):
    compute_device = torch.device( device )
    model = torch.load( model_path , map_location=compute_device )
    idx_to_word = load_dict_from_pickle( os.path.join( data_tensors , "idx_to_word.pkl" ) )
    word_to_idx = load_dict_from_pickle( os.path.join( data_tensors , "word_to_idx.pkl" ) )
    predictor = Predictor( model , idx_to_word , word_to_idx , compute_device , temperature )
    if not generate:
        input_str = input( "Enter some text: " )
        output = predictor.predict_tokens( get_tokens( input_str ) , num_tokens )
    else:
        output = predictor.generate_text( num_tokens , config.data.seq_length )
    for i in range( len( output ) ):
        if output[ i ] == "[SEP]":
            output[ i ] = "\n"
    print( " ".join( output ) )

if __name__ == "__main__":
    fire.Fire( predict )