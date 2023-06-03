from utils import Predictor
from utils import load_dict_from_pickle
from config import load_global_config
from process_data import get_tokens
import torch
import os
import fire

config = load_global_config()

def predict( model_path , data_tensors = config.data.data_tensors_path ):
    model = torch.load( model_path )
    idx_to_word = load_dict_from_pickle( os.path.join( data_tensors , "idx_to_word.pkl" ) )
    word_to_idx = load_dict_from_pickle( os.path.join( data_tensors , "word_to_idx.pkl" ) )
    predictor = Predictor( model , idx_to_word , word_to_idx )
    input_str = input( "Enter some text: " )
    output = predictor.predict_tokens( get_tokens( input_str ) , 5 )
    print( " ".join( output ) )

if __name__ == "__main__":
    fire.Fire( predict )