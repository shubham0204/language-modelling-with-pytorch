import os

import fire
import torch

from config import load_global_config
from layers import Transformer
from process_data import get_tokens
from utils import Predictor
from utils import load_dict_from_pickle

config = load_global_config()
data_config = config.data
train_config = config.train
model_config = config.model

def predict( model_path ,
             num_tokens : int,
             generate : bool = False ,
             data_tensors = config.data.data_tensors_path ,
             temperature = 1.0 ,
             device="cuda" ):

    compute_device = torch.device( device )
    print( "Using device {} for inference".format( compute_device ) )
    model = Transformer(
        vocab_size=data_config.vocab_size,
        embedding_dim=model_config.embedding_dim,
        seq_length=data_config.seq_length,
        num_blocks=model_config.num_blocks,
        num_heads_in_block=model_config.num_heads_in_block,
        dropout=model_config.dropout
    )
    checkpoint = torch.load( model_path )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to( compute_device )
    model.eval()

    idx_to_word = load_dict_from_pickle( os.path.join( data_tensors , "idx_to_word.pkl" ) )
    word_to_idx = load_dict_from_pickle( os.path.join( data_tensors , "word_to_idx.pkl" ) )
    predictor = Predictor( model , idx_to_word , word_to_idx , compute_device , temperature , config.data.seq_length )
    if not generate:
        input_str = input( "Prompt: " )
        output = predictor.predict_tokens( get_tokens( input_str ) , num_tokens )
    else:
        output = predictor.generate_text( num_tokens , config.data.seq_length )
    output = " ".join( output )
    output = output.replace( "[SEP]" , "\n" )
    print( "Generated text: \n{}".format( output ) )

if __name__ == "__main__":
    fire.Fire( predict )