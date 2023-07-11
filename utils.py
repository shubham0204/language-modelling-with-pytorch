import os
import pickle

import torch

from layers import Transformer


class Predictor:

    def __init__( self , model_path , data_tensors_path , device ):
        self.device = device
        print("Using device {} for inference".format( self.device ) )
        checkpoint = torch.load( model_path , map_location=self.device )
        config = checkpoint["config"]
        data_config = config.data
        model_config = config.model
        self.model = Transformer(
            vocab_size=data_config.vocab_size,
            embedding_dim=model_config.embedding_dim,
            seq_length=data_config.seq_length,
            num_blocks=model_config.num_blocks,
            num_heads_in_block=model_config.num_heads_in_block,
            dropout=model_config.dropout
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to( self.device )
        self.model.eval()
        self.seq_length = data_config.seq_length
        self.idx_to_word = load_dict_from_pickle(os.path.join(data_tensors_path, "idx_to_word.pkl"))
        self.word_to_idx = load_dict_from_pickle(os.path.join(data_tensors_path, "word_to_idx.pkl"))


    def predict_next_word(self, input_seq , temperature ):
        outputs = self.model( torch.unsqueeze( input_seq , dim=0 ) )
        outputs = outputs[ 0 , -1 , : ]
        outputs = torch.nn.functional.softmax( outputs / temperature , dim=-1 )
        pred_word_index = torch.multinomial( outputs , num_samples=1 )
        return self.idx_to_word[ pred_word_index.item() ]


    def predict_tokens( self , input_seq  , num_tokens , temperature=1.0 ):
        preds = []
        input_seq = [ self.word_to_idx[ word ] for word in input_seq ]
        if len(input_seq) < self.seq_length:
            diff = self.seq_length - len(input_seq)
            input_seq = [0 for _ in range(diff)] + input_seq
        else:
            input_seq = input_seq[ 0 : self.seq_length ]
        for i in range( num_tokens ):
            input_seq = torch.tensor( input_seq , device=self.device )
            predicted_token = self.predict_next_word( input_seq[ i : ] , temperature )
            preds.append( predicted_token )
            input_seq = input_seq.tolist()
            input_seq.append( self.word_to_idx[ predicted_token ] )
        return preds


    def generate_text( self , num_tokens , seq_length ):
        preds = []
        input_seq = [ 0 for _ in range( seq_length ) ]
        for i in range(num_tokens):
            input_seq = torch.tensor(input_seq, device=self.device)
            predicted_token = self.predict_next_word(input_seq[i:])
            preds.append(predicted_token)
            input_seq = input_seq.tolist()
            input_seq.append(self.word_to_idx[predicted_token])
        return preds


    @classmethod
    def beautify_output( cls , predicted_tokens ):
        text = " ".join( [ token.strip() for token in predicted_tokens ] )
        lines = text.split( "[SEP]" )
        lines = [ line.capitalize() for line in lines if len( line.strip().split() ) > 3 ]
        return ". ".join( lines )


def save_dict_as_pickle( data : dict , filename : str ):
    with open( filename , "wb" ) as file:
        pickle.dump( data , file )

def load_dict_from_pickle( filename ) -> dict:
    with open( filename , "rb" ) as file:
        return pickle.load( file )

