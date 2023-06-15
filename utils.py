import pickle

import torch


class Predictor:

    def __init__( self , model , idx_to_word , word_to_idx , device , temperature ):
        self.model = model
        self.idx_to_word = idx_to_word
        self.word_to_idx = word_to_idx
        self.device = device
        self.temperature = temperature
        self.model.to( self.device )

    @torch.no_grad()
    def predict_next_word(self, input_seq):
        outputs = self.model( torch.unsqueeze( input_seq , dim=0 ) )
        outputs = outputs[ 0 , -1 , : ]
        outputs = torch.nn.functional.softmax( outputs / self.temperature , dim=-1 )
        pred_word_index = torch.multinomial( outputs , num_samples=1 )
        return self.idx_to_word[ pred_word_index.item() ]

    def predict_tokens( self , input_seq  , num_tokens ):
        preds = []
        input_seq = [ self.word_to_idx[ word ] for word in input_seq ]
        if len(input_seq) < 128:
            diff = 128 - len(input_seq)
            input_seq = [0 for _ in range(diff)] + input_seq
        for i in range( num_tokens ):
            input_seq = torch.tensor( input_seq , device=self.device )
            predicted_token = self.predict_next_word( input_seq[ i : ] )
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


def save_dict_as_pickle( data : dict , filename : str ):
    with open( filename , "wb" ) as file:
        pickle.dump( data , file )

def load_dict_from_pickle( filename ) -> dict:
    with open( filename , "rb" ) as file:
        return pickle.load( file )

