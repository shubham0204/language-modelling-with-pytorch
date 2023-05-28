from layers import Transformer
from utils import save_dict_as_pickle , load_dict_from_pickle
from torch.utils.data import TensorDataset , DataLoader , random_split
from torch import nn
import torch
import os

def make_data_loaders( data_tensors_path : str , test_split : float = 0.3 , batch_size : int = 128 ):
    inputs = torch.load( os.path.join( data_tensors_path , "inputs.pt" ) )
    outputs = torch.load( os.path.join( data_tensors_path , "outputs.pt" ) )
    ds = TensorDataset( inputs , outputs )
    train_ds , test_ds = random_split( ds , [ 1 - test_split , test_split ] )
    train_ds_loader = DataLoader( train_ds , batch_size , shuffle=True , drop_last=True )
    test_ds_loader = DataLoader( test_ds , batch_size , shuffle=True , drop_last=True )
    return train_ds_loader , test_ds_loader
