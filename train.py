from layers import Transformer
from utils import save_dict_as_pickle , load_dict_from_pickle
from loss import sparse_crossentropy_with_logits
from torch.utils.data import TensorDataset , DataLoader , random_split
from torch import nn
import torch
import wandb
import fire
import os
import datetime

wandb.login()

def make_data_loaders( data_tensors_path : str , test_split : float = 0.3 , batch_size : int = 128 ):
    inputs = torch.load( os.path.join( data_tensors_path , "inputs.pt" ) )
    outputs = torch.load( os.path.join( data_tensors_path , "outputs.pt" ) )
    ds = TensorDataset( inputs , outputs )
    train_ds , test_ds = random_split( ds , [ 1 - test_split , test_split ] )
    train_ds_loader = DataLoader( train_ds , batch_size , shuffle=True , drop_last=True )
    test_ds_loader = DataLoader( test_ds , batch_size , shuffle=True , drop_last=True )
    return train_ds_loader , test_ds_loader

def train_epoch( model , train_ds_loader , optimizer ):
    model.train()
    avg_loss = 0.0
    avg_acc = 0.0
    for batch_idx , ( inputs , outputs ) in enumerate( train_ds_loader ):
        optimizer.zero_grad()
        preds = model( inputs )
        preds = preds[ : , -1 , : ]
        loss = sparse_crossentropy_with_logits( preds , outputs )
        loss.backward()
        optimizer.step()
        preds = torch.argmax( torch.nn.functional.softmax( preds , dim=1 ) , dim=1 )
        acc = torch.mean( torch.eq( preds , outputs[ : , -1 ] ).float() )
        avg_loss += loss.item()
        avg_acc += acc.item()
    avg_loss /= len( train_ds_loader )
    avg_acc /= len( train_ds_loader )
    return avg_loss , avg_acc

def test_epoch( model , test_ds_loader ):
    model.eval()
    avg_loss = 0.0
    avg_acc = 0.0
    for batch_idx , ( inputs , outputs ) in enumerate( test_ds_loader ):
        preds = model( inputs )
        preds = preds[ : , -1 , : ]
        loss = sparse_crossentropy_with_logits( preds , outputs )
        preds = torch.argmax( torch.nn.functional.softmax( preds , dim=1 ) , dim=1 )
        acc = torch.mean( torch.eq( preds , outputs[ : , -1 ] ).float() )
        avg_loss += loss.item()
        avg_acc += acc.item()
    avg_loss /= len( test_ds_loader )
    avg_acc /= len( test_ds_loader )
    return avg_loss , avg_acc

model_config = {
    "vocab_size" : 5589 ,
    "embedding_dim" : 64 ,
    "seq_length" : 10 ,
    "num_blocks" : 2 ,
    "num_heads_in_block" : 3
}

def train(
        data_tensors_path ,
        num_epochs ,
        batch_size : int = 128 ,
        test_split : float = 0.3 ,
        checkpoints_path : str = None ,
        tracking_enabled = False
):
    if checkpoints_path is None:
        checkpoints_path = datetime.datetime.now().strftime( "%H_%M_%d_%m_%Y" )
    if not os.path.exists( checkpoints_path ):
        os.makedirs( checkpoints_path )

    train_ds_loader , test_ds_loader = make_data_loaders( data_tensors_path , test_split , batch_size )
    model = Transformer( **model_config )
    optimizer = torch.optim.Adam( model.parameters() , lr=0.001 )

    training_config = {
        "batch_size" : batch_size ,
        "num_epochs" : num_epochs
    }

    if tracking_enabled:
        wandb.init(
            project="Poem_Maker_Transformer",
            config=model_config.update(training_config)
        )

    for e in range( num_epochs ):
        print( f"--------- EPOCH {e + 1} -----------" )
        train_loss , train_acc = train_epoch( model , train_ds_loader , optimizer )
        val_loss , val_acc = test_epoch( model , test_ds_loader )
        if tracking_enabled:
            wandb.log( {
                    "loss" : train_loss ,
                    "acc" : train_acc ,
                    "val_loss" : val_loss ,
                    "val_acc" : val_acc
                } )
        torch.save( model , "model_{}.pt".format( e + 1 ) )
        print("{} loss={:.5f}, acc={:.5f} , val_loss={:.5f}, val_acc={:.5f}"
              .format(e + 1 , train_loss , train_acc , val_loss , val_acc ) )


if __name__ == "__main__":
    fire.Fire( train )

