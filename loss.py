import torch

def sparse_crossentropy_with_logits( logits , labels ):
    log_softmax_preds = torch.log2( torch.nn.functional.softmax( logits , dim=-1 ) )
    return torch.take( -log_softmax_preds , labels ).mean()

if __name__ == "__main__":
    vocab_size = 5
    batch_size = 10
    labels = torch.randint( 0 , vocab_size , size=( batch_size , 1 ) )
    logits = torch.rand( size=( batch_size , vocab_size ) )
    loss = sparse_crossentropy_with_logits( logits , labels )
    print( loss , loss.shape )