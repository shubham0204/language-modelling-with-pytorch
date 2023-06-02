import torch

def sparse_crossentropy_with_logits( logits , labels ):
    log_softmax_preds = torch.log2( torch.nn.functional.softmax( logits , dim=-1 ) )
    return torch.take( -log_softmax_preds , labels ).mean()

if __name__ == "__main__":
    labels = torch.randint( 0 , 10 , size=( 32 , 1 ) )
    logits = torch.rand( size=( 32 , 10 ) )
    loss = sparse_crossentropy_with_logits( logits , labels )
    print( loss , loss.shape )