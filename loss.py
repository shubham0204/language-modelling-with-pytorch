import torch

def sparse_crossentropy_with_logits(logits, targets):
    log_softmax_preds = torch.log2( torch.nn.functional.softmax( logits , dim=-1 ) )
    return torch.take(-log_softmax_preds, targets).mean()

def accuracy( preds , targets ):
    preds = torch.argmax(torch.nn.functional.softmax(preds, dim=-1), dim=-1)
    acc = torch.mean(torch.eq(preds, targets).float())
    return acc

if __name__ == "__main__":
    vocab_size = 5
    seq_length = 7
    batch_size = 10
    labels = torch.randint( 0 , vocab_size , size=( batch_size , seq_length ) )
    logits = torch.rand( size=( batch_size , seq_length , vocab_size ) )
    loss = sparse_crossentropy_with_logits( logits , labels )
    print( loss , loss.shape )

    print( accuracy( logits , labels ) )