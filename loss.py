import torch

def sparse_crossentropy_with_logits(logits, targets):
    log_softmax_preds = -torch.nn.functional.log_softmax( logits , dim=-1 )
    return torch.take( log_softmax_preds , targets ).mean()

def perplexity( cross_entropy_loss ):
    return torch.exp( cross_entropy_loss )

if __name__ == "__main__":
    vocab_size = 5
    seq_length = 7
    batch_size = 10
    labels = torch.randint( 0 , vocab_size , size=( batch_size , seq_length ) )
    logits = torch.rand( size=( batch_size , seq_length , vocab_size ) )

    labels = labels.view( batch_size * seq_length )
    logits = logits.view( batch_size * seq_length , vocab_size )
    loss = sparse_crossentropy_with_logits(logits, labels)
    loss2 = torch.nn.functional.cross_entropy( logits , labels )
    print( loss , loss.shape )
    print( loss2 )