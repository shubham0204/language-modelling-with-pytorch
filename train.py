import datetime
import os
import pickle

import torch
import wandb

from config import GlobalConfig
from model import Transformer
from loss import cross_entropy_loss, perplexity

data_config = GlobalConfig.data_config
train_config = GlobalConfig.train_config
model_config = GlobalConfig.model_config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device {} for training".format(device))

torch.manual_seed(1335)

if train_config.wandb_logging_enabled:
    wandb.login()


def log_metrics(train_loss, train_ppl, val_loss, val_ppl):
    wandb.log(
        {
            "loss": train_loss,
            "perplexity": train_ppl,
            "val_loss": val_loss,
            "val_perplexity": val_ppl,
        }
    )


def get_checkpoint_path() -> str:
    checkpoints_path = train_config.checkpoint_path
    if train_config.checkpoint_path == "auto":
        checkpoints_path = datetime.datetime.now().strftime("%H_%M_%d_%m_%Y")
    if not os.path.exists(checkpoints_path):
        os.makedirs(checkpoints_path)
    return checkpoints_path


def get_batch_loader(data_tensors_path: str, batch_size: int, input_length: int):
    with open(os.path.join(data_tensors_path, "sequences.pkl"), "rb") as file:
        indexed_sequences = pickle.load(file)
    test_split_index = int((1.0 - data_config.test_split) * len(indexed_sequences))
    train_indexed_sequences = indexed_sequences[:test_split_index]
    test_indexed_sequences = indexed_sequences[test_split_index:]

    def get_train_batch() -> tuple[torch.Tensor, torch.Tensor]:
        random_indices = torch.randint(
            0, len(train_indexed_sequences) - input_length - 2, size=(batch_size,)
        )
        inputs = torch.stack(
            [
                torch.tensor(train_indexed_sequences[i : i + input_length])
                for i in random_indices
            ]
        )
        outputs = torch.stack(
            [
                torch.tensor(train_indexed_sequences[i + 1 : i + input_length + 1])
                for i in random_indices
            ]
        )
        return inputs, outputs

    def get_test_batch() -> tuple[torch.Tensor, torch.Tensor]:
        random_indices = torch.randint(
            0, len(test_indexed_sequences) - input_length - 2, size=(batch_size,)
        )
        inputs = torch.stack(
            [
                torch.tensor(test_indexed_sequences[i : i + input_length])
                for i in random_indices
            ]
        )
        outputs = torch.stack(
            [
                torch.tensor(test_indexed_sequences[i + 1 : i + input_length + 1])
                for i in random_indices
            ]
        )
        return inputs, outputs

    return get_train_batch, get_test_batch


def train_on_batch(model, batch_dispatcher, optimizer):
    model.train()
    inputs, outputs = batch_dispatcher()
    inputs, outputs = inputs.to(device), outputs.to(device)
    batch_size, seq_length = inputs.shape
    optimizer.zero_grad(set_to_none=True)
    preds = model(inputs)
    preds = preds.view(batch_size * seq_length, data_config.vocab_size)
    targets = outputs.view(
        batch_size * seq_length,
    )
    loss = cross_entropy_loss(preds, targets)
    loss.backward()
    optimizer.step()
    ppl = perplexity(loss)
    return loss.cpu().item(), ppl.cpu().item()


def test_on_batch(model, batch_dispatcher):
    model.eval()
    inputs, outputs = batch_dispatcher()
    inputs, outputs = inputs.to(device), outputs.to(device)
    batch_size, seq_length = inputs.shape
    preds = model(inputs)
    preds = preds.view(batch_size * seq_length, data_config.vocab_size)
    targets = outputs.view(
        batch_size * seq_length,
    )
    loss = cross_entropy_loss(preds, targets)
    ppl = perplexity(loss)
    return loss.cpu().item(), ppl.cpu().item()


ckpt_path = get_checkpoint_path()

model = Transformer(
    vocab_size=data_config.vocab_size,
    embedding_dim=model_config.embedding_dim,
    seq_length=data_config.seq_length,
    num_blocks=model_config.num_blocks,
    num_heads_in_block=model_config.num_heads_in_block,
    dropout=model_config.dropout,
)
optimizer: torch.optim.Optimizer = torch.optim.AdamW(
    model.parameters(), lr=train_config.learning_rate
)

if train_config.resume_training:
    checkpoint = torch.load(train_config.resume_training_checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.learning_rate)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    print(
        "Resuming training with model {}".format(
            train_config.resume_training_checkpoint_path
        )
    )
else:
    model.to(device)
    print("No checkpoint loaded")

model.to(device)

if train_config.compile_model:
    execute_model = torch.compile(model)
    print("Model compiled with torch.compile")
else:
    execute_model = model
    print("Training uncompiled model")

if train_config.wandb_logging_enabled:
    wandb.init(
        project=train_config.wandb_project_name,
        config=model_config.update(train_config),
    )

train_batch_dispatcher, test_batch_dispatcher = get_batch_loader(
    data_config.data_tensors_path, train_config.batch_size, data_config.seq_length
)

prev_val_loss = 1e5
for iter in range(train_config.num_train_iter):
    train_loss, train_ppl = train_on_batch(
        execute_model, train_batch_dispatcher, optimizer
    )

    if (
        iter % train_config.test_interval == 0
        or iter == train_config.num_train_iter - 1
    ):
        avg_val_loss = 0.0
        avg_val_ppl = 0.0
        for val_iter in range(train_config.num_test_iter):
            val_loss, val_ppl = test_on_batch(model, test_batch_dispatcher)
            avg_val_loss += val_loss
            avg_val_ppl += val_ppl
        avg_val_loss /= train_config.num_test_iter
        avg_val_ppl /= train_config.num_test_iter

        if train_config.wandb_logging_enabled:
            log_metrics(train_loss, train_ppl, avg_val_loss, avg_val_ppl)

        print(
            "{} loss={:.5f}, perplexity={:.5f} , val_loss={:.5f}, val_perplexity={:.5f}".format(
                iter, train_loss, train_ppl, avg_val_loss, avg_val_ppl
            )
        )

        if avg_val_loss < prev_val_loss:
            prev_val_loss = avg_val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": avg_val_loss,
                    "config": GlobalConfig.config,
                },
                os.path.join(ckpt_path, "model_{}.pt".format(iter)),
            )
