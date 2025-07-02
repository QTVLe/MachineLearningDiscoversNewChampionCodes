import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

from dataclasses import dataclass
from pathlib import Path
import time

@dataclass
class TransformerConfig:
    # ugly pre-defined parameters
    field_embedding_dim = 4
    n_input = 49
    n_embd = field_embedding_dim * n_input

    # actual parameters
    device: str
    # fixed parameters
    field_base: int = 8
    batchsize: int = 128
    n_output: int = 49 # predict a class (minimum distance) 0, ... n_output
    n_input: int = n_input
    block_size: int = 49 # max length of input for positional embedding
    field_embedding_dim: int = field_embedding_dim # encoding of the elements of the field, equals the base of the field
    # Tunable parameters
    n_head: int = 7
    n_layer: int = 2  # number of transformer blocks
    n_embd: int = n_embd
    dropout: float = 0.2
    learning_rate: float = 3e-6
    
    global_dtype: torch.dtype = torch.float32

    def get_dict(self):
        hparam_dict =  {
                  'n_embd': self.n_embd,
                  'n_head': self.n_head,
                  'n_layer': self.n_layer,
                  'dropout': self.dropout,
                  'learning_rate': self.learning_rate,
              }
        return hparam_dict

# Custom Dataset
class MyDataset(Dataset):
    def __init__(self, codes, lengths, mindists):
        self.codes = codes
        self.lengths = lengths
        self.mindists = mindists

    def __len__(self):
        return len(self.codes)

    def __getitem__(self, index):
        return self.codes[index].long(), self.lengths[index], self.mindists[index].long()

class Head(nn.Module):
    """ one head of self-attention - with proper variable sequence length handling"""

    def __init__(self, n_embd, head_size, dropout):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, padding_mask):
        # input of size (batch, time-step, n_input)
        # output of size (batch, time-step, head size)

        B, T, C = x.shape
        # .masked_fill(padding_mask == 0, float('nan'))   # (B,T,C)*(C, hs) -> (B,T,hs)
        k = self.key(x)
        # .masked_fill(padding_mask == 0, float('nan')) # (B,T,C)*(C, hs) -> (B,T,hs)
        q = self.query(x)

        # compute attention scores ("affinities")
        # (B, T, hs) @ (B, hs, T) * 1/sqrt(head_size) -> (B, T, T)
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5
        # (B, T, T) - masking T in dim = 2
        wei = wei.masked_fill(padding_mask == 0, float('-inf'))
        # (B, T, T) - apply softmax normalization to the last (T) dimension
        wei = F.softmax(wei, dim=-1)
        padding_mask_transpose = padding_mask.transpose(-2, -1)
        # (B, T, T) - masking T in dim = 1
        wei = wei.masked_fill(padding_mask_transpose == 0, float('0'))
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x)  # (B,T,C)*(C, hs) -> (B,T,hs)
        out = wei @ v  # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out

def create_padding_mask(seq, lengths, pad_token=0, transposed=False):
    """
    Creates a padding mask for variable-length sequences.

    Args:
        seq (torch.Tensor): Tensor of shape (batch_size, seq_len, input_size), containing the input sequences.
        lengths (torch.Tensor): Tensor of shape (batch_size), containing the lengths of the input sequences.
        pad_token (int, optional): Token used for padding. Defaults to 0.

    Returns:
        torch.Tensor: Padding mask of shape (batch_size, 1, seq_len).
    """
    # Generate a mask where padding tokens are 1 and other tokens are 0
    B = seq.size(0)
    seq_len = seq.size(1)
    mask = torch.arange(seq_len, device=lengths.device).expand(
        B, seq_len) < lengths.unsqueeze(1)
    if transposed:
        mask = mask.unsqueeze(2)
    else:
        mask = mask.unsqueeze(1)
    return mask.long()  # Shape: transposed=False -> (batch_size, 1, seq_len); transposed=True -> (batch_size, seq_len, 1)

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, n_embd, num_heads, head_size, dropout):
        super().__init__()
        self.heads = nn.ModuleList([Head(n_embd, head_size, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, lengths):
        # crate a padding mask for variable-length sequences
        padding_mask = create_padding_mask(x, lengths)
        out = torch.cat([h(x, padding_mask) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x, lengths):
        # x size (batch, time-step, n_embd)
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head, dropout):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_embd, n_head, head_size, dropout)
        self.ffwd = FeedFoward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x, lengths):
        x = x + self.sa(self.ln1(x), lengths)
        x = x + self.ffwd(self.ln2(x), lengths)
        return (x, lengths)

class mySequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs

def positionalencoding1d(seq_length, n_channels, device='cpu'):
    """
    :param n_channels: dimension of the model
    :param seq_length: length of positions
    :return: length*d_model position matrix
    """
    if n_channels % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(n_channels))
    pe = torch.zeros(seq_length, n_channels)
    position = torch.arange(0, seq_length).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, n_channels, 2, dtype=torch.float) *
                         -(torch.log(torch.tensor(10000.0)) / n_channels)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)
    pe = pe.unsqueeze(0)  # adding batch dim

    return pe.to(device)

class TimeAttention(nn.Module):
    """ Attention-Based Summarization along time dimension"""

    def __init__(self, n_embd):
        super().__init__()
        self.attn_proj = nn.Linear(n_embd, 1)  # projecting

    def forward(self, x, lengths):
        # input of size (batch, time-step, n_embed)
        # output of size (batch, n_embed)
        attn_scores = self.attn_proj(x)  # (B, T, 1)
        padding_mask_transposed = create_padding_mask(
            x, lengths, transposed=True)  # (B, T, 1)
        attn_scores = attn_scores.masked_fill(
            padding_mask_transposed == 0, float('-inf'))
        # (B, T, 1) attention scores
        attn_weights = F.softmax(attn_scores, dim=1)
        x_summary = torch.sum(attn_weights * x, dim=1)  # (B, C) weighted sum

        return x_summary

class ToricTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device = config.device
        self.n_embd = config.n_embd

        # each token directly reads off the logits for the next token from a lookup table
        self.embeddings = nn.Embedding(config.field_base, config.field_embedding_dim)

        self.blocks = mySequential(
            *[Block(config.n_embd, config.n_head, config.dropout) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)  # final layer norm
        self.lm_head = nn.Linear(config.n_embd, config.n_output)
        self.time_summary = TimeAttention(config.n_embd)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, code, lengths, targets=None, weights=None):
        # target shape (B,)
        B, T, _ = code.shape
        
        tok_emb = self.embeddings(code).view(B, T, self.n_embd) # (B,T, n_input) -> (B,T,n_embd)
        pos_emb = positionalencoding1d(T, self.n_embd, device=self.device)  # (1,T,n_embd)
        x = tok_emb + pos_emb  # (B,T,n_embd)

        x, _ = self.blocks(x, lengths)  # (B,T,n_embd)
        x = self.ln_f(x)  # (B,T,n_embd)

        x_summary = self.time_summary(x, lengths)  # (B, n_embd)

        logits = self.lm_head(x_summary)  # (B, n_output)

        if targets is None:
            loss = None
        else:
            loss = F.cross_entropy(logits, targets, weight=weights)

        return logits, loss

    def configure_optimizers(self, config):
        optimizer = torch.optim.AdamW(self.parameters(), lr=config.learning_rate)
        return optimizer

    def save(self, hparams, save_path, optimizer, n_epochs, test_loss):
        torch.save({
            'epoch': n_epochs,
            'model_parameters': hparams,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'test_loss': test_loss
        }, save_path)

    def load(self, load_path, optimizer):
        device = self.device
        checkpoint = torch.load(load_path, weights_only=False, map_location=device)
        self.load_state_dict(checkpoint['model_state_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['model_parameters'], checkpoint['epoch']

    def predict(self, code):
        self.eval()
        with torch.no_grad():
            lengths = torch.tensor([code.shape[1]]).to(self.device)
            if len(code.shape) < 3:
                code = code.to(self.device).unsqueeze(0)
            
            logits, _ = self.forward(code, lengths)

            probs = F.softmax(logits, dim=1).cpu()
            vector = torch.unsqueeze(torch.arange(49), dim=0) # shape (1, n_output)
            expectations = (vector * probs).sum(dim=1) # shape (B,)
            return expectations.item()

def load_data(data_dir, batchsize, shuffle=True, split=0, get_weights=False):
    data = torch.load(data_dir, weights_only=False)

    generators_data = data['generators']
    generator_lens_data = data['generator_lens']
    mindist_data = data['mindists']
    dataset_len = mindist_data.shape[0]

    if shuffle:
        permutation = torch.randperm(dataset_len)
        generators_data = generators_data[permutation]
        generator_lens_data = generator_lens_data[permutation]
        mindist_data = mindist_data[permutation]

    if split != 0:
        split_len = int(dataset_len * split)
        trainset = MyDataset(generators_data[:split_len],
                            generator_lens_data[:split_len], mindist_data[:split_len])
        testset = MyDataset(generators_data[split_len:],
                            generator_lens_data[split_len:], mindist_data[split_len:])

        train_dataloader = DataLoader(trainset, batch_size=batchsize, shuffle=shuffle)
        test_dataloader = DataLoader(testset, batch_size=batchsize, shuffle=shuffle)
        if get_weights:
            return train_dataloader, test_dataloader, data['upsampling_factors']
        else:
            return train_dataloader, test_dataloader
    else:
        trainset = MyDataset(generators_data,
                            generator_lens_data, mindist_data)
        train_dataloader = DataLoader(trainset, batch_size=batchsize, shuffle=shuffle)
        if get_weights:
            return train_dataloader, data['upsampling_factors']
        else:
            return train_dataloader

def get_weights(weights_dict, config):
    # define data weights to cancel oversampling
    if weights_dict is not None:
        weights = torch.full((config.n_output,), 1, dtype=torch.float32)
        for key in weights_dict.keys():
            weights[key] = 1./weights_dict[key]
        weights = weights.to(config.device)
        return weights
    else:
        weights = None
        return weights

@torch.no_grad()
def estimate_loss(model, testset, weights=None):
    model.eval()
    device = model.device

    test_loss = torch.zeros(len(testset))
    for i, test_data in enumerate(testset):
        code, lengths, mindist = test_data
        code, lengths, mindist = code.to(
            device), lengths.to(device), mindist.to(device)

        output, loss = model(code, lengths, targets=mindist, weights=weights)
        test_loss[i] = loss.item()
    test_loss = test_loss.mean()

    model.train()

    return test_loss

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss * (1. + self.min_delta)):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def log_gradients(model, epoch, writer):
    for name, param in model.named_parameters():
        writer.add_histogram(f'gradients/{name}', param.grad, epoch)

def train_transformer(name, mode, n_epochs, config):
    name_mode = name + "_" + mode

    device = config.device
    hparam_dict = config.get_dict()

    # Set up loggig to TensorBoard
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(name_mode)

    if mode == 'pre_train':
        # load data
        data_dir = 'gigaset2_F8/gigaset2_F8_pre_train.pt'
        train_dataloader, test_dataloader = load_data(data_dir, config.batchsize, shuffle=True, split=0.9)
        weights_dict = None
    elif mode == 'post_train':
        data_dir = 'gigaset2_F8/gigaset2_F8_post_train.pt'
        train_dataloader, weights_dict = load_data(data_dir, config.batchsize, shuffle=True, get_weights=True)

        data_dir = 'gigaset2_F8/gigaset2_F8_post_test.pt'
        test_dataloader = load_data(data_dir, config.batchsize, shuffle=True)
    else:
      raise ValueError("mode must be \'pre_train\' or \'post_train\'")

    # initialize model
    model = ToricTransformer(config)
    model.to(device)
    # print the number of parameters in the model
    print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')

    # create a PyTorch optimizer
    optimizer = model.configure_optimizers(config)

    # create Early stopper
    if mode == 'pre_train':
        early_stopper = EarlyStopper(patience=10, min_delta=0.05)
    elif mode == 'post_train':
        early_stopper = EarlyStopper(patience=5, min_delta=0.05)
    else:
        raise ValueError("mode must be \'pre_train\' or \'post_train\'")

    # load correct model
    if mode == 'post_train':
        if Path(name + '_pre_train_best.pt').exists():
            model.load(name + '_pre_train_best.pt', optimizer)
            print("Loading ", name, "_pre_train_best.pt model")
        else:
            model.load(name + '_pre_train.pt', optimizer)
            print("Loading ", name, "_pre_train.pt model")

    # define data weights to cancel oversampling
    weights = get_weights(weights_dict, config)

    # log model
    code, lengths, mindist = next(iter(train_dataloader))
    code, lengths, mindist = code.to(
        device), lengths.to(device), mindist.to(device)
    writer.add_graph(model, [code, lengths, mindist], use_strict_trace=False)

    # test input
    model.eval()
    logits, loss = model(code, lengths, targets=mindist)
    # print("test logits: ", logits)
    print("test loss: ", loss)

    # run training
    print("dataset length: ", len(train_dataloader), " batches")
    eval_interval = int(len(train_dataloader)/5)

    # timings
    t0 = time.time()
    t1 = time.time()

    # init best loss
    best_test_loss = float('inf')

    model.train()
    for epoch in range(n_epochs):
        print(f'epoch {epoch+1}')

        # init losses
        current_loss = 0
        test_loss = 0
        for iteration, train_data in enumerate(train_dataloader):
            # sample a batch of data
            code, lengths, mindist = train_data
            code, lengths, mindist = code.to(
                device), lengths.to(device), mindist.to(device)

            # evaluate the loss
            output, loss = model(code, lengths, targets=mindist, weights=weights)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            current_loss += loss.item()
            optimizer.step()

            # every once in a while evaluate the loss on train and val sets
            if iteration % eval_interval == 0 and iteration != 0:
                train_loss = current_loss / eval_interval
                current_loss = 0

                test_loss = estimate_loss(model, test_dataloader)

                t2 = time.time()
                dt = t2 - t1
                t1 = t2
                print(f"step {iteration} ({dt:.2f}s): train loss {train_loss:.4f}, val loss {test_loss:.4f}")

                # log losses
                writer.add_scalars('Losses', {'train': train_loss,'test': test_loss},epoch*len(train_dataloader)+iteration)

                if test_loss < best_test_loss:
                    best_test_loss = test_loss
                    save_path = name_mode + "_best.pt"
                    model.save(hparam_dict, save_path, optimizer, epoch, test_loss)

        if early_stopper.early_stop(test_loss):
            break

    t2 = time.time()
    dt = t2 - t0
    print(f"Total time: {dt:.2f}s")

    metrics_dict = {
        'test_loss': test_loss
    }

    writer.add_hparams(hparam_dict, metrics_dict)

    writer.close()

    save_path = name_mode + ".pt"
    model.save(hparam_dict, save_path, optimizer, n_epochs, test_loss)

if __name__ == '__main__':    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    config = TransformerConfig(device=device)
    train_transformer('transformer_436_vector', 'post_train', 100, config)

