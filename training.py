import torch
import torch.nn as nn
from torch.nn import functional as F
import mmap
import random 
import pickle
import argparse

parser = argparse.ArgumentParser(description='This is a demo program')

# We add the arguments, specify the expected type, error message etc.
parser.add_argument('-batch_size', type=str, required=True, help='Please provide a batch_size')

args = parser.parse_args()

# use of the args
print(f'Batch size: {args.batch_size}')
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(device)

# hyper params
batch_size = int(args.batch_size) 
block_size = 64 # sequence length
max_iters = 200
learning_rate = 3e-4
eval_iters = 25
n_embd = 384
n_layer = 1 # number of decoded blocks
n_head = 1
dropout = 0.2

# opening the wizard of oz text 
chars = ""
with open('openwebtext/vocab.txt', 'r', encoding='utf-8') as f:
    text = f.read()
    chars = sorted(set(text))

vocab_size = len(chars)
    
# print(chars)

string_to_int = { ch:i for i,ch in enumerate(chars) }
int_to_string = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [string_to_int[c] for c in s]
decode = lambda l: ''.join([int_to_string[i] for i in l])

# entire wizard of oz text as data
data = torch.tensor(encode(text), dtype=torch.long)
# print(data[:100])

# getting chunks from the split
def get_random_chunk(split):
    filename = 'openwebtext/train_text.txt' if split == 'train' else 'openwebtext/valid_text.txt'
    with open(filename, 'rb') as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            # calculate the file size and randomize the starting position 
            file_size = len(mm)
            start_pos = random.randint(0, (file_size )- block_size*batch_size)
            
            # now look for the random block and read all text int hat block 
            mm.seek(start_pos)
            block = mm.read(block_size*batch_size-1)
            
            # decode into a string, ignore errors, ignore invalid byte sequences
            decoded_block = block.decode('utf-8', errors='ignore').replace('\r', '')
            
            # train test splits
            data = torch.tensor(encode(decoded_block), dtype=torch.long)
    
    return data

def get_batch(split):
    data = get_random_chunk(split)

    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """ one head of self-attention """
    
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        # expected values 
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # register the no look-ahead masking in the model state (just to reduce the training times)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        
        self.dropout = nn.Dropout(dropout)
      
    # attention heads
    def forward(self, x):
        # input of size (batch, time-stamp, channels)
        # output of (batch, time-stamp, head_size)
        B,T,C = x.shape
        k = self.key(x) # (B, T, HS)
        q = self.query(x)
        # now we compute the attention scores ('affinities')
        # weighted
        wei = q @ k.transpose(-2,-1) * k.shape[-1] **0.5  # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        # masked fill prevents lookahead
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        # Softmax to sharpen some values
        wei = F.softmax(wei, dim =-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B, T, HS)
        # finally out weighted value is multiplied by our weights
        out = wei @ v # ()
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """
    
    def __init__(self, num_heads, head_size):
        
        super().__init__()
        # create number of heads running in parallel and place them into the Module list 
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        # projection: projection of the 
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        # drop 20% of the networks neurons
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)# (B, T, F) -> (B, T, [h1, h1, h1, h1, h2, h2, h2, h2, h3, h3, h3, h3])
        out = self.dropout(self.proj(out))
        return out
        
class FeedForward(nn.Module):
    
    """ a simple linear layer followed by a non-linearity """
    
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear( 4 * n_embd,n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    
    """ Transformer block: communication followed by computation """
    
    def __init__(self, n_embd, n_head):
        # n_embd is embedding dimensionality, n_head is the number of heads we'd like
        super().__init__()
        # the number of features each head will be capturing in the MultiHeadAttention 
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        
    def forward(self, x):
        y = self.sa(x)
        x = self.ln1(x + y)
        y = self.ffwd(x)
        x = self.ln1(x + y)
        return x

# Logits typically refer to the raw scores or predictions generated by your model before applying softmax
# Targets represent the ground truth labels for your examples. These are the correct classes that you want your model to predict.

class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # Initialise embedding tables
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd) 
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        # creating the blocks (decoding blocks) sequentially
        self.blocks = nn.Sequential(*[Block(n_embd, n_head = n_head) for i in range(n_layer)])
        # Normalization using Layer Normalization
        self.ln_f = nn.LayerNorm(n_embd) 
        # The Linear Transformation defined with input size n_embd and output size vocab_size
        self.lm_head = nn.Linear(n_embd, vocab_size) 
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.0)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init
    
    # the forward propagation function including calculating the Loss 
    def forward(self, index, targets=None):
        # initializing the weights
        B, T = index.shape
        
        # index and targets (B, T) are both tensors of integers 
        tok_emd = self.token_embedding_table(index) # (B,T,C)
        pos_emd = self.position_embedding_table(torch.arange(T, device=device))# (T,C)
        x = tok_emd + pos_emd # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)
        
        if targets is None:
            loss = None
        else:
            # B representing batch size, T representing the sequence length (block_size), and c representing the original embedding dimensionality 
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss

    def generate(self, index, max_new_tokens):
        for _ in range(max_new_tokens):
            # crop index to the last block_size tokens
            conditional_index = index[:, block_size:]
            # get predictions
            logits, loss = self.forward(conditional_index)
            # focus only on the last time step
            logits = logits[:, -1, :] 
            # because we only selected the last step in each batch, logits now only has two dimensions. 
            # Of the original B, T, C as T is the last one chose, we are left with just B, C. A (2,2) matrix
            # use soft max to find the probabilities 
            probs = F.softmax(logits, dim=-1 )# (B, C)
            # sample from the distribution
            index_next = torch.multinomial(probs, num_samples = 1) # (B, 1)
            # finally append the sampled index to the running sequence
            index = torch.cat((index, index_next), dim=-1) # (B, T+1)
            
        return index 
    
model = GPTLanguageModel(vocab_size)

#------------------only works with an already made pkl file with the SAME  hyper parameters---------------
# print('Loading model params.....')
# with open('BigramLM-GPT-model-01.pkl') as f:
#     model=pickle.load(f)
# print('Loading complete!')
# --------------------------------------------------------------------------------------------------------

m = model.to(device)

#  creating the PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    print(iter)
    if iter % eval_iters == 0:
        losses = estimate_loss()
        print(f"step: {iter}, train loss: {losses['train']:.3f}, val loss: {losses['val']:.3f}")
        
    # sample batch of data, xb is the predictions, yb is the targets, during the initiation this is set to None
    xb, yb =get_batch('train')
    
    # evaluate the loss using standard training loop architecture for basic models
    logits, loss = model.forward(xb,yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
#  A scalar representation of how the model is doing, compared to the true values calculated during 'Forward Propagation'
# A lowe value indicates an improvement of the Bigram Language Model
print(loss.item())

with open('BigramLM-GPT-model-01.pkl', 'wb') as f:
    pickle.dump(model, f)
print('model has been saved!')

# context = torch.zeros((1,1), dtype = torch.long, device=device)
# generated_chars = decode(m.generate(context, max_new_tokens=500)[0].tolist())
# print(generated_chars)