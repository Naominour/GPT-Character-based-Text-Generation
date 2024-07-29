import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 64 # number of independent sequences we will process in parallel
block_size = 256 # number of maximum context lenght for predicctions
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_mbed = 384
n_head = 6
n_layer = 6
dropout = 0.2
#--------------

torch.manual_seed(1337)

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

# creat a mapping from charecters to integers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# train and test split
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]


# data loading
def get_batch(split):
    data = train_data if split=='train' else val_data
    ix = torch.randint(len(data)-block_size, (batch_size,))
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
    def __init__(self, head_size):
       super().__init__()
       self.key = nn.Linear(n_mbed, head_size, bias=False)
       self.query = nn.Linear(n_mbed, head_size, bias=False)
       self.value = nn.Linear(n_mbed, head_size, bias=False)
       self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
       self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        q = self.query(x)
        k = self.key(x)

        wei = q @ k.transpose(-2,-1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
   
   def __init__(self, num_heads, head_size):
      super().__init__()
      self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
      self.proj = nn.Linear(n_mbed, n_mbed)
      self.dropout = nn.Dropout(dropout)

   def forward(self, x):
      out = torch.cat([h(x) for h in self.heads], dim=-1)
      out = self.proj(out)        
      out = self.dropout(out)
      return out 
    

class FeedForward(nn.Module):
   def __init__(self, n_embd):
    super().__init__()
    self.net = nn.Sequential(
        nn.Linear(n_mbed, 4 * n_mbed),
        nn.ReLU(),
        nn.Linear(4 * n_mbed, n_mbed),
        nn.Dropout(dropout),  # apply dropout to the output of the feed-forward network
   )
   def forward(self, x):
      return self.net(x)



class Block(nn.Module):
   
   def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

   def forward(self, x):
        x = self.sa(self.ln1(x))
        x = self.ffwd(self.ln2(x)) 
        return x
      



class BigramLanguageModel(nn.Module):

  def __init__(self):
    super().__init__()
    self.token_embedding_table = nn.Embedding(vocab_size, n_mbed)
    self.position_embedding_table = nn.Embedding(block_size, n_mbed)
    self.blocks = nn.Sequential(*[Block(n_mbed, n_head=n_head) for _ in range(n_layer)])
    self.sa_heads = MultiHeadAttention(4, n_mbed//4)
    self.ffwd = FeedForward(n_mbed)
    self.lm_head = nn.Linear(n_mbed, vocab_size)

  def forward(self, idx, targets=None):
    B, T = idx.shape
    tok_emb = self.token_embedding_table(idx) # batch, time, channels
    pos_emb = self.position_embedding_table(torch.arange(T, device=device))
    x = tok_emb + pos_emb 
    x = self.sa_heads(x) # batch, time, channels
    x = self.ffwd(x) # batch, time, channels
    x = self.blocks(x)
    logits = self.lm_head(x) # batch, time, vocab_size
   
    if targets is None:
      loss = None 
    else:
      B, T, C = logits.shape
      logits = logits.view(B*T, C)
      targets = targets.view(B*T)
      loss = F.cross_entropy(logits, targets)

    return logits, loss

  def generate(self, idx, max_new_tokens):
    for _ in range(max_new_tokens):
      
      idx_cond = idx[:, -block_size:]
      # get the predictions
      logits, loss = self(idx_cond)
      # focus only on the last time step
      logits = logits[:, -1, :] #
      # apply softmax to get probabilities
      probs = F.softmax(logits, dim=-1) # batch,
      # sample from the distribution
      idx_next = torch.multinomial(probs, num_samples=1) # batch,
      # append sampled index to the running sequence
      idx = torch.cat((idx, idx_next), dim=1) # batch
    return idx

model = BigramLanguageModel()
m = model.to(device)

optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

for iter in range(max_iters):
    if iter % eval_interval ==0:    
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')
    
    logit, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

   
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
    

   
