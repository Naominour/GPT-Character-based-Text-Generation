from flask import Flask, request, render_template
import torch
from model import GPTConfig, GPT

app = Flask(__name__)

# Load the model
out_dir = 'out-shakespeare-char'
ckpt_path = os.path.join(out_dir, 'ckpt.pt')
checkpoint = torch.load(ckpt_path, map_location='cpu')
gptconf = GPTConfig(**checkpoint['model_args'])
model = GPT(gptconf)
model.load_state_dict(checkpoint['model'])
model.eval()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    prompt = request.form['prompt']
    start_ids = encode(prompt)
    x = torch.tensor(start_ids, dtype=torch.long).unsqueeze(0)

    with torch.no_grad():
        y = model.generate(x, max_new_tokens=100, temperature=0.8, top_k=200)
    
    result = decode(y[0].tolist())
    return {'result': result}

def encode(prompt):
    enc = tiktoken.get_encoding("gpt2")
    return enc.encode(prompt)

def decode(tokens):
    enc = tiktoken.get_encoding("gpt2")
    return enc.decode(tokens)

if __name__ == '__main__':
    app.run(debug=True)
