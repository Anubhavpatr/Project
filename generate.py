import torch
import sys
import time
from gpt import GPTLanguageModel
from gpt import itos

if __name__ == "__main__":
    state_dict = torch.load('GPT.pth', map_location=torch.device('cpu'))
    model = GPTLanguageModel()
    model.load_state_dict(state_dict)
    model.eval()

    context = torch.zeros((1,1),dtype=torch.long)
    print("======== GPT MODEL GENERATING ===========")
    output_idx = model.generate(context, max_new_tokens=100)

    # Convert tokens to readable text (if using a tokenizer)
    for token in output_idx[0]:  # Iterate over generated tokens
        token_text = itos[token.item()]  # Convert token to string (or use tokenizer.decode())
        
        sys.stdout.write(token_text + " ")  # Print each token progressively
        sys.stdout.flush()  # Force immediate printing
        time.sleep(0.05)  # Delay to simulate typing effect

    print()