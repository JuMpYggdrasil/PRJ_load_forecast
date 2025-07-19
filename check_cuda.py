# pip install --upgrade pip
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

import torch
print(torch.__version__)
print("CUDA available:", torch.cuda.is_available())
