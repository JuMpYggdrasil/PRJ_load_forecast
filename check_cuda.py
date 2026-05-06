# pip install --upgrade pip
# old
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# new
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

import torch
print(torch.__version__)
print("CUDA available:", torch.cuda.is_available())
