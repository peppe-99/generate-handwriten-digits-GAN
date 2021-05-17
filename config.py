import torch

batch_size = 100
image_size = 784  # 28 x 28
hidden_size = 256
latent_size = 64
lr = 0.0002
sample_dir = 'samples'
num_epochs = 150
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')