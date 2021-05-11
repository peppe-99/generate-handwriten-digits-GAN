import torch
import torch.nn as nn

discriminator = nn.Sequential()
generator = nn.Sequential()

d_checkpoint = torch.load("./discrimiator.ckpt")
g_checkpoint = torch.load("./generator.ckpt")
print(d_checkpoint)
discriminator.load_state_dict(d_checkpoint['model'])
generator.load_state_dict(g_checkpoint['model_state_dict'])