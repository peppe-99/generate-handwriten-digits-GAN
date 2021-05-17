import os
from training import make_generator
from training import denorm
from training import save_image
from config import *


def main():
    generator = make_generator()
    generator.to(device)
    generator.load_state_dict(torch.load('generator.pth'))

    sample_vectors = torch.randn(batch_size, latent_size).to(device)
    fake_images = generator(sample_vectors)
    fake_images = fake_images.reshape(fake_images.size(0), 1, 28, 28)
    fake_fname = "fake_images_testing.png"
    print(f"\nSaving {fake_fname}\n")
    save_image(denorm(fake_images), os.path.join(sample_dir, fake_fname), nrow=10)


if __name__ == '__main__':
    main()
