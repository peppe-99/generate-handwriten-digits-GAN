"""https://aakashns.medium.com/generative-adverserial-networks-gans-from-scratch-in-pytorch-ad48256458a7"""
import os
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Normalize, Compose
from torchvision.datasets import MNIST
from random import randint
from config import *
from torchvision.utils import save_image
from utils import make_gif


def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)


def show_random():
    img, label = mnist[randint(0, len(mnist) - 1)]
    plt.imshow(denorm(img)[0], cmap='gray')
    plt.show()


def generate_one_image():
    y = generator(torch.randn(2, latent_size))
    gen_img = denorm(y.reshape((-1, 28, 28)).detach())
    plt.imshow(gen_img[0], cmap='gray')
    plt.show()


def make_discriminator():
    return nn.Sequential(
        nn.Linear(image_size, hidden_size),
        nn.LeakyReLU(0.2),
        nn.Linear(hidden_size, hidden_size),
        nn.LeakyReLU(0.2),
        nn.Linear(hidden_size, 1),
        nn.Sigmoid()
    )


def make_generator():
    return nn.Sequential(
        nn.Linear(latent_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, image_size),
        nn.Tanh()
    )


def reset_grad():
    d_optimizer.zero_grad()
    g_optimizer.zero_grad()


def train_discriminator(images):
    # Etichiette per la funzione BCE
    real_labels = torch.ones(batch_size, 1).to(device)
    fake_labels = torch.zeros(batch_size, 1).to(device)

    # Calcoliamo la perdità nel riconoscere immagini reali
    outputs = discriminator(images)
    d_loss_real = criterion(outputs, real_labels)
    real_score = outputs

    # Calcoliamo la perdità nel riconoscere immagini generate
    z = torch.randn(batch_size, latent_size).to(device)
    fake_images = generator(z)
    outputs = discriminator(fake_images)
    d_loss_fake = criterion(outputs, fake_labels)
    fake_score = outputs

    # Uniamo le perdite calcolate
    d_loss = d_loss_real + d_loss_fake

    # Resettiamo il gradiente
    reset_grad()

    # Aggiustiamo i parametri con la backpropagation
    d_loss.backward()
    d_optimizer.step()

    return d_loss, real_score, fake_score


def train_generator():
    # Generiamo immagini false e calcoliamo la perdita del discriminatore
    z = torch.randn(batch_size, latent_size).to(device)
    fake_images = generator(z)
    labels = torch.ones(batch_size, 1).to(device)
    g_loss = criterion(discriminator(fake_images), labels)

    # Backpropagation e ottimizzazione
    reset_grad()
    g_loss.backward()
    g_optimizer.step()
    return g_loss, fake_images


def save_real_images():
    for images, _ in data_loader:
        images = images.reshape(images.size(0), 1, 28, 28)
        save_image(denorm(images), os.path.join(sample_dir, "real_images.png"), nrow=10)
        break


def train_model():
    total_step = len(data_loader)
    d_losses, g_losses, real_scores, fake_scores = [], [], [], []

    for epoch in range(num_epochs):
        for i, (images, _) in enumerate(data_loader):
            # carichiamo un batch di immagini e convertiamole in tensori
            images = images.reshape(batch_size, -1).to(device)

            # alleniamo il discriminator ed il generatore
            d_loss, real_score, fake_score = train_discriminator(images)
            g_loss, fake_images = train_generator()

            if (i + 1) % 200 == 0:
                d_losses.append(d_loss.item())
                g_losses.append(g_loss.item())
                real_scores.append(real_score.mean().item())
                fake_scores.append(fake_score.mean().item())
                print('Epoch [{}/{}], Step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, D(x): {:.2f}, D(G(z)): {:.2f}'
                      .format(epoch+1, num_epochs, i + 1, total_step, d_loss.item(), g_loss.item(),
                              real_score.mean().item(), fake_score.mean().item()))

        save_fake_images(epoch + 1, sample_vectors)

    torch.save(generator.state_dict(), 'generator.pth')
    torch.save(discriminator.state_dict(), 'discriminator.pth')


# questo salvataggio funge da testing, poichè ad ogni epoca il generatore creerà
# dei numeri partendo dal medesimo vettore latente
def save_fake_images(index, sample_vectors):
    fake_images = generator(sample_vectors)
    fake_images = fake_images.reshape(fake_images.size(0), 1, 28, 28)
    fake_fname = "fake_images-{0:0=4d}.png".format(index)
    print(f"\nSaving {fake_fname}\n")
    save_image(denorm(fake_images), os.path.join(sample_dir, fake_fname), nrow=10)


if __name__ == "__main__":
    # ================== Prepariamo i dati ==================#
    mnist = MNIST(root="./", train=True, download=True, transform=Compose([
        ToTensor(),
        Normalize(mean=(0.5,), std=(0.5,))
    ]))
    data_loader = DataLoader(dataset=mnist, batch_size=batch_size, shuffle=True)
    print(f"device: {device}")

    # ================== Discriminator, Generator e Optimizer ==================#
    discriminator = make_discriminator()
    discriminator.to(device)

    generator = make_generator()
    generator.to(device)

    criterion = nn.BCELoss()
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=lr)
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=lr)

    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)

    sample_vectors = torch.randn(batch_size, latent_size).to(device)
    save_real_images()
    save_fake_images(0, sample_vectors)

    train_model()
    make_gif()
