import torch

from vae import VAE
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from PIL import Image

import matplotlib.pyplot as plt
import os
import argparse
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim

class CustomDataset(Dataset):
    def __init__(self, path_to_dir, data, transform=None):
        self.path_to_dir = path_to_dir
        self.data = data
        if not transform:
            transform = transforms.Compose([
                transforms.ToTensor()
            ])
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.path_to_dir, self.data[idx])
        img = Image.open(img_path)
        img = self.transform(img)
        return img

def prepare_data(path_to_dir, batch_size=32, transform=None):
    train_data = os.listdir(path_to_dir)
    train_dataset = CustomDataset(path_to_dir, train_data, transform)
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
    return train_dataloader

def vae_loss(x, x_reconstructed, mu, logvar, alpha=3.0):
    bce_loss = nn.BCELoss(reduction="sum")
    reconstructed_loss = bce_loss(x_reconstructed, x)
    kl_loss = -0.5 * torch.sum(1 + logvar - torch.pow(mu, 2) - torch.exp(logvar))
    loss = reconstructed_loss + alpha * kl_loss
    return loss

def train_model(num_epochs, model, train_dataloader, optimizer, alpha, device, path_to_checkpoint):
    os.makedirs(path_to_checkpoint, exist_ok=True)

    model.train()
    model.to(device)

    min_loss = float("inf")
    train_losses = []

    for epoch in range(num_epochs):
        total_train_loss = 0
        for img in tqdm(train_dataloader, desc=f"Epoch {epoch+1}", unit="batch", colour="RED"):
            img = img.to(device)

            optimizer.zero_grad()

            img_reconstructed, mu, logvar = model(img)
            loss = vae_loss(img, img_reconstructed, mu, logvar, alpha)
            loss.backward()

            optimizer.step()

            total_train_loss += loss.item()
        train_losses.append(total_train_loss)

        if total_train_loss < min_loss:
            min_loss = total_train_loss
            torch.save(model.state_dict(), os.path.join(path_to_checkpoint, "best_model.pt"))
            print(f"    Save Model in Epoch {epoch+1}")
        torch.save(model.state_dict(), os.path.join(path_to_checkpoint, "last_model.pt"))
        print(f"    Training loss = {(total_train_loss/len(train_dataloader.dataset)):.5f}")

    epochs = range(1, num_epochs + 1)
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, label='Train Loss', marker='o')
    plt.title('Training Loss arcording to Epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.show()
def main(num_epochs, latent_dim, batch_size, lr, alpha, path_to_checkpoint, path_to_data):
    train_dataloader = prepare_data(path_to_data, batch_size)

    model = VAE(latent_dim)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = optim.Adam(model.parameters(), lr)

    print(f"Details of training:")
    print(f"    Epochs = {num_epochs}, Latent dim = {latent_dim}, Batch size = {batch_size}")
    print(f"    Alpha = {alpha}, Device = {device}, Save checkpoint = {path_to_checkpoint}")
    print("Ready Training !!!")

    train_model(num_epochs, model, train_dataloader, optimizer, alpha, device, path_to_checkpoint)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyper-parameters for training")
    parser.add_argument("--epoch", type=int, help="No. of epochs for training", default=50)
    parser.add_argument("--latent_dim", type=int, help="No. dim of latent space", default=256)
    parser.add_argument("--lr", type=float, help="Learning rate", default=1e-4)
    parser.add_argument("--batch_size", type=int, help="Batch size", default=64)
    parser.add_argument("--alpha", type=float, help="Loss factor", default=3.0)
    parser.add_argument("--path_checkpoint", type=str, help="Save checkpoint path", default="../checkpoint/")
    parser.add_argument("--path_data", type=str, help="Dir of data", default="../data/")

    args = parser.parse_args()

    main(
        num_epochs=args.epoch, latent_dim=args.latent_dim,
        batch_size=args.batch_size,
        lr=args.lr, alpha=args.alpha,
        path_to_checkpoint=args.path_checkpoint,
        path_to_data=args.path_data
    )