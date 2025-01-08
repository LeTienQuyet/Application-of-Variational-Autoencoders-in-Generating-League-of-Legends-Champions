from src.models.vae import VAE
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from PIL import Image

import matplotlib.pyplot as plt
import os
import argparse
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim

class CustomDataset(Dataset):
    def __init__(self, path_to_dir, split, data, input_dim, transform=None):
        self.path_to_dir = path_to_dir
        self.split = split
        self.data = data
        self.input_dim = input_dim
        if transform is None:
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.path_to_dir, self.split, self.data[idx])
        img = Image.open(img_path)
        if self.input_dim != 4:
            if self.input_dim == 3:
                img = img.convert("RGB")
            else:
                raise ValueError("input_dim should be 3 or 4.")
        img = self.transform(img)
        return img

def prepare_data(path_to_dir, input_dim, batch_size=32, transform=None):
    train_data = os.listdir(os.path.join(path_to_dir, "train"))
    val_data = os.listdir(os.path.join(path_to_dir, "val"))

    train_dataset = CustomDataset(path_to_dir, "train", train_data, input_dim, transform)
    val_dataset = CustomDataset(path_to_dir, "val", val_data, input_dim, transform)

    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size, shuffle=False)
    return train_dataloader, val_dataloader

def vae_loss(x, x_reconstructed, mu, logvar, beta):
    reconstructed_loss = F.mse_loss(x_reconstructed, x, reduction="sum")
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    loss = reconstructed_loss + beta * kl_loss
    return loss, reconstructed_loss, kl_loss

def val_model(model, val_dataloader, beta, device):
    model.to(device)
    model.val()

    total_val_loss = 0
    total_val_reconstructed_loss = 0
    total_val_kl_loss = 0

    with torch.no_grad():
        for img in val_dataloader:
            img = img.to(device)

            img_reconstructed, mu, logvar = model(img)
            loss, reconstructed_loss, kl_loss = vae_loss(img, img_reconstructed, mu, logvar, beta)

            total_val_loss += loss.item()
            total_val_reconstructed_loss += reconstructed_loss.item()
            total_val_kl_loss += kl_loss.item()

    avg_val_loss = total_val_loss / len(val_dataloader.dataset)
    avg_val_reconstructed_loss = total_val_reconstructed_loss / len(val_dataloader.dataset)
    avg_val_kl_loss = total_val_kl_loss / len(val_dataloader.dataset)

    return avg_val_loss, avg_val_reconstructed_loss, avg_val_kl_loss
def train_model(num_epochs, model, train_dataloader, val_dataloader, optimizer, alpha, device, path_to_checkpoint):
    os.makedirs(path_to_checkpoint, exist_ok=True)

    model.to(device)
    model.train()

    min_loss = float("inf")

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        total_train_loss = 0
        total_train_reconstructed_loss = 0
        total_train_kl_loss = 0

        beta = min(alpha, 0.01 + 0.02 * epoch)
        for img in tqdm(train_dataloader, desc=f"Epoch {epoch+1}", unit="batch", colour="RED"):
            img = img.to(device)

            optimizer.zero_grad()

            img_reconstructed, mu, logvar = model(img)
            loss, reconstructed_loss, kl_loss = vae_loss(img, img_reconstructed, mu, logvar, beta)
            loss.backward()

            optimizer.step()

            total_train_loss += loss.item()
            total_train_reconstructed_loss += reconstructed_loss.item()
            total_train_kl_loss += kl_loss.item()

        avg_train_loss = total_train_loss / len(train_dataloader.dataset)
        avg_train_reconstructed_loss = total_train_reconstructed_loss / len(train_dataloader.dataset)
        avg_train_kl_loss = total_train_kl_loss / len(train_dataloader.dataset)

        avg_val_loss, avg_val_reconstructed_loss, avg_val_kl_loss = val_model(model, val_dataloader, beta, device)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        print(f"    Avg Total Loss        : train = {avg_train_loss:.4f}, val = {avg_val_loss:.4f}")
        print(f"    Avg Reconstructed Loss: train = {avg_train_reconstructed_loss:.4f}, val = {avg_val_reconstructed_loss:.4f}")
        print(f"    Avg KL Loss           : train = {avg_train_kl_loss:.4f}, val = {avg_val_kl_loss:.4f}")

        if avg_val_loss < min_loss:
            min_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(path_to_checkpoint, "best_model.pt"))
            print(f"    Save Model in Epoch {epoch+1}")
        torch.save(model.state_dict(), os.path.join(path_to_checkpoint, "last_model.pt"))

    epochs = range(1, num_epochs + 1)
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, label='Train Loss', marker='o')
    plt.plot(epochs, val_losses, label='Val Loss', marker='o')
    plt.title('Training & Val Loss arcording to Epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(path_to_checkpoint, "loss.png"), dpi=300, bbox_inches='tight')

def main(num_epochs, latent_dim, input_dim, batch_size, lr, alpha, path_to_checkpoint, path_to_data):
    train_dataloader, val_dataloader = prepare_data(path_to_data, input_dim, batch_size)

    model = VAE(latent_dim, input_dim)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = optim.Adam(model.parameters(), lr)

    print(f"Details of training:")
    print(f"    Epochs = {num_epochs}, Latent dim = {latent_dim}, Input dim = {input_dim}")
    print(f"    Batch size = {batch_size}, Alpha = {alpha}, Device = {device}, Save checkpoint = {path_to_checkpoint}")
    print("Ready Training !!!")

    train_model(num_epochs, model, train_dataloader, val_dataloader, optimizer, alpha, device, path_to_checkpoint)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyper-parameters for training")
    parser.add_argument("--epoch", type=int, help="No. of epochs for training", default=50)
    parser.add_argument("--latent_dim", type=int, help="No. dim of latent space", default=1024)
    parser.add_argument("--input_dim", type=int, help="No. channels of images", default=4)
    parser.add_argument("--lr", type=float, help="Learning rate", default=1e-4)
    parser.add_argument("--batch_size", type=int, help="Batch size", default=32)
    parser.add_argument("--alpha", type=float, help="Loss factor", default=3.0)
    parser.add_argument("--path_checkpoint", type=str, help="Save checkpoint path", default="../../checkpoint/")
    parser.add_argument("--path_data", type=str, help="Dir of data", default="../../data/")

    args = parser.parse_args()

    main(
        num_epochs=args.epoch, latent_dim=args.latent_dim,
        input_dim=args.input_dim, batch_size=args.batch_size,
        lr=args.lr, alpha=args.alpha,
        path_to_checkpoint=args.path_checkpoint,
        path_to_data=args.path_data
    )