from vae import VAE
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
from PIL import Image

import matplotlib.pyplot as plt
import argparse
import os
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

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
    reconstructed_loss = F.mse_loss(x_reconstructed, x, reduction="mean") * x.numel() / x.size(0)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    loss = reconstructed_loss + beta * kl_loss
    # loss = reconstructed_loss * beta + kl_loss
    return loss, reconstructed_loss, kl_loss


def save_generated_images(decoder, epoch, path_to_output, num_images, latent_dim, device):
    epoch_folder = os.path.join(path_to_output, f"epoch_{epoch}")
    os.makedirs(epoch_folder, exist_ok=True)

    z = torch.randn(num_images, latent_dim).to(device)

    decoder.eval()

    with torch.no_grad():
        generated_images = decoder(z).cpu()

    for i, img in enumerate(generated_images):
        save_path = os.path.join(epoch_folder, f"generated_{i+1}.png")
        save_image(img, save_path)

def val_model(model, val_dataloader, beta, device):
    model.eval()

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
def train_model(num_epochs, model, train_dataloader, val_dataloader, optimizer, alpha, device, step, path_to_checkpoint, path_to_output, pretrained):
    os.makedirs(path_to_checkpoint, exist_ok=True)
    os.makedirs(path_to_output, exist_ok=True)

    model.to(device)

    beta = alpha

    min_loss = float("inf")
    start_epoch = 0

    if pretrained:
        checkpoint_model = torch.load(pretrained)
        model.load_state_dict(checkpoint_model["model_state_dict"])
        optimizer.load_state_dict(checkpoint_model["optimizer_state_dict"])
        start_epoch = checkpoint_model["epoch"] + 1
        min_loss = checkpoint_model["min_loss"]
        print(f"Model has been trained {start_epoch} epochs before !!!")

    train_losses = []
    val_losses = []

    for epoch in range(start_epoch, start_epoch+num_epochs):
        model.train()

        total_train_loss = 0
        total_train_reconstructed_loss = 0
        total_train_kl_loss = 0

        # beta = max(alpha-0.02*epoch, 0.02)

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
            best_model_path = os.path.join(path_to_checkpoint, "best_model.pt")
            torch.save(model.state_dict(), best_model_path)
            print(f"    Save Model in Epoch {epoch+1}")

        last_model_path = os.path.join(path_to_checkpoint, "last_model.pt")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "min_loss": min_loss
        }, last_model_path)

        if (epoch+1) % step == 0:
            save_generated_images(model.decoder, epoch+1, path_to_output, 10, model.latent_dim, device)

    epochs = range(start_epoch, start_epoch+num_epochs)
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, label='Train Loss', marker='o')
    plt.plot(epochs, val_losses, label='Val Loss', marker='o')
    plt.title('Training & Val Loss arcording to Epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(path_to_output, f"loss_At{start_epoch+num_epochs}.png"), dpi=300, bbox_inches='tight')

def main(num_epochs, latent_dim, input_dim, batch_size, lr, alpha, step, path_to_checkpoint, path_to_data, path_to_output, pretrained):
    train_dataloader, val_dataloader = prepare_data(path_to_data, input_dim, batch_size)

    model = VAE(latent_dim, input_dim)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = optim.Adam(model.parameters(), lr)

    print(f"Details of training:")
    print(f"    Epochs = {num_epochs}, Latent dim = {latent_dim}, Input dim = {input_dim}")
    print(f"    Batch size = {batch_size}, Alpha = {alpha}, Device = {device}, Save checkpoint = {path_to_checkpoint}")
    print("Ready Training !!!")

    train_model(num_epochs, model, train_dataloader, val_dataloader, optimizer, alpha, device, step, path_to_checkpoint, path_to_output, pretrained)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyper-parameters for training")
    parser.add_argument("--epoch", type=int, help="No. of epochs for training", default=50)
    parser.add_argument("--latent_dim", type=int, help="No. dim of latent space", default=1024)
    parser.add_argument("--input_dim", type=int, help="No. channels of images", default=4)
    parser.add_argument("--lr", type=float, help="Learning rate", default=1e-4)
    parser.add_argument("--batch_size", type=int, help="Batch size", default=32)
    parser.add_argument("--alpha", type=float, help="Loss factor", default=5.0)
    parser.add_argument("--step", type=int, help="Step of generative in training mode", default=100)
    parser.add_argument("--path_checkpoint", type=str, help="Save checkpoint path", default="../../checkpoint/")
    parser.add_argument("--path_data", type=str, help="Dir of data", default="../../data/")
    parser.add_argument("--path_output", type=str, help="Dir of output", default="../../output/")
    parser.add_argument("--pretrained", type=str, help="Checkpoint to continue training", default=None)

    args = parser.parse_args()

    main(
        num_epochs=args.epoch, latent_dim=args.latent_dim,
        input_dim=args.input_dim, batch_size=args.batch_size,
        lr=args.lr, alpha=args.alpha,
        path_to_checkpoint=args.path_checkpoint,
        path_to_data=args.path_data, step=args.step,
        path_to_output=args.path_output,
        pretrained=args.pretrained
    )