"""
9_gan_image_generation.py
Comprehensive GAN Pipeline for Image Generation
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

class Generator(nn.Module):
    """DCGAN-style Generator"""
    def __init__(self, latent_dim=100, feature_maps=64, channels=3):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.main = nn.Sequential(
            # Input: latent_dim x 1 x 1
            nn.ConvTranspose2d(latent_dim, feature_maps * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feature_maps * 8),
            nn.ReLU(True),
            # State: (feature_maps*8) x 4 x 4
            nn.ConvTranspose2d(feature_maps * 8, feature_maps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.ReLU(True),
            # State: (feature_maps*4) x 8 x 8
            nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.ReLU(True),
            # State: (feature_maps*2) x 16 x 16
            nn.ConvTranspose2d(feature_maps * 2, feature_maps, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps),
            nn.ReLU(True),
            # State: (feature_maps) x 32 x 32
            nn.ConvTranspose2d(feature_maps, channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # Output: channels x 64 x 64
        )

    def forward(self, x):
        return self.main(x)

class Discriminator(nn.Module):
    """DCGAN-style Discriminator"""
    def __init__(self, feature_maps=64, channels=3):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # Input: channels x 64 x 64
            nn.Conv2d(channels, feature_maps, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # State: (feature_maps) x 32 x 32
            nn.Conv2d(feature_maps, feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # State: (feature_maps*2) x 16 x 16
            nn.Conv2d(feature_maps * 2, feature_maps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # State: (feature_maps*4) x 8 x 8
            nn.Conv2d(feature_maps * 4, feature_maps * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # State: (feature_maps*8) x 4 x 4
            nn.Conv2d(feature_maps * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x).view(-1)

class WGAN_GP_Generator(nn.Module):
    """WGAN-GP Generator for more stable training"""
    def __init__(self, latent_dim=100, feature_maps=64, channels=3):
        super(WGAN_GP_Generator, self).__init__()
        self.latent_dim = latent_dim
        self.main = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, feature_maps * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feature_maps * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_maps * 8, feature_maps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_maps * 2, feature_maps, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_maps, channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)

class WGAN_GP_Discriminator(nn.Module):
    """WGAN-GP Discriminator (Critic)"""
    def __init__(self, feature_maps=64, channels=3):
        super(WGAN_GP_Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(channels, feature_maps, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_maps, feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_maps * 2, feature_maps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_maps * 4, feature_maps * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_maps * 8, 1, 4, 1, 0, bias=False)
            # No sigmoid for WGAN-GP
        )

    def forward(self, x):
        return self.main(x).view(-1)

class GANTrainer:
    """Comprehensive GAN Trainer"""
    
    def __init__(self, generator, discriminator, device='cuda', gan_type='dcgan'):
        self.generator = generator
        self.discriminator = discriminator
        self.device = device
        self.gan_type = gan_type
        
        # Move models to device
        self.generator.to(device)
        self.discriminator.to(device)
        
        # Initialize weights
        self.generator.apply(self.weights_init)
        self.discriminator.apply(self.weights_init)
        
        # Training history
        self.history = {
            'g_losses': [],
            'd_losses': [],
            'real_scores': [],
            'fake_scores': []
        }
    
    def weights_init(self, m):
        """Initialize weights for DCGAN"""
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.batch_norm.bias.data, 0)
    
    def train_dcgan(self, dataloader, epochs, lr=0.0002, beta1=0.5):
        """Train DCGAN"""
        criterion = nn.BCELoss()
        optimizer_g = optim.Adam(self.generator.parameters(), lr=lr, betas=(beta1, 0.999))
        optimizer_d = optim.Adam(self.discriminator.parameters(), lr=lr, betas=(beta1, 0.999))
        
        fixed_noise = torch.randn(64, self.generator.latent_dim, 1, 1, device=self.device)
        
        for epoch in range(epochs):
            for i, (real_imgs, _) in enumerate(tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}')):
                batch_size = real_imgs.size(0)
                real_imgs = real_imgs.to(self.device)
                
                # Labels
                real_labels = torch.ones(batch_size, device=self.device)
                fake_labels = torch.zeros(batch_size, device=self.device)
                
                # Train Discriminator
                optimizer_d.zero_grad()
                
                # Real images
                output_real = self.discriminator(real_imgs)
                loss_real = criterion(output_real, real_labels)
                
                # Fake images
                noise = torch.randn(batch_size, self.generator.latent_dim, 1, 1, device=self.device)
                fake_imgs = self.generator(noise)
                output_fake = self.discriminator(fake_imgs.detach())
                loss_fake = criterion(output_fake, fake_labels)
                
                loss_d = loss_real + loss_fake
                loss_d.backward()
                optimizer_d.step()
                
                # Train Generator
                optimizer_g.zero_grad()
                output_fake = self.discriminator(fake_imgs)
                loss_g = criterion(output_fake, real_labels)
                loss_g.backward()
                optimizer_g.step()
                
                # Save losses
                if i % 50 == 0:
                    self.history['d_losses'].append(loss_d.item())
                    self.history['g_losses'].append(loss_g.item())
                    self.history['real_scores'].append(output_real.mean().item())
                    self.history['fake_scores'].append(output_fake.mean().item())
            
            # Generate sample images
            if epoch % 5 == 0:
                self.generate_and_save_images(epoch, fixed_noise)
                print(f'Epoch [{epoch+1}/{epochs}] Loss_D: {loss_d.item():.4f} Loss_G: {loss_g.item():.4f}')
    
    def train_wgan_gp(self, dataloader, epochs, lr=0.0002, n_critic=5, lambda_gp=10):
        """Train WGAN-GP"""
        optimizer_g = optim.Adam(self.generator.parameters(), lr=lr, betas=(0.5, 0.999))
        optimizer_d = optim.Adam(self.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
        
        fixed_noise = torch.randn(64, self.generator.latent_dim, 1, 1, device=self.device)
        
        for epoch in range(epochs):
            for i, (real_imgs, _) in enumerate(tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}')):
                batch_size = real_imgs.size(0)
                real_imgs = real_imgs.to(self.device)
                
                # Train Discriminator (more times than generator)
                for _ in range(n_critic):
                    optimizer_d.zero_grad()
                    
                    # Real images
                    real_validity = self.discriminator(real_imgs)
                    
                    # Fake images
                    noise = torch.randn(batch_size, self.generator.latent_dim, 1, 1, device=self.device)
                    fake_imgs = self.generator(noise)
                    fake_validity = self.discriminator(fake_imgs)
                    
                    # Gradient penalty
                    gradient_penalty = self.compute_gradient_penalty(real_imgs, fake_imgs)
                    
                    # WGAN loss
                    loss_d = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty
                    loss_d.backward()
                    optimizer_d.step()
                
                # Train Generator
                optimizer_g.zero_grad()
                fake_imgs = self.generator(noise)
                fake_validity = self.discriminator(fake_imgs)
                loss_g = -torch.mean(fake_validity)
                loss_g.backward()
                optimizer_g.step()
                
                # Save losses
                if i % 50 == 0:
                    self.history['d_losses'].append(loss_d.item())
                    self.history['g_losses'].append(loss_g.item())
            
            # Generate sample images
            if epoch % 5 == 0:
                self.generate_and_save_images(epoch, fixed_noise)
                print(f'Epoch [{epoch+1}/{epochs}] Loss_D: {loss_d.item():.4f} Loss_G: {loss_g.item():.4f}')
    
    def compute_gradient_penalty(self, real_imgs, fake_imgs):
        """Compute gradient penalty for WGAN-GP"""
        batch_size = real_imgs.size(0)
        epsilon = torch.rand(batch_size, 1, 1, 1, device=self.device)
        
        # Interpolated images
        interpolated = (epsilon * real_imgs + (1 - epsilon) * fake_imgs).requires_grad_(True)
        interpolated_validity = self.discriminator(interpolated)
        
        # Compute gradients
        gradients = torch.autograd.grad(
            outputs=interpolated_validity,
            inputs=interpolated,
            grad_outputs=torch.ones_like(interpolated_validity),
            create_graph=True,
            retain_graph=True
        )[0]
        
        gradients = gradients.view(batch_size, -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        
        return gradient_penalty
    
    def generate_and_save_images(self, epoch, fixed_noise, save_dir='generated_images'):
        """Generate and save sample images"""
        os.makedirs(save_dir, exist_ok=True)
        
        with torch.no_grad():
            fake_imgs = self.generator(fixed_noise).detach().cpu()
        
        # Save grid of images
        grid = torchvision.utils.make_grid(fake_imgs, normalize=True)
        plt.figure(figsize=(8, 8))
        plt.imshow(grid.permute(1, 2, 0))
        plt.axis('off')
        plt.title(f'Generated Images - Epoch {epoch}')
        plt.savefig(f'{save_dir}/epoch_{epoch}.png')
        plt.close()
    
    def plot_training_history(self):
        """Plot training history"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Losses
        ax1.plot(self.history['d_losses'], label='Discriminator Loss')
        ax1.plot(self.history['g_losses'], label='Generator Loss')
        ax1.set_title('Generator and Discriminator Loss')
        ax1.legend()
        
        # Scores
        ax2.plot(self.history['real_scores'], label='Real Scores')
        ax2.plot(self.history['fake_scores'], label='Fake Scores')
        ax2.set_title('Real and Fake Scores')
        ax2.legend()
        
        plt.tight_layout()
        plt.show()

class ConditionalGAN:
    """Conditional GAN for controlled generation"""
    
    def __init__(self, num_classes, latent_dim=100, feature_maps=64, channels=3):
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        
        self.generator = ConditionalGenerator(num_classes, latent_dim, feature_maps, channels)
        self.discriminator = ConditionalDiscriminator(num_classes, feature_maps, channels)
    
    def generate_with_condition(self, condition, num_images=1):
        """Generate images with specific condition"""
        noise = torch.randn(num_images, self.latent_dim, 1, 1)
        condition_tensor = torch.tensor([condition] * num_images)
        return self.generator(noise, condition_tensor)

class ConditionalGenerator(nn.Module):
    """Conditional Generator with class information"""
    def __init__(self, num_classes, latent_dim=100, feature_maps=64, channels=3):
        super(ConditionalGenerator, self).__init__()
        self.label_embedding = nn.Embedding(num_classes, latent_dim)
        
        self.main = nn.Sequential(
            nn.ConvTranspose2d(latent_dim * 2, feature_maps * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feature_maps * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_maps * 8, feature_maps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_maps * 2, feature_maps, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_maps, channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        label_embedding = self.label_embedding(labels).unsqueeze(2).unsqueeze(3)
        gen_input = torch.cat((noise, label_embedding), 1)
        return self.main(gen_input)

class ConditionalDiscriminator(nn.Module):
    """Conditional Discriminator with class information"""
    def __init__(self, num_classes, feature_maps=64, channels=3):
        super(ConditionalDiscriminator, self).__init__()
        self.label_embedding = nn.Embedding(num_classes, channels * 64 * 64)
        
        self.main = nn.Sequential(
            nn.Conv2d(channels * 2, feature_maps, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_maps, feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_maps * 2, feature_maps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_maps * 4, feature_maps * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_maps * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        label_embedding = self.label_embedding(labels).view(img.size(0), 3, 64, 64)
        disc_input = torch.cat((img, label_embedding), 1)
        return self.main(disc_input).view(-1)

# Example usage
if __name__ == "__main__":
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create GAN models
    latent_dim = 100
    generator = Generator(latent_dim=latent_dim)
    discriminator = Discriminator()
    
    # Create trainer
    gan_trainer = GANTrainer(generator, discriminator, device=device)
    
    print("GAN Pipeline Ready!")
    print("Usage:")
    print("1. Prepare your image dataset")
    print("2. Create DataLoader")
    print("3. Call gan_trainer.train_dcgan(dataloader, epochs=100)")
    print("4. Generate images with generator(torch.randn(batch_size, latent_dim, 1, 1))")
