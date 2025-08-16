import torch
import torch.utils
import torch.utils.data
from tqdm.auto import tqdm
from torch import nn
import argparse
import torch.nn.functional as F
import utils
import dataset
import json
import os
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import my_utils

os.environ["CUDA_VISIBLE_DEVICES"] = "4"
device = None

class NoiseScheduler():
    """
    Noise scheduler for the DDPM model

    Args:
        num_timesteps: int, the number of timesteps
        type: str, the type of scheduler to use
        **kwargs: additional arguments for the scheduler

    This object sets up all the constants like alpha, beta, sigma, etc. required for the DDPM model

    """
    def __init__(self, num_timesteps=50, type="linear", **kwargs):
        self.num_timesteps = num_timesteps
        self.type = type
        if type == "linear":
            self.init_linear_schedule(**kwargs)
        elif type == "cosine":
            self.init_cosine_schedule()
        elif type == "sigmoid":
            self.init_sigmoid_schedule()
        else:
            raise NotImplementedError(f"{type} scheduler is not implemented") # change this if you implement additional schedulers

        self.sqrt_alpha = torch.sqrt(self.alpha)
        self.sqrt_alpha_hat = torch.sqrt(self.alpha_hat)
        self.sqrt_one_minus_alpha_hat = torch.sqrt(1. - self.alpha_hat)
        self.sampling_variance = self.beta[1:] * (1. - self.alpha_hat[:-1]) / (1. - self.alpha_hat[1:])
        self.sampling_variance = torch.cat([torch.tensor([0.0]), self.sampling_variance]) 

    def init_linear_schedule(self, beta_start=1e-4, beta_end=2e-2):
        """
        Precompute whatever quantities are required for training and sampling
        """
        self.beta = torch.linspace(beta_start, beta_end, self.num_timesteps, dtype=torch.float32)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def init_cosine_schedule(self, s=0.008, clip_min=1e-9, clip_max=0.999):
        T = self.num_timesteps
        t = torch.linspace(0, T, T, dtype=torch.float32) / T

        def cosine_schedule(t):
            f_t = (torch.cos(((t + s) / (1 + s)) * (math.pi / 2))) ** 2
            f_0 = torch.cos(torch.tensor((s / (1 + s)) * (math.pi / 2), dtype=torch.float32)) ** 2
            return torch.clamp(f_t / f_0, clip_min, clip_max)

        gamma_t = cosine_schedule(t)
        self.alpha_hat = gamma_t

        self.alpha_hat = torch.clamp(self.alpha_hat, clip_min, clip_max)
        self.alpha = torch.ones_like(self.alpha_hat)
        for i in range(1, len(self.alpha)):
            self.alpha[i] = torch.clamp(self.alpha_hat[i] / self.alpha_hat[i - 1], clip_min, clip_max)
        self.beta = torch.clamp(1. - self.alpha, clip_min, clip_max)


    def init_sigmoid_schedule(self, start=-3, end=3, tau=1.0, clip_min=1e-9, clip_max=0.999):
        t = torch.linspace(0, 1, self.num_timesteps)

        def sigmoid_schedule(t):
            v_start = 1 / (1 + np.exp(-start / tau))
            v_end = 1 / (1 + np.exp(-end / tau))
            output = 1 / (1 + np.exp(-((t * (end - start) + start) / tau)))
            return np.clip((v_end - output) / (v_end - v_start), clip_min, 1.0)

        gamma_t = torch.tensor([sigmoid_schedule(ti.item()) for ti in t], dtype=torch.float32)
        self.alpha_hat = gamma_t

        self.alpha_hat = torch.clamp(self.alpha_hat, clip_min, clip_max)

        self.alpha = torch.ones_like(self.alpha_hat)
        for i in range(1, len(self.alpha)):
            denominator = max(self.alpha_hat[i-1], clip_min)
            self.alpha[i] = torch.clamp(self.alpha_hat[i] / denominator, clip_min, 1.0)

        self.beta = 1. - self.alpha

    def __len__(self):
        return self.num_timesteps

class Model(nn.Module):
    def __init__(self, data_dim, time_dim, label_dim=0):
        super(Model,self).__init__()
        self.ff = nn.Sequential(
            nn.Linear(data_dim + label_dim + time_dim, 64),
            nn.GELU(),
            nn.Linear(64,128),
            nn.GELU(),
            nn.Linear(128,512),
            nn.GELU(),
            nn.Linear(512,128),
            nn.GELU(),
            nn.Linear(128,64),
            nn.GELU(),
            nn.Linear(64, data_dim)
        )
    def forward(self, x, t):
        x = torch.cat((x, t), dim=1)        
        return self.ff(x)

class DDPM(nn.Module):
    def __init__(self, n_dim=3, n_steps=200,time_dim=4):
        """
        Noise prediction network for the DDPM

        Args:
            n_dim: int, the dimensionality of the data
            n_steps: int, the number of steps in the diffusion process
        We have separate learnable modules for `time_embed` and `model`. `time_embed` can be learned or a fixed function as well

        """
        super(DDPM,self).__init__()
        self.data_dim = n_dim
        self.time_dim = time_dim
        self.time_embedding = nn.Sequential(
                nn.Linear(1,16),
                nn.Linear(16, 4),
                nn.ReLU()
            )
        self.model = Model(self.data_dim, self.time_dim)

    def forward(self, x, t, num_steps):
        """
        Args:
            x: torch.Tensor, the input data tensor [batch_size, n_dim]
            t: torch.Tensor, the timestep tensor [batch_size]

        Returns:
            torch.Tensor, the predicted noise tensor [batch_size, n_dim]
        """
        t = t / num_steps
        time_embed = self.time_embedding(t.view(-1, 1))
        return self.model(x, time_embed)


class ConditionalDDPM(DDPM):
    def __init__(self, n_classes=2, n_dim=3 ,n_steps=200,label_dim=1, time_dim=4):
        """
        Noise prediction network for the DDPM

        Args:
            n_dim: int, the dimensionality of the data
            n_steps: int, the number of steps in the diffusion process
        We have separate learnable modules for `time_embed` and `model`. `time_embed` can be learned or a fixed function as well

        """
        super(ConditionalDDPM, self).__init__(n_dim=n_dim, n_steps=n_steps,time_dim=time_dim)
        self.label_dim = label_dim
        self.num_classes = n_classes
        self.model = Model(self.data_dim, self.time_dim, self.label_dim)
        self.label_embed = nn.Embedding(self.num_classes, self.label_dim)
    
    def forward(self, x, t, y, num_steps):
        """
        Args:
            x: torch.Tensor, the input data tensor [batch_size, n_dim]
            t: torch.Tensor, the timestep tensor [batch_size]

        Returns:
            torch.Tensor, the predicted noise tensor [batch_size, n_dim]
        """
        t = t / num_steps
        time_embed = self.time_embedding(t.view(-1, 1))
        y_mask = (y != -1)
        y_embed = torch.zeros((x.shape[0], self.label_dim), device=x.device)
    
        valid_y = y[y_mask]
        y_embed[y_mask] = self.label_embed(valid_y)

        input = torch.cat((x, y_embed), dim=1)         
        return self.model(input , time_embed)
    
class ClassifierDDPM():
    """
    ClassifierDDPM implements a classification algorithm using the DDPM model
    """
    
    def __init__(self, model: ConditionalDDPM, noise_scheduler: NoiseScheduler):
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.num_classes = model.num_classes
        self.device = next(model.parameters()).device
        self.num_trials = 10

    def __call__(self, x):
        return self.predict(x)

    def predict(self, x):
        """
        Args:
            x: torch.Tensor, the input data tensor [batch_size, n_dim]
        Returns:
            torch.Tensor, the predicted class tensor [batch_size]
        """

        x = x.to(self.device)
        probs = self.predict_proba(x)
        return torch.argmax(probs, dim=1)

    def predict_proba(self, x):
        """
        Args:
            x: torch.Tensor, the input data tensor [batch_size, n_dim]
        Returns:
            torch.Tensor, the predicted probabilites for each class  [batch_size, n_classes]
        """
        batch_x = x.to(self.device)
        batch_size = batch_x.shape[0]
        errors = torch.zeros((batch_size, self.num_classes), device=self.device)
        for _ in range(self.num_trials):
            time_steps = torch.randint(1, self.noise_scheduler.num_timesteps, (batch_size,), device=self.device)
            noisy_batch, noise_true = noisy_input(batch_x, noise_scheduler, time_steps)
            noisy_batch, noise_true = noisy_batch.to(self.device), noise_true.to(self.device)

            for c in range(self.num_classes):
                class_labels = torch.full((batch_size,), c, device=self.device)
                predicted_noise = self.model(noisy_batch, class_labels, time_steps, self.noise_scheduler.num_timesteps)
                mse = torch.mean((noise_true - predicted_noise) ** 2, dim=1)
                errors[:, c] += mse

        errors /= self.num_trials
        probs = F.softmax(-errors, dim=1)
        return probs

def noisy_input(batch, noise_scheduler, time_steps):
    epsilon = torch.randn_like(batch, device=batch.device)
    sqrt_alpha = noise_scheduler.sqrt_alpha_hat[time_steps].view(-1, 1).to(device)
    sqrt_one_minus_alpha = noise_scheduler.sqrt_one_minus_alpha_hat[time_steps].view(-1, 1).to(device)
    return batch * sqrt_alpha + epsilon * sqrt_one_minus_alpha, epsilon

def train(model, noise_scheduler, dataloader, optimizer, epochs, run_name):
    """
    Train the model and save the model and necessary plots

    Args:
        model: DDPM, model to train
        noise_scheduler: NoiseScheduler, scheduler for the noise
        dataloader: torch.utils.data.DataLoader, dataloader for the dataset
        optimizer: torch.optim.Optimizer, optimizer to use
        epochs: int, number of epochs to train the model
        run_name: str, path to save the model
    """
    num_batch = len(dataloader)
    criterion = nn.MSELoss()
    best_loss = torch.inf
    best_iter = 0
    for i in range(epochs):
        total_loss = 0
        for batch in tqdm(dataloader):
            batch_X = batch[0].to(device)
            batch_size = batch_X.shape[0]
            time_steps = torch.randint(1, noise_scheduler.num_timesteps, (batch_size,)).to(device) 
            noisy_batch, noise_true = noisy_input(batch_X, noise_scheduler, time_steps)
            noisy_batch, noise_true = noisy_batch.to(device), noise_true.to(device)
            noise_pred = model.forward(noisy_batch, time_steps, noise_scheduler.num_timesteps)
            loss = criterion(noise_true,noise_pred)
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        avg_loss = total_loss/num_batch
        if avg_loss < best_loss:
            best_iter = i
            best_loss = avg_loss
            torch.save(model.state_dict(),f"{run_name}/model.pth")
        print(f"Epoch: {i} Loss: {total_loss/num_batch}")
    print(f"Best Loss: {best_loss} at iteration {best_iter}")
        

@torch.no_grad()
def sample(model, n_samples, noise_scheduler, return_intermediate=False): 
    """
    Sample from the model
    
    Args:
        model: DDPM
        n_samples: int
        noise_scheduler: NoiseScheduler
        return_intermediate: bool
    Returns:
        torch.Tensor, samples from the model [n_samples, n_dim]

    If `return_intermediate` is `False`,
            torch.Tensor, samples from the model [n_samples, n_dim]
    Else
        the function returns all the intermediate steps in the diffusion process as well 
        Return: [[n_samples, n_dim]] x n_steps
        Optionally implement return_intermediate=True, will aid in visualizing the intermediate steps
    """
    n_dim = model.data_dim
    device = next(model.parameters()).device
    x = torch.randn((n_samples, n_dim), device=device)
    intermediate_steps = [x.clone()] if return_intermediate else None
    for t in range(noise_scheduler.num_timesteps - 1, 0, -1):
        time = torch.full((n_samples,), t, device=device)
        predicted_noise = model(x, time)
        alpha_t = noise_scheduler.alpha[t].to(device)  
        sqrt_alpha_t = noise_scheduler.sqrt_alpha[t].to(device)      
        sqrt_one_minus_alpha_hat_t = noise_scheduler.sqrt_one_minus_alpha_hat[t].to(device)
        noise = torch.zeros_like(x) if t == 1 else torch.randn_like(x)
        mu = (1 / sqrt_alpha_t) * (x - ((1 - alpha_t) / sqrt_one_minus_alpha_hat_t) * predicted_noise)
        x = mu + noise_scheduler.sampling_variance[t].to(device) * noise
        if return_intermediate:
            intermediate_steps.append(x.clone())
    return intermediate_steps if return_intermediate else x

def trainConditional(model, noise_scheduler, dataloader, optimizer, epochs, run_name, p_uncond = 0.1):
    num_batch = len(dataloader)
    criterion = nn.MSELoss()
    best_loss = torch.inf
    best_iter = 0
    for i in range(epochs):
        total_loss = 0
        for batch_X, batch_y in tqdm(dataloader):
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            mask = torch.rand(batch_y.shape, device=batch_y.device) < p_uncond
            batch_y[mask] = -1
            batch_size = batch_X.shape[0]
            time_steps = torch.randint(1, noise_scheduler.num_timesteps, (batch_size,)).to(device) 
            noisy_batch, noise_true = noisy_input(batch_X, noise_scheduler, time_steps)
            noisy_batch, noise_true = noisy_batch.to(device), noise_true.to(device)
            noise_pred = model.forward(noisy_batch, time_steps, batch_y, noise_scheduler.num_timesteps)
            loss = criterion(noise_true, noise_pred)
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        avg_loss = total_loss/num_batch
        if avg_loss < best_loss:
            best_iter = i
            best_loss = avg_loss
            torch.save(model.state_dict(),f"{run_name}/model.pth")
        print(f"Epoch: {i} Loss: {total_loss/num_batch}")
    print(f"Best Loss: {best_loss} at iteration {best_iter}")

@torch.no_grad()
def sampleConditional(model, n_samples, noise_scheduler, label, return_intermediate=False):
    n_dim = model.data_dim
    device = next(model.parameters()).device
    x = torch.randn((n_samples, n_dim)).to(device)
    label = torch.tensor([label], device=device).repeat(n_samples)

    intermediate_steps = [x.clone()] if return_intermediate else None
    for t in range(noise_scheduler.num_timesteps - 1, 0, -1):
        time = torch.full((n_samples,), t, device=device)
        predicted_noise = model(x, label,  time, noise_scheduler.num_timesteps)
        alpha_t = noise_scheduler.alpha[t].to(device)  
        sqrt_alpha_t = noise_scheduler.sqrt_alpha[t].to(device)      
        sqrt_one_minus_alpha_hat_t = noise_scheduler.sqrt_one_minus_alpha_hat[t].to(device)
        noise = torch.zeros_like(x) if t == 1 else torch.randn_like(x)
        mu = (1 / sqrt_alpha_t) * (x - ((1 - alpha_t) / sqrt_one_minus_alpha_hat_t) * predicted_noise)
        x = mu + noise_scheduler.sampling_variance[t].to(device) * noise
        if return_intermediate:
            intermediate_steps.append(x.clone())
    return intermediate_steps if return_intermediate else x

@torch.no_grad()
def sampleCFG(model, n_samples, noise_scheduler, guidance_scale, class_label):
    """
    Sample from the conditional model
    
    Args:
        model: ConditionalDDPM
        n_samples: int
        noise_scheduler: NoiseScheduler
        guidance_scale: float
        class_label: int

    Returns:
        torch.Tensor, samples from the model [n_samples, n_dim]
    """
    n_dim = model.data_dim
    device = next(model.parameters()).device
    x = torch.randn((n_samples, n_dim), device=device)
    
    cond_label = torch.tensor([class_label], device=device).repeat(n_samples)
    uncond_label = torch.tensor([-1], device=device).repeat(n_samples)

    for t in range(noise_scheduler.num_timesteps - 1, 0, -1):
        time = torch.full((n_samples,), t, device=device)
        predicted_noise_cond = model(x, cond_label, time, noise_scheduler.num_timesteps)
        predicted_noise_uncond = model(x, uncond_label, time, noise_scheduler.num_timesteps)
        
        predicted_noise = predicted_noise_uncond + guidance_scale * (predicted_noise_cond - predicted_noise_uncond)

        alpha_t = noise_scheduler.alpha[t].to(device)  
        sqrt_alpha_t = noise_scheduler.sqrt_alpha[t].to(device)      
        sqrt_one_minus_alpha_hat_t = noise_scheduler.sqrt_one_minus_alpha_hat[t].to(device)
        noise = torch.zeros_like(x) if t == 1 else torch.randn_like(x)
        mu = (1 / sqrt_alpha_t) * (x - ((1 - alpha_t) / sqrt_one_minus_alpha_hat_t) * predicted_noise)
        x = mu + noise_scheduler.sampling_variance[t].to(device) * noise
    return x

@torch.no_grad()
def sampleSVDD(model, n_samples, noise_scheduler, reward_scale, reward_fn):
    """
    Sample from the SVDD model

    Args:
        model: ConditionalDDPM
        n_samples: int
        noise_scheduler: NoiseScheduler
        reward_scale: float
        reward_fn: callable, takes in a batch of samples torch.Tensor:[n_samples, n_dim] and returns torch.Tensor[n_samples]

    Returns:
        torch.Tensor, samples from the model [n_samples, n_dim]
    """
    n_dim = model.data_dim
    device = next(model.parameters()).device
    x_t = torch.randn((n_samples, n_dim)).to(device)

    for t in range(noise_scheduler.num_timesteps - 1, 0, -1):
        time_t = torch.full((n_samples,), t, device=device)

        all_candidates = []
        rewards = [] 

        for _ in range(15):
            predicted_noise = model(x_t, time_t, noise_scheduler.num_timesteps)        
            alpha_t = noise_scheduler.alpha[t].to(device)  
            sqrt_alpha_t = noise_scheduler.sqrt_alpha[t].to(device)      
            sqrt_one_minus_alpha_hat_t = noise_scheduler.sqrt_one_minus_alpha_hat[t].to(device)
            sqrt_alpha_hat_t = noise_scheduler.sqrt_alpha_hat[t].to(device)   
            noise = torch.zeros_like(x_t) if t == 1 else torch.randn_like(x_t)
            mu = (1 / sqrt_alpha_t) * (x_t - ((1 - alpha_t) / sqrt_one_minus_alpha_hat_t) * predicted_noise)
            x_t_minus_1 = mu + noise_scheduler.sampling_variance[t].to(device) * noise
            x_0_hat = (x_t - sqrt_one_minus_alpha_hat_t * predicted_noise) / sqrt_alpha_hat_t
            all_candidates.append(x_t_minus_1)
            rewards.append(reward_fn(x_0_hat))

        all_candidates = torch.stack(all_candidates, dim=1)
        rewards = torch.stack(rewards, dim=1)
        if reward_scale <= 1e-5:
            best_indices = torch.argmax(rewards, dim=1)
        else:
            weights = torch.exp(rewards / reward_scale)
            probs = weights / weights.sum(dim=1, keepdim=True)
            dist = torch.distributions.Categorical(probs=probs)
            best_indices = dist.sample()
        row_indices = torch.arange(n_samples, device=device)
        x_t = all_candidates[row_indices, best_indices]
    return x_t
    
def generate_reward_function(classifier, target_label):
    """
    Generates a reward function for scikit-learn's MLPClassifier.
    Args:
    classifier: MLPClassifier (from sklearn)
    target_label: int, the desired class label
    Returns:
    Callable reward function that takes torch.Tensor:[n_samples, n_dim]
    and returns torch.Tensor:[n_samples].
    """
    def reward_fn(samples):
        samples_np = samples.cpu().numpy()
        probs = classifier.predict_proba(samples_np)
        
        # Apply log transformation to emphasize probabilities closer to 1
        # and heavily penalize very low probabilities
        # log_probs = np.log(probs[:, target_label] + 1e-5)
        
        # Square the probabilities to further emphasize high confidence predictions
        # squared_probs = probs[:, target_label] ** 2
        
        # Use margin between target class and next highest class
        margins = probs[:, target_label] - np.max(np.delete(probs, target_label, axis=1), axis=1)
        
        rewards = torch.tensor(margins, device=samples.device)
        return rewards
    
    return reward_fn

def plot_samples(samples, labels, save_path):
    """
    Plots 2D samples with different colors for each unique label.
    
    Args:
        samples (torch.Tensor or np.array): Sample points, shape (N, 2)
        labels (torch.Tensor): Corresponding labels for each sample
    """
    samples = samples.cpu().numpy()
    labels = labels.cpu().numpy()
    unique_labels = np.unique(labels)
    colors = plt.cm.get_cmap("tab10", len(unique_labels))
    plt.figure(figsize=(8, 6))
    for i, label in enumerate(unique_labels):
        mask = labels == label
        plt.scatter(samples[mask, 0], samples[mask, 1], label=f"Class {label}", alpha=0.6, color=colors(i))
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title(f"2D Scatter Plot of {len(samples)} Points")
    plt.legend(title="Class Labels")
    plt.grid(True)
    plt.savefig(save_path)
    plt.show()

def evaluate_sample_quality_per_class(samples, labels, classifier, unique_labels):
    samples_np = samples.cpu().detach().numpy()
    labels_np = labels.cpu().numpy()
    
    predicted_labels = classifier.predict(samples_np)
    overall_accuracy = (predicted_labels == labels_np).mean()
    
    class_accuracies = {}
    for label in unique_labels:
        label = int(label.item())
        label_mask = (labels_np == label)
        if label_mask.sum() > 0:
            class_accuracy = (predicted_labels[label_mask] == label).mean()
            class_accuracies[label] = class_accuracy
    
    return overall_accuracy, class_accuracies

def experiment_reward_scales(model, noise_scheduler, unique_labels, reward_scales, n_samples_per_class, classifier, run_name):
    """
    Run experiments with different reward scales and report per-class accuracies

    Args:
        model: ConditionalDDPM
        noise_scheduler: NoiseScheduler
        unique_labels: list of unique class labels
        reward_scales: list of reward scale values to try
        n_samples_per_class: number of samples to generate per class
        classifier: trained classifier for evaluation

    Returns:
        results: dict mapping reward scales to overall and per-class accuracies
    """
    results = {}
    data_X, data_y = dataset.load_dataset(args.dataset)
    data_X_np = data_X.cpu().numpy()
    data_y_np = data_y.cpu().numpy()
    train_preds = classifier.predict(data_X_np)
    train_accuracy = (train_preds == data_y_np).mean()

    class_train_accuracies = {}
    for label in unique_labels:
        label = int(label.item())
        label_mask = (data_y_np == label)
        class_train_accuracy = (train_preds[label_mask] == label).mean()
        class_train_accuracies[label] = class_train_accuracy

    results['training_data'] = {
        'overall_accuracy': train_accuracy,
        'class_accuracies': class_train_accuracies
    }

    for reward_scale in reward_scales:
        all_samples = []
        all_labels = []

        for label in unique_labels:
            label = int(label.item())
            reward_fn = generate_reward_function(classifier, label)
            samples = sampleSVDD(model, n_samples_per_class, noise_scheduler, reward_scale, reward_fn)
            labels = torch.full((len(samples),), label, dtype=torch.long)

            all_samples.append(samples)
            all_labels.append(labels)

        all_samples = torch.cat(all_samples, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        overall_accuracy, class_accuracies = evaluate_sample_quality_per_class(
            all_samples, all_labels, classifier, unique_labels
        )

        results[reward_scale] = {
            'overall_accuracy': overall_accuracy,
            'class_accuracies': class_accuracies
        }

        plot_samples(all_samples, all_labels, save_path=f"{run_name}/reward_scale_{reward_scale}.png")

    return results

def experiment_guidance_scales(model, noise_scheduler, unique_labels, guidance_scales, n_samples_per_class, classifier, run_name):
    """
    Run experiments with different guidance scales and report per-class accuracies
    
    Args:
        model: ConditionalDDPM
        noise_scheduler: NoiseScheduler
        unique_labels: list of unique class labels
        guidance_scales: list of guidance scale values to try
        n_samples_per_class: number of samples to generate per class
        classifier: trained classifier for evaluation
        
    Returns:
        results: dict mapping guidance scales to overall and per-class accuracies
    """
    results = {}    
    data_X, data_y = dataset.load_dataset(args.dataset)
    data_X_np = data_X.cpu().numpy()
    data_y_np = data_y.cpu().numpy()
    train_preds = classifier.predict(data_X_np)
    train_accuracy = (train_preds == data_y_np).mean()
    
    class_train_accuracies = {}
    for label in unique_labels:
        label = int(label.item())
        label_mask = (data_y_np == label)
        class_train_accuracy = (train_preds[label_mask] == label).mean()
        class_train_accuracies[label] = class_train_accuracy
    
    results['training_data'] = {
        'overall_accuracy': train_accuracy,
        'class_accuracies': class_train_accuracies
    }
    
    for guidance_scale in guidance_scales:
        all_samples = []
        all_labels = []
        
        for label in unique_labels:
            label = int(label.item())
            samples = sampleCFG(model, n_samples_per_class, noise_scheduler, guidance_scale=guidance_scale, label)
            labels = torch.full((len(samples),), label, dtype=torch.long)
            
            all_samples.append(samples)
            all_labels.append(labels)
        
        all_samples = torch.cat(all_samples, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        overall_accuracy, class_accuracies = evaluate_sample_quality_per_class(
            all_samples, all_labels, classifier, unique_labels
        )
        
        results[guidance_scale] = {
            'overall_accuracy': overall_accuracy,
            'class_accuracies': class_accuracies
        }
        
        plot_samples(all_samples, all_labels, save_path=f"{run_name}/cfg_scale_{guidance_scale}.png")
    
    return results

def compare_classifiers(run_name, model, noise_scheduler, clf, test_X, test_y, device):
    """
    Compare the diffusion-based classifier with a traditional classifier
    
    Args:
        run_name: str, the name of the run
        model: ConditionalDDPM, the trained conditional diffusion model
        noise_scheduler: NoiseScheduler, the noise scheduler
        clf: sklearn classifier, the trained traditional classifier
        test_X: torch.Tensor, the test data
        test_y: torch.Tensor, the test labels
        device: str, the device to use
        
    Returns:
        dict, the results of the comparison
    """
    diff_clf = ClassifierDDPM(model, noise_scheduler)
    
    test_X_np = test_X.cpu().numpy()
    test_y_np = test_y.cpu().numpy()
    
    trad_preds = clf.predict(test_X_np)
    batch_size = 64
    num_samples = len(test_X)
    diff_preds = []
    
    for i in range(0, num_samples, batch_size):
        end_idx = min(i + batch_size, num_samples)
        batch = test_X[i:end_idx].to(device)
        with torch.no_grad():
            batch_preds = diff_clf.predict(batch)
        diff_preds.append(batch_preds.cpu())
    
    diff_preds = torch.cat(diff_preds).numpy()
    
    trad_acc = accuracy_score(test_y_np, trad_preds)
    diff_acc = accuracy_score(test_y_np, diff_preds)
    
    print(f"Traditional classifier accuracy: {trad_acc:.4f}")
    print(f"Diffusion classifier accuracy: {diff_acc:.4f}")
    
    trad_report = classification_report(test_y_np, trad_preds, output_dict=True)
    diff_report = classification_report(test_y_np, diff_preds, output_dict=True)
    
    trad_cm = confusion_matrix(test_y_np, trad_preds).tolist()
    diff_cm = confusion_matrix(test_y_np, diff_preds).tolist()
    
    results = {
        "traditional_classifier": {
            "accuracy": float(trad_acc),
            "classification_report": trad_report,
            "confusion_matrix": trad_cm
        },
        "diffusion_classifier": {
            "accuracy": float(diff_acc),
            "classification_report": diff_report,
            "confusion_matrix": diff_cm
        }
    }
    
    with open(f'{run_name}/classifier_comparison_results.json', 'w') as f:
        json.dump(results, f, indent=2)    
        
    plt.figure(figsize=(18, 6))
    
    plt.subplot(1, 3, 1)
    unique_labels = np.unique(test_y_np)
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        mask = test_y_np == label
        plt.scatter(test_X_np[mask, 0], test_X_np[mask, 1], c=[colors[i]], label=f'Class {label}', alpha=0.6)
    plt.title('Ground Truth')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    for i, label in enumerate(unique_labels):
        mask = trad_preds == label
        plt.scatter(test_X_np[mask, 0], test_X_np[mask, 1], c=[colors[i]], label=f'Class {label}', alpha=0.6)
    plt.title(f'Traditional Classifier (Acc: {trad_acc:.4f})')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    for i, label in enumerate(unique_labels):
        mask = diff_preds == label
        plt.scatter(test_X_np[mask, 0], test_X_np[mask, 1], c=[colors[i]], label=f'Class {label}', alpha=0.6)
    plt.title(f'Diffusion Classifier (Acc: {diff_acc:.4f})')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{run_name}/classifier_comparison_visualization.png', dpi=300)
    return results

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=['train', 'sample', 'cfg_experiment', 'classifier_experiment', 'svdd_experiment'], default='sample')
    parser.add_argument("--n_steps", type=int, default=None)
    parser.add_argument("--lbeta", type=float, default=None)
    parser.add_argument("--ubeta", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--n_samples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--dataset", type=str, default = None)
    parser.add_argument("--seed", type=int, default = 42)
    parser.add_argument("--n_dim", type=int, default = None)
    parser.add_argument("--schedule", type=str, default = "linear")
    parser.add_argument("--guidance_scales", type=str, default="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2")

    args = parser.parse_args()
    utils.seed_everything(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    type_="cfg"
    run_name = f'exps/ddpm_{args.n_dim}_{args.n_steps}_{args.lbeta}_{args.ubeta}_{args.dataset}_{type_}'

    os.makedirs(run_name, exist_ok=True)
    data_X, data_y = dataset.load_dataset(args.dataset)
    unique_labels = torch.unique(data_y)
    n_c = unique_labels.numel()
    model = ConditionalDDPM(n_dim=args.n_dim, n_classes=n_c,label_dim=4,time_dim=4)
    # model = DDPM(n_dim=args.n_dim, time_dim=4)
    noise_scheduler = NoiseScheduler(num_timesteps=args.n_steps, beta_start=args.lbeta, beta_end=args.ubeta, type=args.schedule)
    model = model.to(device)

    if args.mode == 'train':
        epochs = args.epochs
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        data_X = data_X.to(device)
        # dataloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(data_X), batch_size=args.batch_size, shuffle=True, drop_last=True)
        # train(model, noise_scheduler, dataloader, optimizer, epochs, run_name)
        data_y = data_y.to(device)
        dataloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(data_X, data_y), batch_size=args.batch_size, shuffle=True, drop_last=True)
        trainConditional(model, noise_scheduler, dataloader, optimizer, epochs, run_name)

    elif args.mode == 'sample':
        model.load_state_dict(torch.load(f'{run_name}/model.pth'))
        
        all_samples = []
        all_labels = []
        
        for label in unique_labels:
            label = int(label.item())
            if type_ == "cfg":
                samples = sampleCFG(model, args.n_samples, noise_scheduler, guidance_scale=1.0,label)
            else:
                samples = sampleConditional(model, args.n_samples, noise_scheduler, label)
            labels = torch.full((len(samples),), label, dtype=torch.long)
            
            all_samples.append(samples)
            all_labels.append(labels)

        all_samples = torch.cat(all_samples, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        dataset_path = f'{run_name}/samples_conditional_{args.seed}_{args.n_samples}.pth'
        torch.save((all_samples, all_labels), dataset_path)
        plot_samples(all_samples, all_labels, save_path=f"moons_{type_}.png")
        print(f"Saved dataset with {len(all_samples)} samples at {dataset_path}")        
        # output = my_utils.evaluate_samples(dataset_path, args.dataset)

    elif args.mode == 'cfg_experiment':
        model.load_state_dict(torch.load(f'{run_name}/model.pth'))            
        clf = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=1000, random_state=args.seed).fit(data_X.cpu().numpy(), data_y.cpu().numpy())
        
        train_preds = clf.predict(data_X.cpu().numpy())
        train_acc = (train_preds == data_y.cpu().numpy()).mean()
        print(f"Classifier accuracy on training data: {train_acc:.4f}")
        
        guidance_scales = [float(x) for x in args.guidance_scales.split(',')]
        
        results = experiment_guidance_scales(
            model, 
            noise_scheduler, 
            unique_labels, 
            guidance_scales, 
            args.n_samples, 
            clf,
            run_name
        )
        
        print("\nResults for different guidance scales:")
        print("----------------------------------------")
        print(f"Training data overall accuracy: {results['training_data']['overall_accuracy']:.4f}")
        print("Training data per-class accuracy:")
        for label, acc in results['training_data']['class_accuracies'].items():
            print(f"  Class {label}: {acc:.4f}")
        print("----------------------------------------")
        
        for scale in guidance_scales:
            result = results[scale]
            print(f"\nGuidance Scale: {scale}")
            print(f"Overall accuracy: {result['overall_accuracy']:.4f}")
            print("Per-class accuracy:")
            for label, acc in result['class_accuracies'].items():
                print(f"  Class {label}: {acc:.4f}")
        
        import json
        with open(f'{run_name}/cfg_detailed_results.json', 'w') as f:
            serializable_results = {}
            for k, v in results.items():
                if k == 'training_data':
                    serializable_results['training_data'] = {
                        'overall_accuracy': float(v['overall_accuracy']),
                        'class_accuracies': {str(label): float(acc) for label, acc in v['class_accuracies'].items()}
                    }
                else:
                    serializable_results[str(k)] = {
                        'overall_accuracy': float(v['overall_accuracy']),
                        'class_accuracies': {str(label): float(acc) for label, acc in v['class_accuracies'].items()}
                    }
            json.dump(serializable_results, f, indent=2)

        plt.figure(figsize=(8, 6))

        for label in unique_labels:
            label = int(label)
            class_accs = [results[scale]['class_accuracies'][label] for scale in guidance_scales]
            plt.plot(guidance_scales, class_accs, marker='o', linestyle='-', label=f'Class {label}')

        plt.xlabel('Guidance Scale')
        plt.ylabel('Class Accuracy')
        plt.title('Class Accuracy vs. Guidance Scale')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{run_name}/class_accuracy_vs_guidance.png", dpi=300)
        plt.show()

    elif args.mode == 'classifier_experiment':
        train_X, train_y, test_X, test_y = utils.split_data(data_X, data_y, split_ratio=0.9)
        clf = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=1000, random_state=args.seed).fit(train_X.cpu().numpy(), train_y.cpu().numpy())
        train_X, train_y, test_X, test_y = train_X.to(device), train_y.to(device), test_X.to(device), test_y.to(device)
        dataloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_X, train_y), batch_size=args.batch_size, shuffle=True, drop_last=True)
        run_name = f'exps/ddpm_{args.n_dim}_{args.n_steps}_{args.lbeta}_{args.ubeta}_{args.dataset}_{type_}_classifier'
        os.makedirs(run_name, exist_ok=True)
        epochs = args.epochs
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        trainConditional(model, noise_scheduler, dataloader, optimizer, epochs, run_name)
        model.load_state_dict(torch.load(f'{run_name}/model.pth'))

        results = compare_classifiers(run_name, model, noise_scheduler, clf, test_X, test_y, device)        
        print("\n=== Classifier Comparison Summary ===")
        print(f"Traditional Classifier Accuracy: {results['traditional_classifier']['accuracy']:.4f}")
        print(f"Diffusion Classifier Accuracy: {results['diffusion_classifier']['accuracy']:.4f}")
        print(f"Relative Performance: {(results['diffusion_classifier']['accuracy'] / results['traditional_classifier']['accuracy'] * 100):.2f}%")

    elif args.mode == 'svdd_experiment':
        run_name = "exps/ddpm_2_50_1e-05_0.02_manycircles"
        model.load_state_dict(torch.load(f'{run_name}/model.pth'))            
        clf = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=1000, random_state=args.seed).fit(data_X.cpu().numpy(), data_y.cpu().numpy())
        
        train_preds = clf.predict(data_X.cpu().numpy())
        train_acc = (train_preds == data_y.cpu().numpy()).mean()
        print(f"Classifier accuracy on training data: {train_acc:.4f}")
        reward_scales = [-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10]
        experiment_reward_scales(model, noise_scheduler, unique_labels, reward_scales, args.n_samples, clf, run_name)
    else:
        raise ValueError(f"Invalid mode {args.mode}")