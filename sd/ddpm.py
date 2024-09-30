import numpy as np
import torch

class DDPMSampler:
    def __init__(self, generator: torch.Generator, beta_start: float = 0.00085, beta_end: float = 0.0120, num_training_steps=1000):

        self.betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_training_steps, dtype=torch.float32) ** 2
        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)

        self.one = torch.tensor(1.0)
        self.generator = generator
        self.num_training_steps = num_training_steps
        self.timesteps = torch.from_numpy(np.arange(0, num_training_steps)[::-1].copy())
    
    def set_inference_timesteps(self, n_inference_steps):
        self.n_inference_steps = n_inference_steps

        timesteps_ratio = n_inference_steps // self.num_training_steps

        self.timesteps = torch.from_numpy((np.arange(0, n_inference_steps)*timesteps_ratio).round()[::-1].copy().astype(np.int64))

    def _get_previous_timestep(self, timestep: int) -> int:
        return timestep - (self.n_inference_steps // self.num_training_steps)
    
    def _get_varaince(self, timestep: int):
        prev_t = self._get_previous_timestep(timestep)
        alpha_prod_t = self.alpha_cumprod[timestep]
        alpha_prod_t_prev = self.alpha_cumprod[prev_t] if prev_t >= 0 else self.one
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t

        var = (beta_prod_t_prev / beta_prod_t) * current_beta_t
        return torch.clamp(var, 1e-20)

    
    def step(self, timestep: torch.Tensor, latent: torch.Tensor, model_output: torch.Tensor):
        t = timestep
        prev_t = self._get_previous_timestep(t)

        alpha_prod_t = self.alpha_cumprod[timestep]
        alpha_prod_t_prev = self.alpha_cumprod[prev_t] if prev_t >= 0 else self.one
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t

        pred_original_sample = ((latent - (beta_prod_t ** 0.5)) * model_output) / alpha_prod_t ** 0.5

        pred_original_sample_coeff = (((alpha_prod_t_prev ** 0.5) * current_beta_t) / beta_prod_t ) * pred_original_sample
        current_sample_coeff = ((current_alpha_t ** 0.5 * beta_prod_t_prev) / alpha_prod_t) * latent

        pred_prev_sample = pred_original_sample_coeff + current_sample_coeff

        variance = 0
        if t > 0:
            noise = torch.rand(model_output.shape, generator=self.generator, device=model_output.device, dtype=model_output.device)
            variance = self._get_varaince(timestep) ** 0.5
        
        return pred_prev_sample + variance * noise

    def get_strength(self, strenght=1):
        start_step = self.num_training_steps - int(self.num_training_steps * strenght)
        self.timesteps = self.timesteps[start_step:]
        self.start_step = start_step


    def add_noise(self, original_samples: torch.FloatTensor, timesteps: torch.IntTensor) -> torch.FloatTensor : 
        
        alpha_cumprod = self.alpha_cumprod.to(device=original_samples.device, dtype=original_samples.dtype)
        timesteps = timesteps.to(device=original_samples.device)

        # var is alphacum prod
        sqrt_alpha_prod = alpha_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()

        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
        
        mean = sqrt_alpha_prod * original_samples
        
        sqrt_one_minus_sqrt_alpha_prod = (1 - sqrt_alpha_prod) ** 0.5 # Basically var 

        while len(sqrt_one_minus_sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_sqrt_alpha_prod = sqrt_one_minus_sqrt_alpha_prod.unsqueeze(-1)
        
        var = sqrt_one_minus_sqrt_alpha_prod
        
        noise = torch.randn(original_samples.shape, generator=self.generator, dtype=original_samples.dtype)
        noisy_samples =  mean + var * noise

        return noisy_samples