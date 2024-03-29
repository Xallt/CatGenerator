import torch
import math
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F

# utility function. basicalle, returns arr[timesteps], where timesteps are indices. (look at class Diffusion)
def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D torch tensor for a batch of indices.
    :param arr: 1-D torch tensor.
    :param timesteps: a tensor of indices into torche array to extract.
    :param broadcast_shape: a larger shape of K dimensions witorch torche batch
                            dimension equal to torche lengtorch of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where torche shape has K dims.
    """
    res = arr.to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)

# out beta_t. we use linear scheduler
def get_named_beta_schedule(schedule_name, num_diffusion_timesteps, beta_last=0.02):
    """
    Get a pre-defined beta schedule for the given name.
    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    scale = 1000 / num_diffusion_timesteps
    beta_start = scale * 0.0001
    beta_end = scale * beta_last 
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "quad":
        betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_diffusion_timesteps) ** 2
        return betas.numpy()
    elif schedule_name == "sigmoid":
        betas = torch.linspace(-6, 6, num_diffusion_timesteps)
        betas = torch.sigmoid(betas) * (beta_end - beta_start) + beta_start
        return betas.numpy()
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")

class Diffusion(torch.nn.Module):
    def __init__(
        self,
        model,
        betas: np.array,
        loss_type: str = "mse",
        mode = "pred_start",
    ):
        """
        Class that simulates Diffusion process. Does not store model or optimizer.
        """
        super().__init__()

        assert mode in ["pred_start", "pred_noise"]
        self.mode = mode

        self.model = model

        betas = torch.from_numpy(betas).double()
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        self.alphas = 1.0 - betas
        self.alphas_cumprod = self.alphas.cumprod(dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), self.alphas_cumprod[:-1]], dim=0)
        self.alphas_cumprod_next = torch.cat([self.alphas_cumprod[1:], torch.tensor([0.0]), ], dim=0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # calculations for diffusion q(x_t | x_{t-1})
        self.sqrt_alphas_cumprod = self.alphas_cumprod.sqrt()
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)


        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * self.betas

        # log calculation clipped because posterior variance is 0.
        self.posterior_log_variance_clipped = torch.log(
            torch.cat([self.posterior_variance[1:2], self.posterior_variance[1:]], dim=0)
        )
        self.posterior_mean_coef1 = self.alphas.sqrt() * (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)
        self.posterior_mean_coef2 = self.alphas_cumprod_prev.sqrt() * self.betas / (1 - self.alphas_cumprod)

    def q_mean_variance(self, x0, t):
        """
        Get mean and variance of distribution q(x_t | x_0). Use equation (1).
        """

        mean = _extract_into_tensor(self.sqrt_alphas_cumprod, t, x0.shape) * x0
        variance = 1 - self.alphas_cumprod
        log_variance = variance.log()
        return mean, variance, log_variance

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute mean and variance of diffusion posterior q(x_{t-1} | x_t, x_0).
        Use equation (2) and (3).
        """
        assert x_start.shape == x_t.shape
        posterior_mean = \
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_t + \
            _extract_into_tensor(self.posterior_mean_coef2, t, x_start.shape) * x_start
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    
    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse data for a given number of diffusion steps.
        Sample from q(x_t | x_0).
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        return _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start + \
            _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise

    def p_mean_variance(self, model_output, x, t):
        """
        Apply model to get p(x_{t-1} | x_t). Use Equation (2) and plug in \hat{x}_0;
        """
        model_variance = torch.cat([self.posterior_variance[1:2], self.betas[1:]], dim=0)
        model_log_variance = torch.log(model_variance)
        model_variance = _extract_into_tensor(model_variance, t, x.shape)
        model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)

        # Assuming model_output is \hat{\theta}
        if self.mode == "pred_noise":
            pred_xstart = self._predict_xstart_from_eps(x, t, model_output)
            model_mean, *_ = self.q_posterior_mean_variance(pred_xstart, x, t)
        elif self.mode == "pred_start":
            pred_xstart = model_output
            model_mean, *_ = self.q_posterior_mean_variance(model_output, x, t)

        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    def _predict_xstart_from_eps(self, x_t, t, eps):
        """
        Get \hat{x0} from epsilon_{theta}. Use equation (4) to derive it.
        """

        return _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * \
            (
                x_t - _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * eps
            )


    def p_sample(self, model_output, x, t):
        """
        Sample from p(x_{t-1} | x_t).
        """
        out = self.p_mean_variance(model_output, x, t)
        noise = torch.randn_like(x)
        nonzero_mask = _extract_into_tensor(torch.ones_like(self.alphas_cumprod), t, x.shape) > 0

        sample = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample}
    
    def p_sample_loop(self, shape, device=None):
        """
        Samples a batch=shape[0] using diffusion model.
        """

        x = torch.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]

        for i in tqdm(indices):
            t = torch.tensor([i] * shape[0], device=x.device)
            with torch.no_grad():
                model_output = self.model(x, t)
                out = self.p_sample(
                    model_output,
                    x,
                    t
                )
                x = out["sample"]
        return x
    
    def compute_loss(self, x0):
        """
        Calculates loss L^{simple}_t for the given model, x0.
        """
        t = torch.randint(0, self.num_timesteps, size=(x0.size(0),), device=x0.device)
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        model_output = self.model(x_t, t)
        if self.mode == "pred_noise":
            loss = F.mse_loss(model_output, noise)
        elif self.mode == "pred_start":
            loss = F.mse_loss(model_output, x0)
        return {
            'loss': loss
        }