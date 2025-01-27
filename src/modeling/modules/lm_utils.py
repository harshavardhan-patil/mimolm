import itertools
import torch

def create_vocabulary(max_delta, n_quantization_bins, n_verlet_steps):
    bins = [torch.linspace(-max_delta, max_delta, steps=n_quantization_bins),torch.linspace(-max_delta, max_delta, steps=n_quantization_bins),]
    verlet_wrapper = torch.linspace(-n_verlet_steps // 2 + 1, n_verlet_steps // 2, steps=n_verlet_steps)

    cartesian_product = list(itertools.product(torch.arange(n_verlet_steps), torch.arange(n_verlet_steps)))
    vocabulary = [[] for _ in range(n_verlet_steps ** 2)]
    k = 0
    for i, j in cartesian_product:
        vocabulary[k] = [i * n_verlet_steps + j, i, j]
        k+=1
        
    return vocabulary, bins, verlet_wrapper


def tokenize_motion(motion_tokens, vocabulary, verlet_wrapper, n_verlet_steps, n_time_steps):
    # delta_x and delta_y
    motion_tokens = torch.diff(motion_tokens, dim=2, prepend = motion_tokens[:, :, :1, :])
    # MotionLM uses greedy search, using bucketize here for simplicity
    x_tokens = torch.bucketize(motion_tokens[:, :, :, 0].contiguous(), vocabulary[0],)
    y_tokens = torch.bucketize(motion_tokens[:, :, :, 1].contiguous(), vocabulary[1],)
    x_tokens_diff = torch.diff(x_tokens, dim=2, prepend = x_tokens[:, :, :1])
    y_tokens_diff = torch.diff(y_tokens, dim=2, prepend = y_tokens[:, :, :1])
    # Verlet Wrapper (see paper): The idea is that velocity of cars changes smoothly, so we can use a smaller vocabulary to represent the relative motion between the last two time steps.
    # e.g: max_delta: float = 4.0,  n_quantization_bins: int = 128,  n_verlet_steps: int = 13, 10 Hz predicition, 
    # the max speed for the modeled agent is 4 x 10 = 40 m/s. 0 to max steps in Verlet Wrapper represents the max distance delta modeled. For -6 to 6 with 13 steps in Verlet and for 128 bins, the max acceleration between timesteps is 3.1 m/s^2.
    x_tokens = torch.clamp(torch.bucketize(x_tokens_diff, verlet_wrapper,), min = 0, max = n_verlet_steps - 1)
    y_tokens = torch.clamp(torch.bucketize(y_tokens_diff, verlet_wrapper,), min = 0, max = n_verlet_steps - 1)
    # collapse the per-coordinate actions to a single integer indexing into their Cartesian product
    return x_tokens * n_verlet_steps + y_tokens


def get_attention_mask(n_time_steps, size):
    i = torch.arange(size)[:, None] % n_time_steps
    j = torch.arange(size) % n_time_steps
    mask = i >= j
    mask = torch.logical_not(mask)
    return mask
