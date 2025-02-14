import itertools
import torch
import torch.nn.functional as F
from torch_kmeans import KMeans

def create_vocabulary(max_delta, n_quantization_bins, n_verlet_steps):
    bins = torch.linspace(-max_delta, max_delta, steps=n_quantization_bins)
    verlet_wrapper = torch.linspace(-n_verlet_steps // 2 + 1, n_verlet_steps // 2, steps=n_verlet_steps)

    cartesian_product = list(itertools.product(torch.arange(n_verlet_steps), torch.arange(n_verlet_steps)))
    vocabulary = [[] for _ in range(n_verlet_steps ** 2)]
    k = 0
    for i, j in cartesian_product:
        vocabulary[k] = [i * n_verlet_steps + j, i, j]
        k+=1

    return torch.tensor(vocabulary), bins, verlet_wrapper


def tokenize_motion(motion_tokens, pos_bins, verlet_wrapper, n_verlet_steps):
    # delta_x and delta_y
    motion_tokens = torch.diff(motion_tokens, dim=2, prepend = motion_tokens[:, :, :1, :])
    # MotionLM uses greedy search, using bucketize here for simplicity
    x_tokens = torch.bucketize(motion_tokens[:, :, :, 0].contiguous(), pos_bins,)
    y_tokens = torch.bucketize(motion_tokens[:, :, :, 1].contiguous(), pos_bins,)
    x_last = x_tokens[:, :, -1].unsqueeze(-1)
    y_last = y_tokens[:, :, -1].unsqueeze(-1)
    x_tokens_diff = torch.diff(x_tokens, dim=2, prepend = x_tokens[:, :, :1])
    y_tokens_diff = torch.diff(y_tokens, dim=2, prepend = y_tokens[:, :, :1])
    # Verlet Wrapper (see paper): The idea is that velocity of cars changes smoothly, so we can use a smaller vocabulary to represent the relative motion between the last two time steps.
    # e.g: max_delta: float = 4.0,  n_quantization_bins: int = 128,  n_verlet_steps: int = 13, 10 Hz predicition, 
    # the max speed for the modeled agent is 4 x 10 = 40 m/s. 0 to max steps in Verlet Wrapper represents the max distance delta modeled. For -6 to 6 with 13 steps in Verlet and for 128 bins, the max acceleration between timesteps is 3.1 m/s^2.
    x_tokens = torch.clamp(torch.bucketize(x_tokens_diff, verlet_wrapper,), min = 0, max = n_verlet_steps - 1)
    y_tokens = torch.clamp(torch.bucketize(y_tokens_diff, verlet_wrapper,), min = 0, max = n_verlet_steps - 1)
    # collapse the per-coordinate actions to a single integer indexing into their Cartesian product
    return x_tokens * n_verlet_steps + y_tokens, torch.cat((x_last, y_last), dim=-1)

def get_attention_mask(n_time_steps, size):
    i = torch.arange(size)[:, None] % n_time_steps
    j = torch.arange(size) % n_time_steps
    mask = i >= j
    mask = torch.logical_not(mask)
    return mask

def nucleus_sampling(logits, top_p=0.95):
    probs = F.softmax(logits, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Mask tokens outside top_p
    nucleus = cumulative_probs < top_p
    nucleus = torch.cat([nucleus.new_ones(nucleus.shape[:-1] + (1,)), nucleus[..., :-1]], dim=-1)
    sorted_log_probs = torch.log(sorted_probs)
    sorted_log_probs[~nucleus] = float('-inf')
    
    # Sample from the filtered distribution
    sampled_indices = torch.multinomial(sorted_probs, num_samples=1).squeeze(-1)
    return sorted_indices.gather(-1, sampled_indices.unsqueeze(-1)).squeeze(-1)

def interpolate_trajectory(trajectory, scale_factor, device):
    batch, num_agents, timesteps, coords = trajectory.shape
    
    interpolated = torch.zeros(
        (batch, num_agents, 60, coords),
        dtype=trajectory.dtype,
        device=device
    )
    interpolated[:, :, ::scale_factor] = trajectory
    
    # Interpolate between points
    for i in range(1, scale_factor):
        alpha = i / scale_factor
        interp_idx = i + scale_factor * torch.arange(timesteps - 1)
        start_points = trajectory[:, :, :-1]
        end_points = trajectory[:, :, 1:]
        interpolated_points = (1 - alpha) * start_points + alpha * end_points
        interpolated[:, :, interp_idx] = interpolated_points
    
    # Add final point by extrapolating from last two points using verlet
    accl = (interpolated[:, :, -2] - interpolated[:, :, -3]) - (interpolated[:, :, -3] - interpolated[:, :, -4])
    interpolated[:, :, -1] = interpolated[:, :, -2] + (interpolated[:, :, -2] - interpolated[:, :, -3]) + accl / 2.
    
    return interpolated

def non_maximum_suppression(trajectories, threshold):
    """
    Apply NMS to remove redundant trajectories while ensuring all agents are within the distance threshold.
    
    Args:
    - trajectories: numpy array of shape (n_rollouts, n_agents, 60, 2)
    - threshold: distance threshold for suppression
    
    Returns:
    - Filtered trajectories as numpy array
    """
    keep_indices = []
    n_rollouts, n_agents, timesteps, _ = trajectories.shape

    for i in range(n_rollouts):
        keep = True
        for j in keep_indices:
            # Compute per-agent mean distance over all timesteps
            agent_distances = torch.linalg.norm(trajectories[i] - trajectories[j], dim=2).mean(-1)

            # Suppress only if all are within the threshold
            if torch.all(agent_distances < threshold):
                keep = False
                break
        
        if keep:
            keep_indices.append(i)

    return trajectories[keep_indices]


def cluster_rollouts(trajectories, n_clusters):
    """
    Apply K-Means clustering to the NMS-filtered trajectories.
    - trajectories: numpy array of shape (n_filtered_rollouts, n_agents, 60, 2)
    - n_clusters: desired number of representative joint trajectory modes
    - Returns: cluster centers (mode trajectories) and mode probabilities
    """
    n_rollouts, n_agents, timesteps, _ = trajectories.shape

    # Flatten each trajectory for clustering (n_rollouts, n_agents * n_time_steps * 2)
    reshaped_traj = trajectories.flatten(1, 3)

    # Apply K-Means clustering
    kmeans = KMeans(init_method="k-means++"
                    , num_init=reshaped_traj.shape[0] - 1 # doesnt support n_sample as clusters
                    , n_clusters=n_clusters
                    , random_state=42
                    , verbose=False) 
    results = kmeans(reshaped_traj.unsqueeze(0))
    cluster_labels = results.labels.squeeze(0)
    cluster_centers = results.centers.squeeze(0).unflatten(dim=1, sizes=(n_agents, timesteps, 2))

    # Compute mode probabilities based on cluster assignment
    mode_probs = torch.bincount(cluster_labels) / len(cluster_labels)

    return cluster_centers, mode_probs
