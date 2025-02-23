import itertools
import torch
import torch.nn.functional as F
from torch_kmeans import KMeans

def delta_to_scalar(delta_x, delta_y):
    # Compute magnitude
    magnitude = torch.sqrt(delta_x**2 + delta_y**2)
    
    # Compute direction (optional)
    direction = torch.arctan2(delta_y, delta_x)  # in radians
    
    return magnitude, direction


def scalar_to_delta(magnitude, direction):
    # Reconstruct delta x and delta y
    delta_x = magnitude * torch.cos(direction)
    delta_y = magnitude * torch.sin(direction)
    
    return delta_x, delta_y

def create_vocabulary(vocab_size):
    central_bins = torch.linspace(-0.1, 0.1, 41)
    medium_pos = torch.linspace(0.1, 3.0, 46)
    medium_neg = torch.linspace(-3.0, -0.1, 26)
    tail_pos = torch.linspace(3.0, 18.0, 52)
    tail_neg = torch.linspace(-18.0, -3.0, 31)
    bins = torch.unique(torch.cat([
        tail_neg,
        medium_neg,
        central_bins,
        medium_pos,
        tail_pos
    ]))

    r_space = torch.tensor(list(itertools.product(bins, bins)))
    theta_space = torch.arctan2(r_space[:, 1], r_space[:, 0])
    
    r_space = torch.sqrt(r_space[:, 0] ** 2 +  r_space[:, 1] ** 2)
    r_space = torch.unique(r_space)
    r_bins = r_space[::r_space.shape[0] // vocab_size]
    r_vocab = torch.cat([r_bins, torch.tensor([1e-9])]) # additional mask token

    theta_space = torch.unique(theta_space)
    theta_bins = theta_space[::theta_space.shape[0] // vocab_size]
    theta_vocab = torch.cat([theta_bins, torch.tensor([1e-9])]) # addtional mask token

    return r_vocab, theta_vocab, r_bins, theta_bins


def tokenize_motion(motion_tokens, max_delta, r_bins, theta_bins):
    # delta_x and delta_y
    motion_tokens = torch.diff(motion_tokens, dim=2, prepend=motion_tokens[:, :, :1, :])
    # for masking transitional diffs
    invalid_indices = torch.cat(((motion_tokens < -max_delta).nonzero()
                                 , (motion_tokens > max_delta).nonzero())) 
    
    motion_tokens[invalid_indices[:, 0], invalid_indices[:, 1], invalid_indices[:, 2]] = 0.0

    r = torch.sqrt(motion_tokens[:, :, :, 0] ** 2 +  motion_tokens[:, :, :, 1] ** 2)
    r_tokens = torch.clamp(torch.bucketize(r, r_bins.contiguous()), min=0, max=127)

    theta = torch.arctan2(motion_tokens[:, :, :, 1], motion_tokens[:, :, :, 0])
    theta_tokens = torch.clamp(torch.bucketize(theta, theta_bins.contiguous()), min=0, max=127)

    return r_tokens, theta_tokens

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
    for i in range(60 - scale_factor, 60):
        accl = (interpolated[:, :, i-1] - interpolated[:, :, i-2]) - (interpolated[:, :, i-2] - interpolated[:, :, i-3])
        interpolated[:, :, i] = interpolated[:, :, i-1] + (interpolated[:, :, i-1] - interpolated[:, :, i-2]) + accl / 2.
    
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
