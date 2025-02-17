import itertools
import torch
import torch.nn.functional as F
from torch_kmeans import KMeans

# generated as quantiles of diff over the trainset (5Hz, 8.0m, 128 bins)
POS_BINS = [-7.99986982e+00, -4.80515516e-01, -2.90285110e-01, -2.07447052e-01,
       -1.59583569e-01, -1.27577724e-01, -1.04294774e-01, -8.67421402e-02,
       -7.30916895e-02, -6.22835159e-02, -5.35616875e-02, -4.64320183e-02,
       -4.05269618e-02, -3.55925560e-02, -3.14068198e-02, -2.78291702e-02,
       -2.47572687e-02, -2.20771438e-02, -1.97351646e-02, -1.76718980e-02,
       -1.58494115e-02, -1.42278671e-02, -1.27836357e-02, -1.14975572e-02,
       -1.03395581e-02, -9.29996371e-03, -8.36381689e-03, -7.51924515e-03,
       -6.75643329e-03, -6.05793297e-03, -5.42628765e-03, -4.85761165e-03,
       -4.33536619e-03, -3.85799213e-03, -3.41749191e-03, -3.01334262e-03,
       -2.64929608e-03, -2.31454521e-03, -1.99968368e-03, -1.72257423e-03,
       -1.46495062e-03, -1.22833711e-03, -1.01258140e-03, -8.32785008e-04,
       -6.54524192e-04, -4.97348440e-04, -3.70107591e-04, -2.43315590e-04,
       -1.31397753e-04, -4.13954258e-05,  5.31226397e-06,  8.01086426e-05,
        1.87076628e-04,  3.04222107e-04,  4.44378471e-04,  5.76049089e-04,
        7.46726990e-04,  9.39562917e-04,  1.13264285e-03,  1.36695057e-03,
        1.61322951e-03,  1.89649314e-03,  2.19231565e-03,  2.52063614e-03,
        2.88917124e-03,  3.28008792e-03,  3.71067226e-03,  4.17984853e-03,
        4.69755940e-03,  5.26269712e-03,  5.87858405e-03,  6.56014672e-03,
        7.30907917e-03,  8.13221931e-03,  9.04385212e-03,  1.00538624e-02,
        1.11795549e-02,  1.24282837e-02,  1.38215423e-02,  1.53844066e-02,
        1.71356201e-02,  1.91088915e-02,  2.13436857e-02,  2.38749981e-02,
        2.67695154e-02,  3.01017761e-02,  3.39735746e-02,  3.85200977e-02,
        4.38975613e-02,  5.03389764e-02,  5.81321716e-02,  6.76279068e-02,
        7.92440176e-02,  9.34823545e-02,  1.10866308e-01,  1.32049561e-01,
        1.57212257e-01,  1.85974430e-01,  2.16883183e-01,  2.47898817e-01,
        2.79182177e-01,  3.15620422e-01,  3.68264713e-01,  4.39561844e-01,
        5.20139694e-01,  6.09279673e-01,  7.03842163e-01,  8.02555725e-01,
        9.02770581e-01,  1.00376892e+00,  1.10529518e+00,  1.20780325e+00,
        1.30911268e+00,  1.41200435e+00,  1.51671028e+00,  1.62149048e+00,
        1.72696686e+00,  1.83071518e+00,  1.93999265e+00,  2.05144119e+00,
        2.16033173e+00,  2.25523979e+00,  2.39767838e+00,  2.55966759e+00,
        2.72384644e+00,  2.97303963e+00,  3.27389012e+00,  7.99988794e+00]

def create_vocabulary(max_delta, n_quantization_bins, n_verlet_steps):
    bins = torch.tensor(POS_BINS)
    verlet_wrapper = torch.linspace(-n_verlet_steps // 2 + 1, n_verlet_steps // 2, steps=n_verlet_steps)

    cartesian_product = list(itertools.product(torch.arange(n_verlet_steps), torch.arange(n_verlet_steps)))
    vocabulary = [[] for _ in range(n_verlet_steps ** 2)] 
    k = 0
    for i, j in cartesian_product:
        vocabulary[k] = [i * n_verlet_steps + j, i, j]
        k+=1
    vocabulary.append([n_verlet_steps ** 2, float('-inf'), float('-inf')]) # masking token
    return torch.tensor(vocabulary), bins, verlet_wrapper


def tokenize_motion(motion_tokens, pos_bins, verlet_wrapper, n_verlet_steps, max_delta):
    # delta_x and delta_y
    motion_tokens = torch.diff(motion_tokens, dim=2, prepend=motion_tokens[:, :, :1, :])
    # for masking transitional diffs
    invalid_indices = torch.cat(((motion_tokens < -max_delta).nonzero()
                                 , (motion_tokens > max_delta).nonzero())) 
    
    # MotionLM uses greedy search, using bucketize here for simplicity
    x_tokens = torch.bucketize(motion_tokens[:, :, :, 0].contiguous(), pos_bins,)
    y_tokens = torch.bucketize(motion_tokens[:, :, :, 1].contiguous(), pos_bins,)
    x_last = x_tokens[:, :, -1].unsqueeze(-1)
    y_last = y_tokens[:, :, -1].unsqueeze(-1)
    x_tokens_diff = torch.diff(x_tokens, dim=2, prepend = x_tokens[:, :, :1])
    y_tokens_diff = torch.diff(y_tokens, dim=2, prepend = y_tokens[:, :, :1])
    # Verlet Wrapper (see paper): The idea is that velocity of cars changes smoothly, 
    # so we can use a smaller vocabulary to represent the relative motion between the last two time steps.
    # e.g: max_delta: float = 4.0,  n_quantization_bins: int = 128,  n_verlet_steps: int = 13, 10 Hz predicition, 
    # the max speed for the modeled agent is 4 x 10 = 40 m/s. 
    # 0 to max steps in Verlet Wrapper represents the max distance delta modeled. 
    # THIS IS NO LONGER THE CASE WITH NON-UNIFORM BINS 
    # For -6 to 6 with 13 steps in Verlet and for 128 bins, the max acceleration between timesteps is 3.1 m/s^2.
    x_tokens = torch.clamp(torch.bucketize(x_tokens_diff, verlet_wrapper,), min = 0, max = n_verlet_steps - 1)
    y_tokens = torch.clamp(torch.bucketize(y_tokens_diff, verlet_wrapper,), min = 0, max = n_verlet_steps - 1)
    # collapse the per-coordinate actions to a single integer indexing into their Cartesian product
    cart_prod = x_tokens * n_verlet_steps + y_tokens
    # invalid tokens
    cart_prod[invalid_indices[:, 0], invalid_indices[:, 1], invalid_indices[:, 2]] = n_verlet_steps ** 2
    # should be set to mask_token?
    cart_prod[:, :, :1] = cart_prod[:, :, 1:2]
    return cart_prod, torch.cat((x_last, y_last), dim=-1),

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
