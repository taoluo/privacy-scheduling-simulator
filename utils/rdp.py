try:
    from opacus import privacy_analysis
except:
    pass
import math
# import math
# We drop alpha = +∞
ALPHAS = [1.5, 1.75, 2, 2.5, 3, 4, 5, 6, 8, 16, 32, 64]

# Reasonable bounds for parameter search
MIN_ORDER = 2
MAX_ORDER = 1000
MIN_NOISE = 0.01
MAX_NOISE = 1000

# fixme copyright 
# TODO: clean and class structure if we reuse that
def gaussian_dp2sigma(epsilon, sensitivity, delta):
    return (sensitivity/epsilon) * math.sqrt(2 * math.log(1.25/delta))

def compute_rdp_epsilons_gaussian(sigma, alphas):
    return [alpha / (2 * (sigma ** 2) ) for alpha in alphas]

def compute_rdp_epsilons_laplace(laplace_noise, orders):
    """
    RDP curve for a Laplace mechanism with sensitivity 1.
    Table II of the RDP paper (https://arxiv.org/pdf/1702.07476.pdf)
    """
    epsilons = []
    λ = laplace_noise
    for α in orders:
        if ((α - 1) / (2 * α - 1)) * math.exp(-α / λ) > 0:
            ε = (1 / (α - 1)) * math.log(
                (α / (2 * α - 1)) * math.exp((α - 1) / λ)
                + ((α - 1) / (2 * α - 1)) * math.exp(-α / λ)
            )

        else:# fix overflow issue
            ε = (1 / (α - 1)) * ( math.log((α / (2 * α - 1)))+ ((α - 1) / λ))
        epsilons.append(ε)
    return epsilons


def compute_epsilon(target_delta, steps, noise_multiplier, batch_size, dataset_size):
    """Computes epsilon privacy value for given hyperparameters."""
    if noise_multiplier == 0.0:
        return float("inf")

    orders = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))
    sampling_probability = batch_size / dataset_size
    rdp = privacy_analysis.compute_rdp(
        q=sampling_probability,
        noise_multiplier=noise_multiplier,
        steps=steps,
        orders=orders,
    )

    # Delta is set to the inverse of the number of points
    eps, _, order = privacy_analysis.get_privacy_spent(orders, rdp, delta=target_delta)
    print(f"Optimal order: {order}")
    return eps


def compute_steps(epochs, batch_size, dataset_size):
    return epochs * dataset_size // batch_size


def compute_rdp_sgm( # sumsampled gaussian
    epochs,
    batch_size,
    dataset_size,
    noise,
    alphas=None,
):
    steps = compute_steps(epochs, batch_size, dataset_size)
    sampling_rate = batch_size / dataset_size
    if alphas is None:
        alphas = ALPHAS
    return privacy_analysis.compute_rdp(sampling_rate, noise, steps, alphas)


def compute_noise_from_target_epsilon(
    target_epsilon,
    target_delta,
    epochs,
    batch_size,
    dataset_size,
    alphas=None,
    approx_ratio=0.01,
):
    """
    Takes a target epsilon (eps) and some hyperparameters.
    Returns a noise scale that gives an epsilon in [0.99 eps, eps].
    The approximation ratio can be tuned.
    If alphas is None, we'll explore orders.
    """
    steps = compute_steps(epochs, batch_size, dataset_size)
    sampling_rate = batch_size / dataset_size
    if alphas is None:
        alphas = ALPHAS

    def get_eps(noise):
        rdp = privacy_analysis.compute_rdp(sampling_rate, noise, steps, alphas)
        epsilon, order = privacy_analysis.get_privacy_spent(
            alphas, rdp, delta=target_delta
        )
        return epsilon

    # Binary search bounds
    noise_min = MIN_NOISE
    noise_max = MAX_NOISE

    # Start with the smallest epsilon possible with reasonable noise
    candidate_noise = noise_max
    candidate_eps = get_eps(candidate_noise)
    if candidate_eps > target_epsilon:
        raise ("Cannot reach target eps. Try to increase MAX_NOISE.")

    # Search up to approx ratio
    while (
        candidate_eps < (1 - approx_ratio) * target_epsilon
        or candidate_eps > target_epsilon
    ):
        if candidate_eps < (1 - approx_ratio) * target_epsilon:
            noise_max = candidate_noise
        else:
            noise_min = candidate_noise
        candidate_noise = (noise_max + noise_min) / 2
        candidate_eps = get_eps(candidate_noise)

    print("Use noise {} for epsilon {}".format(candidate_noise, candidate_eps))
    return candidate_noise

if __name__ == '__main__':
    pass


