import math

# tag
def gaussian_dp2sigma(epsilon, sensitivity, delta):
    return (sensitivity/epsilon) * math.sqrt(2 * math.log(1.25/delta))
## tag
def compute_rdp_epsilons_gaussian(sigma, alphas):
    return [alpha / (2 * (sigma ** 2) ) for alpha in alphas]
## tag
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
