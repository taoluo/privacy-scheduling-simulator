import math

def gaussian_dp2sigma(epsilon, sensitivity, delta):
    return (sensitivity/epsilon) * math.sqrt(2 * math.log(1.25/delta))

def compute_rdp_epsilons_gaussian(sigma, alphas):
    return [alpha / (2 * (sigma ** 2) ) for alpha in alphas]
