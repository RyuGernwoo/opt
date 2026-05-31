from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .models import ProblemInstance


@dataclass(frozen=True)
class WirelessConfig:
    user_count: int = 15
    rb_count: int = 12
    seed: int = 0
    cell_radius_m: float = 500.0
    min_distance_m: float = 25.0
    rb_bandwidth_hz: float = 180_000.0
    noise_density_w_hz: float = 1.0e-20
    p_max_w: float = 0.2
    local_model_bits: float = 25_000.0
    delay_limit_s: float = 0.25
    energy_limit_j: float = 0.05
    waterfall_threshold: float = 2.0


def _rate_s(config: WirelessConfig, p_w: float, channel_gain: float, interference_w: float) -> float:
    noise_w = config.rb_bandwidth_hz * config.noise_density_w_hz
    sinr = p_w * channel_gain / max(interference_w + noise_w, 1e-30)
    return config.rb_bandwidth_hz * np.log2(1.0 + max(sinr, 0.0))


def _packet_error(config: WirelessConfig, p_w: float, channel_gain: float, interference_w: float) -> float:
    noise_w = config.rb_bandwidth_hz * config.noise_density_w_hz
    sinr = p_w * channel_gain / max(interference_w + noise_w, 1e-30)
    q = 1.0 - np.exp(-config.waterfall_threshold / max(sinr, 1e-12))
    return float(np.clip(q, 1e-6, 0.99))


def generate_instance(config: WirelessConfig) -> ProblemInstance:
    rng = np.random.default_rng(config.seed)
    u_count, r_count = config.user_count, config.rb_count

    sample_counts = rng.integers(20, 201, size=u_count).astype(float)
    radial = config.cell_radius_m * np.sqrt(rng.random(u_count))
    distances = np.maximum(config.min_distance_m, radial)

    fading = rng.exponential(scale=1.0, size=(u_count, r_count))
    shadowing = rng.lognormal(mean=0.0, sigma=0.35, size=(u_count, r_count))
    channel_gain = fading * shadowing / (distances[:, None] ** 2)
    interference = rng.lognormal(mean=np.log(4.0e-8), sigma=0.7, size=r_count)
    train_energy = rng.uniform(0.002, 0.010, size=u_count) * (sample_counts / np.mean(sample_counts))

    p_star = np.zeros((u_count, r_count))
    delay = np.zeros((u_count, r_count))
    energy = np.zeros((u_count, r_count))
    packet_error = np.zeros((u_count, r_count))

    p_grid = np.geomspace(1e-4, config.p_max_w, num=64)
    for i in range(u_count):
        for n in range(r_count):
            best_p = p_grid[0]
            best_delay = float("inf")
            best_energy = float("inf")
            for p_w in p_grid:
                rate = _rate_s(config, float(p_w), float(channel_gain[i, n]), float(interference[n]))
                current_delay = config.local_model_bits / max(rate, 1e-12)
                current_energy = float(train_energy[i] + p_w * current_delay)
                if current_energy <= config.energy_limit_j:
                    best_p = float(p_w)
                    best_delay = float(current_delay)
                    best_energy = current_energy
            p_star[i, n] = best_p
            delay[i, n] = best_delay
            energy[i, n] = best_energy
            packet_error[i, n] = _packet_error(config, best_p, float(channel_gain[i, n]), float(interference[n]))

    feasible = (delay <= config.delay_limit_s) & (energy <= config.energy_limit_j)
    cost = sample_counts[:, None] * (packet_error - 1.0)

    return ProblemInstance(
        user_count=u_count,
        rb_count=r_count,
        sample_counts=sample_counts,
        packet_error=packet_error,
        p_star=p_star,
        delay=delay,
        energy=energy,
        feasible=feasible,
        cost=cost,
        seed=config.seed,
        metadata={
            "cell_radius_m": config.cell_radius_m,
            "delay_limit_s": config.delay_limit_s,
            "energy_limit_j": config.energy_limit_j,
            "local_model_bits": config.local_model_bits,
        },
    )
