#!/usr/bin/env python3
"""Rotation-invariant radial-profile descriptors and similarity scoring."""

import numpy as np
from typing import Dict
from dataclasses import dataclass, field
from scipy.spatial.distance import cosine


@dataclass
class RadialDescriptor:
    """Compact rotation-invariant descriptor for environment matching."""
    angles: np.ndarray = field(default_factory=lambda: np.array([]))
    raw_distances: np.ndarray = field(default_factory=lambda: np.array([]))
    filtered_distances: np.ndarray = field(default_factory=lambda: np.array([]))
    confidence: np.ndarray = field(default_factory=lambda: np.array([]))
    mean_radius: float = 0.0
    std_radius: float = 0.0
    radius_range: float = 0.0
    fourier_magnitudes: np.ndarray = field(default_factory=lambda: np.array([]))
    circularity: float = 0.0

    def to_dict(self) -> Dict:
        """Serialize to a JSON-safe dictionary."""
        return {
            'angles': self.angles.tolist() if self.angles.size > 0 else [],
            'raw_distances': self.raw_distances.tolist() if self.raw_distances.size > 0 else [],
            'filtered_distances': self.filtered_distances.tolist() if self.filtered_distances.size > 0 else [],
            'confidence': self.confidence.tolist() if self.confidence.size > 0 else [],
            'mean_radius': float(self.mean_radius),
            'std_radius': float(self.std_radius),
            'radius_range': float(self.radius_range),
            'fourier_magnitudes': self.fourier_magnitudes.tolist() if len(self.fourier_magnitudes) > 0 else [],
            'circularity': float(self.circularity),
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'RadialDescriptor':
        """Deserialize from to_dict() output."""
        return cls(
            angles=np.array(data.get('angles', []), dtype=np.float32),
            raw_distances=np.array(data.get('raw_distances', []), dtype=np.float32),
            filtered_distances=np.array(data.get('filtered_distances', []), dtype=np.float32),
            confidence=np.array(data.get('confidence', []), dtype=np.float32),
            mean_radius=float(data.get('mean_radius', 0.0)),
            std_radius=float(data.get('std_radius', 0.0)),
            radius_range=float(data.get('radius_range', 0.0)),
            fourier_magnitudes=np.array(data.get('fourier_magnitudes', []), dtype=np.float32),
            circularity=float(data.get('circularity', 0.0)),
        )


class RadialDescriptorMatcher:
    """Weighted statistical + Fourier-magnitude similarity for RadialDescriptor."""

    def __init__(
        self,
        weight_statistical: float = 0.3,
        weight_spectral: float = 0.7,
        sigma_mean: float = 0.5,
        sigma_std: float = 0.3,
        sigma_circularity: float = 0.2,
    ):
        total = weight_statistical + weight_spectral
        self.w_stat = weight_statistical / total
        self.w_spec = weight_spectral / total
        self.sigma_mean = float(sigma_mean)
        self.sigma_std = float(sigma_std)
        self.sigma_circ = float(sigma_circularity)

    def compute_similarity(
        self,
        desc1: RadialDescriptor,
        desc2: RadialDescriptor,
    ) -> Dict[str, float]:
        """Return statistical, spectral, and combined similarity scores."""
        stat_sim = self._statistical_similarity(desc1, desc2)
        spec_sim = self._spectral_similarity(desc1, desc2)
        combined = (
            self.w_stat * stat_sim +
            self.w_spec * spec_sim
        )
        return {
            'statistical_similarity': stat_sim,
            'spectral_similarity': spec_sim,
            'combined_score': combined,
        }

    def _statistical_similarity(
        self,
        desc1: RadialDescriptor,
        desc2: RadialDescriptor,
    ) -> float:
        mean_sim = self._gaussian_similarity(
            desc1.mean_radius,
            desc2.mean_radius,
            self.sigma_mean,
        )
        std_sim = self._gaussian_similarity(
            desc1.std_radius,
            desc2.std_radius,
            self.sigma_std,
        )
        circ_sim = self._gaussian_similarity(
            desc1.circularity,
            desc2.circularity,
            self.sigma_circ,
        )
        range_sim = self._gaussian_similarity(
            desc1.radius_range,
            desc2.radius_range,
            self.sigma_std,
        )
        return 0.4 * mean_sim + 0.3 * std_sim + 0.2 * circ_sim + 0.1 * range_sim

    def _spectral_similarity(
        self,
        desc1: RadialDescriptor,
        desc2: RadialDescriptor,
    ) -> float:
        if len(desc1.fourier_magnitudes) == 0 or len(desc2.fourier_magnitudes) == 0:
            return 0.0
        len1 = len(desc1.fourier_magnitudes)
        len2 = len(desc2.fourier_magnitudes)
        if len1 < len2:
            mag1 = np.pad(desc1.fourier_magnitudes, (0, len2 - len1))
            mag2 = desc2.fourier_magnitudes
        elif len2 < len1:
            mag1 = desc1.fourier_magnitudes
            mag2 = np.pad(desc2.fourier_magnitudes, (0, len1 - len2))
        else:
            mag1 = desc1.fourier_magnitudes
            mag2 = desc2.fourier_magnitudes
        norm1 = np.linalg.norm(mag1)
        norm2 = np.linalg.norm(mag2)
        if norm1 < 1e-9 or norm2 < 1e-9:
            return 0.0
        mag1_norm = mag1 / norm1
        mag2_norm = mag2 / norm2
        similarity = 1.0 - cosine(mag1_norm, mag2_norm)
        return max(0.0, min(1.0, similarity))

    @staticmethod
    def _gaussian_similarity(val1: float, val2: float, sigma: float) -> float:
        """Gaussian kernel: exp(-0.5 * ((val1 - val2) / sigma)^2)."""
        if sigma < 1e-9:
            return 1.0 if abs(val1 - val2) < 1e-9 else 0.0
        diff = abs(val1 - val2)
        sim = np.exp(-0.5 * (diff / sigma) ** 2)
        return float(sim)


def create_descriptor_from_profile(
    profile: 'RadialProfile',
    num_harmonics: int = 8,
    store_raw_profile: bool = False,
) -> RadialDescriptor:
    """Build a RadialDescriptor from a RadialProfile; low-frequency FFT magnitudes are rotation-invariant."""
    stats = profile.get_statistics()
    fourier_mags = profile.compute_fourier_features(num_harmonics=num_harmonics)
    circularity = stats['std_radius'] / max(stats['mean_radius'], 1e-6)
    if store_raw_profile:
        descriptor = RadialDescriptor(
        angles=profile.angles.copy(),
        raw_distances=profile.raw_distances.copy(),
        filtered_distances=profile.filtered_distances.copy(),
        confidence=profile.confidence.copy(),
        mean_radius=stats['mean_radius'],
        std_radius=stats['std_radius'],
            radius_range=stats['range'],
            fourier_magnitudes=fourier_mags,
            circularity=circularity,
        )
    else:
        descriptor = RadialDescriptor(
            mean_radius=stats['mean_radius'],
            std_radius=stats['std_radius'],
        radius_range=stats['range'],
        fourier_magnitudes=fourier_mags,
        circularity=circularity,
    )
    return descriptor
