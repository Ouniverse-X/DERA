"""Few-shot methods exposed by the task root."""

from .few_shot_classifier import FewShotClassifier
from .prototypical_networks import PrototypicalNetworks

__all__ = ["FewShotClassifier", "PrototypicalNetworks"]
