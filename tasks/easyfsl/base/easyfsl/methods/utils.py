"""Utilities for EasyFSL methods."""

from collections import OrderedDict

import torch
from torch import Tensor


def compute_prototypes(support_features: Tensor, support_labels: Tensor) -> Tensor:
    """Compute one mean feature vector for every support label."""

    n_way = len(torch.unique(support_labels))
    return torch.cat(
        [
            support_features[torch.nonzero(support_labels == label)].mean(0)
            for label in range(n_way)
        ]
    )


def strip_prefix(state_dict: OrderedDict, prefix: str) -> OrderedDict:
    """Return a copy of a state dictionary with an optional prefix removed."""

    return OrderedDict(
        (key[len(prefix) :] if key.startswith(prefix) else key, value)
        for key, value in state_dict.items()
    )
