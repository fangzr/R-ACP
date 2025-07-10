# import warnings
# from typing import Any, Callable, List, Optional, Tuple, Union
import numpy as np
import scipy.stats
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from compressai.ops import LowerBound


def GaussianLikelihoodEstimation(inputs, scales, means):

    lower_bound_scale = LowerBound(0.11).to('cuda:0')
    likelihood_lower_bound = LowerBound(1e-9).to('cuda:0')

    def _likelihood(inputs, scales, means):

        half = float(0.5)
        values = inputs - means
        scales = lower_bound_scale(scales)
        values = torch.abs(values)
        upper = _standardized_cumulative((half - values) / scales)
        lower = _standardized_cumulative((-half - values) / scales)
        likelihood = upper - lower
        return likelihood

    def _standardized_cumulative( inputs: Tensor) -> Tensor:
        half = float(0.5)
        const = float(-(2**-0.5))
        # Using the complementary error function maximizes numerical precision.
        return half * torch.erfc(const * inputs)

    likelihood = _likelihood(inputs, scales, means)
    likelihood = likelihood_lower_bound(likelihood)

    return likelihood

