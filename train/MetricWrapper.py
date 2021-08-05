from typing import Union, Callable, Optional, Dict, Any
from copy import deepcopy
import torch
from torch import Tensor

class MetricWrapper:
    r"""
    Allows to initialize a metric from a name or Callable, and initialize the
    `Thresholder` in case the metric requires a threshold.
    """

    def __init__(
        self,
        metric: Union[str, Callable],
        target_nan_mask: Optional[Union[str, int]] = None,
        **kwargs,
    ):
        r"""
        Parameters
            metric:
                The metric to use. See `METRICS_DICT`

            target_nan_mask:

                - None: Do not change behaviour if there are NaNs

                - int, float: Value used to replace NaNs. For example, if `target_nan_mask==0`, then
                  all NaNs will be replaced by zeros

                - 'ignore-flatten': The Tensor will be reduced to a vector without the NaN values.

                - 'ignore-mean-label': NaNs will be ignored when computing the loss. Note that each column
                  has a different number of NaNs, so the metric will be computed separately
                  on each column, and the metric result will be averaged over all columns.
                  *This option might slowdown the computation if there are too many labels*

            kwargs:
                Other arguments to call with the metric
        """

        self.metric = metric
        self.target_nan_mask = target_nan_mask
        self.kwargs = kwargs

    def compute(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        r"""
        Compute the metric, apply the thresholder if provided, and manage the NaNs
        """

        if preds.ndim == 1:
            preds = preds.unsqueeze(-1)

        if target.ndim == 1:
            target = target.unsqueeze(-1)

        target_nans = torch.isnan(target)

        # Manage the NaNs
        if self.target_nan_mask is None:
            pass
        elif isinstance(self.target_nan_mask, (int, float)):
            target = target.clone()
            target[torch.isnan(target)] = self.target_nan_mask
        elif self.target_nan_mask == "ignore-flatten":
            target = target[~target_nans]
            preds = preds[~target_nans]
        elif self.target_nan_mask == "ignore-mean-label":
            target_list = [target[..., ii][~target_nans[..., ii]] for ii in range(target.shape[-1])]
            preds_list = [preds[..., ii][~target_nans[..., ii]] for ii in range(preds.shape[-1])]
            target = target_list
            preds = preds_list
        else:
            raise ValueError(f"Invalid option `{self.target_nan_mask}`")

        if self.target_nan_mask == "ignore-mean-label":

            # Compute the metric for each column, and output nan if there's an error on a given column
            metric_val = []
            for ii in range(len(target)):
                try:
                    metric_val.append(self.metric(preds[ii], target[ii], **self.kwargs))
                except:
                    pass

            # Average the metric
            
            metric_val = self.nan_mean(torch.stack(metric_val))

        else:
            metric_val = self.metric(preds, target, **self.kwargs)
        return metric_val

    def __call__(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        r"""
        Compute the metric with the method `self.compute`
        """
        return self.compute(preds, target)

    def __repr__(self):
        r"""
        Control how the class is printed
        """
        full_str = f"{self.metric.__name__}"

        return full_str
    
    def nan_mean(self, input: Tensor, **kwargs) -> Tensor:
        sum = torch.nansum(input, **kwargs)
        num = torch.sum(~torch.isnan(input), **kwargs)
        mean = sum / num
        return mean


