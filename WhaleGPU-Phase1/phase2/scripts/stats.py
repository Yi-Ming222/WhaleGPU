"""Statistics contract: linear quantiles, sample stddev, no outlier deletion."""
import math
import statistics


def percentile(values, q):
    if not values or not 0 <= q <= 1:
        raise ValueError('nonempty values and 0 <= q <= 1 required')
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lo, hi = math.floor(position), math.ceil(position)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def summarize(values):
    if not values or any(not math.isfinite(x) or x < 0 for x in values):
        raise ValueError('finite nonnegative samples required')
    return dict(count=len(values), mean=statistics.mean(values), median=statistics.median(values),
                stddev=statistics.stdev(values) if len(values) > 1 else 0.0,
                p95=percentile(values, .95), minimum=min(values), maximum=max(values))


def effective_gbps(n, ms):
    if n <= 0 or ms <= 0:
        raise ValueError('positive input and time required')
    return (4 * n + 4) / (ms * 1e6)
