from frozendict import frozendict

"""
Although phase="zero" produces a non-causal filter, it has much less
effect than phase="minimal", which ends up introducing a delay and offsetting
the EEG from the ET, creating a poor ET correction because of temporal
misalignment. phase="zero" compared to no filtering has much less effect
and is unlikely to be responsible for significant pre-RT divergence.
"""
filter_kwargs = frozendict(l_freq=0.1, h_freq=30.0, phase="zero")
