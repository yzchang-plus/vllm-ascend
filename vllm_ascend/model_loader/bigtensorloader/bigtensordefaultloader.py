# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Thin subclass of DefaultModelLoader for the bigtensorloader format."""

import copy

from vllm.config import LoadConfig
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader

_DEFAULT_LOADER_EXTRA_CONFIG_KEYS = frozenset({"enable_multithread_load", "num_threads", "enable_weights_track"})


def _copy_load_config(load_config: LoadConfig, load_format: str) -> LoadConfig:
    """Return an independent config accepted by ``DefaultModelLoader``.

    Upstream rejects BigTensor-only keys in its constructor, before
    ``BigTensorConfig`` can parse them. Copy both the config and its extra-config
    dictionary so neither shared object is mutated; draft/target models can load
    concurrently off one LoadConfig.
    """
    copied = copy.copy(load_config)
    copied.load_format = load_format
    extra_config = load_config.model_loader_extra_config
    if isinstance(extra_config, dict):
        copied.model_loader_extra_config = {
            key: value for key, value in extra_config.items() if key in _DEFAULT_LOADER_EXTRA_CONFIG_KEYS
        }
    return copied


class BigTensorDefaultLoader(DefaultModelLoader):
    """DefaultModelLoader that treats 'bigtensorloader' like 'safetensors'.

    The fresh-load path expects a safetensors checkpoint (the format the
    snapshot round-trip is validated with); ``_prepare_weights`` masquerades
    as safetensors so upstream keeps setting its allow_patterns.
    """

    def __init__(self, load_config: LoadConfig):
        super().__init__(_copy_load_config(load_config, load_config.load_format))
        # BigTensorLoader still needs the private keys for its own strict parser.
        self.load_config = load_config

    def _prepare_weights(self, *args, **kwargs):
        # Masquerade on a copied config handed to an independent loader --
        # never by mutating the shared LoadConfig, which draft/target models
        # can read concurrently off one config object. Signature-agnostic
        # passthrough (*args, **kwargs): upstream changes this method's
        # positional signature across releases.
        if self.load_config.load_format == "bigtensorloader":
            fallback_loader = DefaultModelLoader(_copy_load_config(self.load_config, "safetensors"))
            return fallback_loader._prepare_weights(*args, **kwargs)
        return super()._prepare_weights(*args, **kwargs)
