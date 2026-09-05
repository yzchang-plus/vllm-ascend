# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""BigTensorLoader default loader.

This is intentionally a *thin subclass* of vLLM's ``DefaultModelLoader``
rather than a line-by-line copy.  The only behavioural difference is that
``_prepare_weights`` recognises the ``"bigtensorloader"`` load format and
treats it like ``"safetensors"`` (``allow_patterns = ["*.safetensors"]``).

A full copy of ``DefaultModelLoader`` was used in the prototype to avoid
version coupling, but that duplicates ~1000 lines of upstream code and
drifts on every vLLM release.  Subclassing keeps the change minimal and
inherits all upstream fixes automatically.
"""

from vllm.model_executor.model_loader.default_loader import DefaultModelLoader


class BigTensorDefaultLoader(DefaultModelLoader):
    """DefaultModelLoader that also understands the ``bigtensorloader`` format.

    The ``bigtensorloader`` format uses the same on-disk weight files as
    ``safetensors``; the only difference is the post-process snapshot
    machinery implemented in :class:`BigTensorLoader`.
    """

    def _prepare_weights(self, *args, **kwargs):
        # ``bigtensorloader`` reuses the same weight files as ``safetensors``.
        # Temporarily masquerade as ``safetensors`` so the upstream
        # ``_prepare_weights`` sets ``allow_patterns = ["*.safetensors"]``
        # and enables the safetensors index filtering path.
        #
        # Signature-agnostic passthrough (*args, **kwargs): upstream has
        # changed this method's positional signature across releases (e.g.
        # vLLM 0.27.1 added a ``subfolder`` parameter between
        # ``model_name_or_path`` and ``revision``). Forwarding everything
        # transparently keeps this override working regardless of how the
        # upstream signature evolves.
        load_config = self.load_config
        original_format = load_config.load_format
        if original_format == "bigtensorloader":
            load_config.load_format = "safetensors"
        try:
            return super()._prepare_weights(*args, **kwargs)
        finally:
            load_config.load_format = original_format
