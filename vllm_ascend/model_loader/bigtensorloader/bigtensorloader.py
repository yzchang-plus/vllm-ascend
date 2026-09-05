# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.logger import logger
from vllm.model_executor.model_loader import register_model_loader
from vllm.config import LoadConfig, ModelConfig, VllmConfig
from vllm.model_executor.model_loader.utils import (initialize_model, process_weights_after_loading)
from vllm.utils.torch_utils import set_default_torch_dtype
from vllm_ascend import envs
from .bigtensordefaultloader import BigTensorDefaultLoader

import torch
import torch.nn as nn
import torch_npu
import time
import mmap
import json
import os
import threading
from typing import Set, Dict, Any, Optional


@register_model_loader("bigtensorloader")
class BigTensorLoader(BigTensorDefaultLoader):

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)

    # dtype 映射表，类级别常量避免重复创建
    # FP8 dtypes: v2 manifest stores str(t.dtype) ("float8_e4m3fn"); the
    # reverse lookup in restore_weights_v2 needs them here or FP8 models
    # fail-loud on "not in _DTYPE_MAP".
    _DTYPE_MAP: Dict[str, torch.dtype] = {
        "F32": torch.float32,
        "F64": torch.float64,
        "F16": torch.float16,
        "BF16": torch.bfloat16,
        "I8": torch.int8,
        "I16": torch.int16,
        "I32": torch.int32,
        "I64": torch.int64,
        "U8": torch.uint8,
        "BOOL": torch.bool,
        "F8_E4M3": torch.float8_e4m3fn,
        "F8_E5M2": torch.float8_e5m2,
        "F8_E8M0": torch.float8_e8m0fnu,  # MXFP8 block-scale dtype
    }

    # torch.dtype -> manifest key (reverse of _DTYPE_MAP), O(1) lookup.
    _DTYPE_KEY: Dict[torch.dtype, str] = {
        torch.float32: "F32", torch.float64: "F64", torch.float16: "F16",
        torch.bfloat16: "BF16", torch.int8: "I8", torch.int16: "I16",
        torch.int32: "I32", torch.int64: "I64", torch.uint8: "U8",
        torch.bool: "BOOL", torch.float8_e4m3fn: "F8_E4M3",
        torch.float8_e5m2: "F8_E5M2", torch.float8_e8m0fnu: "F8_E8M0",
    }

    @staticmethod
    def _v2_fingerprint(vllm_config: VllmConfig, model_config: ModelConfig) -> dict:
        """Identity of everything that shapes the post-process snapshot.

        A snapshot is only valid for the exact (model, dtype, quantization,
        parallel layout, nz mode) it was converted under: weights are sharded
        per rank and process output depends on quant config. Restore compares
        fingerprints and refuses mismatches instead of silently loading
        wrong weights.
        """
        pc = vllm_config.parallel_config
        # weight_nz_mode affects which weights are NZ-cast during
        # process_weights_after_loading; a snapshot built under one mode
        # must not be restored under another.
        additional_config = getattr(vllm_config, "additional_config", {}) or {}
        weight_nz_mode = additional_config.get("weight_nz_mode", 1)
        return {
            "model": model_config.model,
            "dtype": str(model_config.dtype).replace("torch.", ""),
            "quantization": model_config.quantization,
            "tp": pc.tensor_parallel_size,
            "pp": pc.pipeline_parallel_size,
            "dp": getattr(pc, "data_parallel_size", 1),
            "ep": pc.enable_expert_parallel,
            "world_size": torch.distributed.get_world_size(),
            "weight_nz_mode": weight_nz_mode,
        }

    def _resolve_param(self, name: str, param_dict: Dict, name_to_module: Dict,
                       target_device: torch.device, dtype_str: str,
                       pre_shapes: Optional[dict] = None,
                       added_params: Optional[Set[str]] = None):
        """Resolve or register a parameter for a snapshot entry.

        Four cases:
        1. Param already exists in param_dict (state_dict) -> return it.
        2. Module exists but attribute is missing AND the entry name is on
           the save-side recorded added_params list (post - pre state_dict
           keys, i.e. created by process_weights_after_loading, e.g. W8A8's
           aclnn_input_scale) -> dynamically register an empty placeholder
           Parameter, to be filled by restore.
        3. Module exists but attribute is missing AND the entry is NOT on
           the added_params list -> ghost entry (manifest tampered/corrupt
           or structure drift), return None for the coverage stale check
           to fail-loud. Legacy manifests without the added_params key
           fall back to the r8 heuristic (name not in pre_shapes).
        4. Module itself doesn't exist -> return None for the coverage
           stale check to fail-loud.
        """
        param = param_dict.get(name)
        if param is not None:
            return param
        module_name, _, local_weight_name = name.rpartition(".")
        if not local_weight_name:
            raise RuntimeError(
                f"v2 resolve: snapshot entry {name!r} has no module prefix "
                f"(expected 'module.weight' form); the snapshot is stale or "
                f"corrupt -- re-run convert.")
        layer = name_to_module.get(module_name)
        if layer is None:
            return None
        dtype = self._DTYPE_MAP.get(dtype_str, torch.float32)
        if hasattr(layer, local_weight_name):
            return getattr(layer, local_weight_name)
        # Attribute is missing. Only entries the save side explicitly
        # recorded as process-added may register; a ghost entry from a
        # tampered manifest is equally "not in pre_shapes", so the
        # pre_shapes complement cannot distinguish the two -- added_params
        # is the authoritative allow-list.
        if added_params is not None:
            allow_register = name in added_params
        else:
            allow_register = name not in (pre_shapes or {})
        if allow_register:
            placeholder = nn.Parameter(
                torch.empty(0, dtype=dtype, device=target_device))
            layer.register_parameter(local_weight_name, placeholder)
            logger.debug(
                "v2 resolve: registered process-added param %r on module %r",
                name, module_name)
            return placeholder
        logger.warning(
            "v2 resolve: snapshot entry %r has no attribute %r on module %r "
            "and is not in added_params; ghost entry or structure drift "
            "-- re-run convert.",
            name, local_weight_name, module_name)
        return None

    def _v2_snapshot_paths(self) -> tuple:
        rank = torch.distributed.get_rank()
        path = envs.VLLM_ASCEND_CHECKPOINT_PATH
        if not path:
            raise RuntimeError(
                "VLLM_ASCEND_BIGTENSOR_V2=1 requires VLLM_ASCEND_CHECKPOINT_PATH "
                "to be set to a writable directory where the post-process "
                "snapshot is stored (e.g. VLLM_ASCEND_CHECKPOINT_PATH="
                "/path/to/snapshot_dir). Unset VLLM_ASCEND_BIGTENSOR_V2 to "
                "use the normal fresh-load path.")
        if os.path.exists(path) and not os.path.isdir(path):
            raise RuntimeError(
                f"VLLM_ASCEND_CHECKPOINT_PATH={path!r} is not a directory; "
                f"bigtensorloader needs a writable directory to store "
                f"snapshot files. Point it to a directory instead.")
        path = path.rstrip("/")
        return f"{path}/{rank}.v2.json", f"{path}/{rank}.v2.snapshot"

    def _has_v2_snapshot(self) -> bool:
        # Avoid raising when CHECKPOINT_PATH is unset: just report "no
        # snapshot" so the caller falls back to the convert path instead
        # of crashing. The friendly error is raised by _v2_snapshot_paths
        # when restore actually needs the path.
        if not envs.VLLM_ASCEND_CHECKPOINT_PATH:
            return False
        manifest, blob = self._v2_snapshot_paths()
        return os.path.exists(manifest) and os.path.exists(blob)

    @staticmethod
    def _v2_is_packed_nz(t: torch.Tensor) -> bool:
        """Packed low-bit NZ weight: int8 NZ storage viewed as int32
        (w4a8/w4a16 pack via ``int8_nz.view(torch.int32)``). The logical
        dtype/shape no longer match the NZ physical layout, so a direct
        D2H transdata fails with a fatal EZ9999 that poisons the process
        (no Python-level recovery). D2H through the int8 view is safe and
        yields the underlying int8 ND-logical bytes. Genuine int32 NZ
        tensors do not occur: every NZ cast in vllm_ascend targets int8
        quantized weights (or bf16, whose dtype matches its storage)."""
        return (t.device.type == "npu"
                and t.dtype == torch.int32
                and int(torch_npu.get_npu_format(t)) == 29)

    def _v2_append_tensor(self, t: torch.Tensor, blob_parts: list,
                          offset: int) -> tuple[dict, int]:
        """Serialize one tensor into blob_parts (16B-aligned) and return
        (base_entry, new_offset). The entry carries dtype/shape/format/offset/
        nbytes; callers add kind-specific fields (kind/device/carrier/attr).

        For packed low-bit NZ tensors (see _v2_is_packed_nz) the bytes come
        from the int8 view and the entry additionally records
        save_dtype/save_shape so restore can rebuild the int8 tensor and
        view it back to the logical int32 dtype after the NZ cast."""
        t = t.detach()
        fmt = int(torch_npu.get_npu_format(t)) if t.device.type == "npu" else 0
        save_dtype = save_shape = None
        d2h = t
        if self._v2_is_packed_nz(t):
            save_dtype = "int8"
            save_shape = list(t.view(torch.int8).shape)
            d2h = t.view(torch.int8)
        raw = (d2h.to("cpu").contiguous().reshape(-1).view(torch.uint8)
               .numpy().tobytes())
        pad = (-offset) % 16  # 16B-align every tensor start (elem_size <= 8)
        if pad:
            blob_parts.append(b"\x00" * pad)
            offset += pad
        entry = {
            "dtype": str(t.dtype).replace("torch.", ""),
            "shape": list(t.shape),
            "format": fmt,
            "offset": offset,
            "nbytes": len(raw),
        }
        if save_dtype is not None:
            entry["save_dtype"] = save_dtype
            entry["save_shape"] = save_shape
        blob_parts.append(raw)
        return entry, offset + len(raw)

    def save_weights_v2(self, model: nn.Module,
                        fingerprint: Optional[dict] = None,
                        removed_params: Optional[list] = None,
                        scalar_attrs: Optional[dict] = None,
                        pre_shapes: Optional[dict] = None,
                        added_params: Optional[list] = None):
        """Serialize POST-process weights to in-memory blob_parts + manifest.

        D2H and serialization run synchronously (caller's context, before
        CUDA graph capture starts). Disk write is deferred to
        _v2_write_snapshot_async so it never touches the device.

        NZ tensors are serialized as ND-logical bytes (transdata via .cpu());
        restore re-casts to NZ, matching the fresh path exactly (byte-level
        roundtrip validated).
        Captures everything process_weights_after_loading produces across the
        three state_dict-invisible carriers: params/buffers (kind:"param"),
        plain tensor attrs (kind:"attr"), tensors inside non-Module attribute
        objects such as MLA's impl (kind:"obj_attr"), plus attrs nulled by
        process (null_attrs).  Also records params deleted by process
        (removed_params) and non-tensor scalar side effects (scalar_attrs)
        so restore can replay them.

        Returns (blob_parts, manifest) for the async writer.
        """
        import hashlib
        rank = torch.distributed.get_rank()
        manifest_path, blob_path = self._v2_snapshot_paths()
        entries: dict[str, dict] = {}
        blob_parts: list[bytes] = []
        offset = 0

        # (1) params/buffers visible to state_dict.
        for name, t in model.state_dict().items():
            entry, offset = self._v2_append_tensor(t, blob_parts, offset)
            entry["kind"] = "param"
            entries[name] = entry

        # (2) Plain tensor attributes set by process_weights_after_loading
        # (e.g. aclnn_input_scale_reciprocal = 1 / Parameter(...): the division
        # result is a plain Tensor in module __dict__, invisible to
        # state_dict). Capture them so restore can skip process entirely.
        for mod_name, mod in model.named_modules(remove_duplicate=False):
            for attr, v in list(mod.__dict__.items()):
                if isinstance(v, (list, tuple, dict)):
                    # Tensor containers are invisible to both state_dict and
                    # this scan -> snapshot would silently lose them. Warn so
                    # coverage gaps are loud. nn.Module internals (_parameters/
                    # _buffers/...) are containers by design, covered above.
                    if attr.startswith("_"):
                        continue
                    vals = v.values() if isinstance(v, dict) else v
                    # empty tensors are placeholders (e.g. Attention.kv_cache
                    # pre-bind) -- no data to lose, runtime rebinds them
                    n_tensors = sum(1 for x in vals
                                    if isinstance(x, torch.Tensor) and x.numel() > 0)
                    if n_tensors:
                        logger.warning(
                            "v2 save: %s.%s is a container holding %d tensor(s); "
                            "containers are NOT captured -- verify restore coverage",
                            mod_name or "<root>", attr, n_tensors)
                    continue
                if not isinstance(v, torch.Tensor):
                    continue
                # Empty tensors are runtime placeholders (e.g.
                # Attention.kv_cache pre-bind, shape [0], device=npu):
                # no data to persist and the runtime rebinds them.  Skipping
                # avoids writing nbytes=0 entries that crash restore's
                # torch.frombuffer(count=0).
                if v.numel() == 0:
                    continue
                key = f"{mod_name}.{attr}" if mod_name else attr
                if key in entries:
                    continue  # already captured via state_dict (params/buffers)
                entry, offset = self._v2_append_tensor(v, blob_parts, offset)
                entry.update({"kind": "attr", "device": v.device.type})
                entries[key] = entry

        # (3) Tensors inside plain (non-Module) attribute objects: e.g. MLA's
        # AscendMLAImpl lives in MLAAttention.__dict__['impl'] but is NOT an
        # nn.Module, so it is invisible to named_modules. Its process products
        # (W_UK_T/W_UV/gamma1/wu_q/...) must be captured via the carrier.
        for mod_name, mod in model.named_modules(remove_duplicate=False):
            for attr, v in list(mod.__dict__.items()):
                if (isinstance(v, torch.nn.Module) or isinstance(v, torch.Tensor)
                        or isinstance(v, type) or callable(v)
                        or not hasattr(v, "__dict__")):
                    continue
                for sub, sv in list(vars(v).items()):
                    if isinstance(sv, (list, tuple, dict)):
                        vals = sv.values() if isinstance(sv, dict) else sv
                        n_tensors = sum(1 for x in vals if isinstance(x, torch.Tensor))
                        if n_tensors:
                            logger.warning(
                                "v2 save: carrier %s.%s holds container attr '%s' "
                                "with %d tensor(s); containers are NOT captured",
                                mod_name or "<root>", attr, sub, n_tensors)
                        continue
                    if not isinstance(sv, torch.Tensor):
                        continue
                    # Skip empty runtime placeholders (same rationale as
                    # the plain-attr branch above).
                    if sv.numel() == 0:
                        continue
                    entry, offset = self._v2_append_tensor(sv, blob_parts, offset)
                    key = f"{mod_name}.{attr}.{sub}" if mod_name else f"{attr}.{sub}"
                    entry.update({
                        "kind": "obj_attr",
                        "carrier": f"{mod_name}.{attr}" if mod_name else attr,
                        "attr": sub,
                        "device": sv.device.type,
                    })
                    entries[key] = entry

        # (4) Params removed by process_weights_after_loading via
        # `layer.x = None` (e.g. MLA sets fused_qkv_a_proj.deq_scale = None
        # after folding it into deq_scale_qkv): they exist in a fresh model's
        # state_dict but not in the saved one. Record them so restore can
        # replay the removal instead of failing the coverage check.
        null_attrs = []
        for mod_name, mod in model.named_modules(remove_duplicate=False):
            for attr, v in mod.__dict__.items():
                if v is None:
                    null_attrs.append(f"{mod_name}.{attr}" if mod_name else attr)

        h = hashlib.sha256()
        for raw in blob_parts:
            h.update(raw)
        manifest = {
            "schema_version": 2,
            "format_note": "ND tensors: raw bytes; NZ tensors: ND-logical bytes, restore must re-cast",
            "world_rank": rank,
            "world_size": torch.distributed.get_world_size(),
            "fingerprint": fingerprint,
            "sha256": h.hexdigest(),
            "tensors": entries,
            "null_attrs": null_attrs,
            "removed_params": removed_params or [],
            "scalar_attrs": scalar_attrs or {},
            "pre_shapes": pre_shapes or {},
            # Explicit allow-list of params created by
            # process_weights_after_loading (post - pre state_dict keys).
            # restore only dynamically registers entries on this list;
            # anything else missing from the model is a ghost entry and
            # fails loud. None -> [] (no process-added params recorded).
            "added_params": added_params if added_params is not None else [],
        }
        logger.info("save_weights_v2 rank=%d tensors=%d bytes=%d sha256=%s "
                    "(serialized, awaiting disk write)",
                    rank, len(entries), offset, manifest["sha256"][:16])
        return blob_parts, manifest

    def _v2_write_snapshot_async(self, blob_parts: list, manifest: dict):
        """Background-thread disk writer: pure CPU/IO, no device ops.

        Writes blob (tmp+fsync+rename) then manifest (tmp+rename) atomically.
        On failure no rename happens, so next startup falls back to fresh.
        """
        try:
            manifest_path, blob_path = self._v2_snapshot_paths()
            tmp_blob = blob_path + ".tmp"
            os.makedirs(os.path.dirname(blob_path) or ".", exist_ok=True)
            # Remove stale .tmp residue from a previous convert that was
            # killed mid-write (the atomic rename never happened, so these
            # are by definition junk -- up to a full blob size squatting on
            # disk until manually cleaned). A fresh convert would truncate
            # them anyway, but a startup that reuses an existing snapshot
            # never rewrites, leaving the residue behind forever.
            for stale in (tmp_blob, manifest_path + ".tmp"):
                if os.path.exists(stale):
                    logger.warning(
                        "v2 snapshot: removing stale tmp file %s left by "
                        "an interrupted convert", stale)
                    os.remove(stale)
            with open(tmp_blob, "wb") as f:
                for raw in blob_parts:
                    f.write(raw)
                f.flush()
                os.fsync(f.fileno())
            os.rename(tmp_blob, blob_path)
            with open(manifest_path + ".tmp", "w") as f:
                json.dump(manifest, f)
            os.rename(manifest_path + ".tmp", manifest_path)
            logger.info("save_weights_v2 disk write complete rank=%d "
                        "tensors=%d sha256=%s",
                        manifest["world_rank"],
                        len(manifest["tensors"]),
                        manifest["sha256"][:16])
        except Exception:
            logger.exception(
                "v2 background snapshot disk write FAILED; "
                "no snapshot written, next startup falls back to "
                "fresh load+process. "
                "Tip: check disk space (df -h), write permissions, "
                "and inode limits on the checkpoint path.")

    def _save_weights_v2_async(self, model: nn.Module,
                               fingerprint: Optional[dict] = None,
                               removed_params: Optional[list] = None,
                               scalar_attrs: Optional[dict] = None,
                               pre_shapes: Optional[dict] = None,
                               added_params: Optional[list] = None):
        """Background-thread wrapper (legacy: D2H + write in one call).

        .. deprecated:: Now that D2H is split out (save_weights_v2 returns
            blob_parts + manifest synchronously, _v2_write_snapshot_async
            handles disk I/O), this method is only used by tests that call
            save_weights_v2 + restore in a single-threaded context.
        """
        try:
            blob_parts, manifest = self.save_weights_v2(
                model, fingerprint, removed_params, scalar_attrs, pre_shapes,
                added_params)
            self._v2_write_snapshot_async(blob_parts, manifest)
        except Exception:
            logger.exception(
                "v2 background snapshot save FAILED; no snapshot written, "
                "next startup falls back to fresh load+process")

    @staticmethod
    def _v2_capture_scalar_state(model: nn.Module) -> Dict[str, Any]:
        """Capture picklable scalar attributes on every module.

        process_weights_after_loading may produce non-tensor side effects
        (e.g. w4a4 sets aclnn_clip_ratio from clip_ratio.item()).  This
        snapshot is used to diff pre- vs post-process state so those
        changes can be replayed during restore without re-running process.
        """
        state: Dict[str, Any] = {}
        for mod_name, mod in model.named_modules(remove_duplicate=False):
            for attr, v in mod.__dict__.items():
                if attr.startswith("_"):
                    continue
                if isinstance(v, (int, float, str, bool)):
                    key = f"{mod_name}.{attr}" if mod_name else attr
                    state[key] = v
        return state

    @staticmethod
    def _v2_check_fingerprint(saved_fp: Optional[dict],
                              fingerprint: Optional[dict],
                              manifest_path: str) -> None:
        """Refuse to restore a snapshot converted under a different config
        (model/dtype/quant/parallel layout)."""
        if saved_fp is None:
            logger.warning("v2 restore: snapshot predates config fingerprinting; "
                           "skipping (model/parallel layout) validation")
            return
        if fingerprint is None:
            return
        mismatched = {
            k: (saved_fp.get(k), fingerprint.get(k))
            for k in set(saved_fp) | set(fingerprint)
            if saved_fp.get(k) != fingerprint.get(k)
        }
        if mismatched:
            raise RuntimeError(
                f"v2 restore: snapshot fingerprint mismatch {mismatched}; "
                f"the snapshot at {os.path.dirname(manifest_path)} was converted "
                f"under a different config -- use a fresh CHECKPOINT_PATH and "
                f"re-run convert")

    @staticmethod
    def _v2_plan_bulk(tensors: Dict[str, dict],
                      blob_len: int) -> tuple[bool, int, int, int]:
        """Decide bulk-chunk H2D parameters.

        Returns (bulk, chunk_size_bytes, dev_bytes, nz_bytes). When bulk is
        enabled the chunk is sized so peak footprint stays at
        weights + one chunk (the previous chunk is released before the next is
        allocated and the caching allocator reuses it); margin covers NZ cast
        temporaries / frac-fill expansion / framework overhead. Budget under
        1GB falls back to per-tensor (peak = single tensor).
        VLLM_ASCEND_BIGTENSOR_BULK_CHUNK_MB overrides dynamic sizing.
        """
        if not envs.VLLM_ASCEND_BIGTENSOR_BULK_H2D:
            return False, 0, 0, 0
        dev_entries = [e for e in tensors.values() if e.get("device") != "cpu"]
        dev_bytes = sum(e["nbytes"] for e in dev_entries)
        nz_bytes = sum(e["nbytes"] for e in dev_entries if e["format"] == 29)
        # If the user explicitly set VLLM_ASCEND_BIGTENSOR_BULK_CHUNK_MB,
        # use that value; otherwise dynamically size based on free HBM.
        env_chunk_raw = os.getenv("VLLM_ASCEND_BIGTENSOR_BULK_CHUNK_MB")
        if env_chunk_raw is not None:
            try:
                chunk_size = envs.VLLM_ASCEND_BIGTENSOR_BULK_CHUNK_MB * 1024 * 1024
            except ValueError:
                raise RuntimeError(
                    f"VLLM_ASCEND_BIGTENSOR_BULK_CHUNK_MB must be an integer "
                    f"(MB), got {env_chunk_raw!r}")
        else:
            free_b, _ = torch.npu.mem_get_info()
            margin = max(4 * 1024**3, free_b // 10) + nz_bytes // 4
            budget = free_b - dev_bytes - margin
            if budget < 1024**3:
                logger.warning(
                    "v2 bulk: dynamic chunk budget %.2fGB "
                    "(free %.2fGB - weights %.2fGB - margin %.2fGB) "
                    "< 1GB, falling back to per-tensor restore",
                    budget / 1024**3, free_b / 1024**3,
                    dev_bytes / 1024**3, margin / 1024**3)
                return False, 0, dev_bytes, nz_bytes
            chunk_size = min(budget, blob_len)
        logger.info("v2 bulk: weights=%.2fGB(nz=%.2fGB) chunk=%.2fGB",
                    dev_bytes / 1024**3, nz_bytes / 1024**3,
                    chunk_size / 1024**3)
        return True, chunk_size, dev_bytes, nz_bytes

    def _v2_resolve_target(self, name: str, e: dict, param_dict: Dict,
                           name_to_module: Dict, target_device,
                           pre_shapes: Optional[dict] = None,
                           added_params: Optional[Set[str]] = None
                           ) -> tuple:
        """Resolve where a manifest entry belongs.

        Returns (param, module, attr_name): for kind "param" the tensor is
        placed via param.set_(); for "attr"/"obj_attr" via setattr(module,
        attr_name, ...). The two non-None cases are mutually exclusive.
        """
        kind = e.get("kind")
        if kind not in ("attr", "obj_attr"):
            dtype = getattr(torch, e["dtype"], None)
            dtype_key = self._DTYPE_KEY.get(dtype) if dtype is not None else None
            if dtype_key is None:
                raise RuntimeError(
                    f"v2 restore: dtype {e['dtype']} not in _DTYPE_MAP for {name}")
            param = self._resolve_param(
                name, param_dict, name_to_module, target_device, dtype_key,
                pre_shapes, added_params)
            if param is None:
                # Stale entry: no matching param in model. Don't crash here --
                # the reverse coverage check (_v2_check_coverage) will report
                # it with a clear business error after the restore loop.
                return None, None, None
            return param, None, None
        if kind == "attr":
            # plain module attribute (process_weights_after_loading product,
            # not a param/buffer): setattr back on the owning module
            mod_name, _, attr_name = name.rpartition(".")
            module = name_to_module.get(mod_name)
            if module is None:
                raise RuntimeError(
                    f"v2 restore: module {mod_name} for attr {name} not found")
            return None, module, attr_name
        # kind == "obj_attr": tensor inside a plain (non-Module) attribute
        # object, e.g. MLAAttention.__dict__['impl'].W_UK_T. Resolve the
        # carrier by longest module prefix + getattr chain.
        carrier_segs = e["carrier"].split(".")
        obj = None
        for i in range(len(carrier_segs), 0, -1):
            cand = ".".join(carrier_segs[:i])
            if cand in name_to_module:
                obj = name_to_module[cand]
                for s in carrier_segs[i:]:
                    obj = getattr(obj, s, None)
                    if obj is None:
                        break
                break
        if obj is None:
            raise RuntimeError(
                f"v2 restore: carrier {e['carrier']} for {name} not found")
        return None, obj, e["attr"]

    @staticmethod
    def _v2_check_coverage(model: nn.Module, tensors: Dict,
                           null_attrs: Set[str], name_to_module: Dict,
                           removed_params: Set[str]) -> None:
        """Every persistent param/buffer (state_dict key) must come from the
        snapshot. Exempt:
        - non-persistent buffers (e.g. rotary cos_sin_cache, computed at
          runtime),
        - attrs nulled by process (null_attrs, e.g. MLA deq_scale),
        - params deleted by process (removed_params, e.g. w4a4 replaces
          layer.weight with layer.weight_packed).
        Replay the nulling/deletion instead of failing.

        Reverse check: snapshot entries that have no corresponding model
        param/buffer (excluding attr/obj_attr kinds, which are not in
        state_dict) indicate structure drift and are warned about."""
        model_keys = set(model.state_dict().keys())
        # (1) forward: model params missing from snapshot
        missing = [n for n in model_keys if n not in tensors]
        exempt = null_attrs | removed_params
        unexpected = [n for n in missing if n not in exempt]
        if unexpected:
            raise RuntimeError(
                f"v2 restore: {len(unexpected)} model params not in snapshot, "
                f"first few: {unexpected[:5]} (re-run convert)")
        # (2) reverse: snapshot "param" entries not in model (structure
        # drift -- e.g. layers removed from code but snapshot not
        # regenerated).  attr/obj_attr entries are excluded because they
        # are not in state_dict by definition.
        stale = [n for n, e in tensors.items()
                 if n not in model_keys
                 and e.get("kind") == "param"
                 and n not in exempt]
        if stale:
            raise RuntimeError(
                f"v2 restore: {len(stale)} snapshot params not found in "
                f"model, first few: {stale[:5]}. The snapshot was generated "
                f"for a different model structure -- re-run convert.")
        for n in missing:
            mod_name, _, attr_name = n.rpartition(".")
            module = name_to_module.get(mod_name)
            if module is None:
                continue
            if n in removed_params and hasattr(module, attr_name):
                delattr(module, attr_name)
            elif n in null_attrs:
                setattr(module, attr_name, None)

    @staticmethod
    def _v2_verify_blob(mm: "mmap.mmap", manifest: dict,
                        manifest_path: str) -> float:
        """Fail-loud integrity gate run before restoring any tensor.

        Returns elapsed time (seconds) for logging.

        A fast-restore snapshot is only useful if it is trustworthy: a
        corrupted/half-written blob must never be silently restored into
        wrong weights (silent corruption is the worst failure mode for a
        cache). The verification level is controlled by
        ``VLLM_ASCEND_BIGTENSOR_VERIFY``:

        - ``size`` (default): per-entry bounds validation only. O(n_entries)
          and sub-millisecond -- catches truncated/half-written snapshots,
          the most common corruption mode. Does NOT detect byte-level
          corruption (disk bit-flips), which is rare but catastrophic.
        - ``sha256``: bounds + whole-blob sha256. Streams the blob through
          ``hashlib`` in 128 MiB chunks; on a ~31GB blob this costs
          tens of seconds (single-core ~0.6GB/s on aarch64), which can
          negate the fast-restore speedup. Use for audits / untrusted
          storage / after copying snapshots across machines.
        - ``none``: no verification (trust storage entirely).
        """
        verify_mode = envs.VLLM_ASCEND_BIGTENSOR_VERIFY
        if verify_mode == "none":
            logger.info("v2 restore: blob verification disabled "
                        "(VLLM_ASCEND_BIGTENSOR_VERIFY=none)")
            return 0.0

        blob_len = len(mm)
        t0 = time.perf_counter()

        # (1) per-entry bounds validation -- always run for size/sha256.
        # Empty placeholder entries (nbytes=0, e.g. legacy kv_cache) never
        # read blob bytes, so skip them to avoid false "truncated" alarms
        # on their stale offset values.
        n_entries = 0
        for name, e in manifest["tensors"].items():
            if e["nbytes"] == 0:
                continue
            n_entries += 1
            end = e["offset"] + e["nbytes"]
            if e["offset"] < 0 or end > blob_len:
                raise RuntimeError(
                    f"v2 restore: snapshot blob is truncated or corrupt -- "
                    f"tensor '{name}' spans bytes [{e['offset']}, {end}) but "
                    f"the blob is only {blob_len} bytes. Delete the snapshot "
                    f"at {os.path.dirname(manifest_path)} and re-run convert.")

        if verify_mode == "size":
            logger.info("v2 restore: blob bounds verified (%d tensors, "
                        "%.3fs)", n_entries, time.perf_counter() - t0)
            return time.perf_counter() - t0

        if verify_mode != "sha256":
            raise RuntimeError(
                f"v2 restore: invalid VLLM_ASCEND_BIGTENSOR_VERIFY="
                f"{verify_mode!r}; must be one of: none, size, sha256")

        # (2) whole-blob sha256, streamed in chunks (no full-file copy)
        saved_sha = manifest.get("sha256")
        if not saved_sha:
            logger.warning(
                "v2 restore: VLLM_ASCEND_BIGTENSOR_VERIFY=sha256 but "
                "manifest has no sha256 field (pre-checksum snapshot); "
                "skipping sha256")
            return time.perf_counter() - t0
        import hashlib
        h = hashlib.sha256()
        chunk_size = 128 * 1024 * 1024  # 128 MiB
        for pos in range(0, blob_len, chunk_size):
            h.update(mm[pos:pos + chunk_size])
        actual_sha = h.hexdigest()
        if actual_sha != saved_sha:
            raise RuntimeError(
                f"v2 restore: blob sha256 mismatch -- manifest records "
                f"{saved_sha[:16]}... but actual is {actual_sha[:16]}...; "
                f"the snapshot at {os.path.dirname(manifest_path)} is corrupt "
                f"or was only partially written. Delete it and re-run convert "
                f"rather than restoring corrupt weights.")
        logger.info("v2 restore: blob integrity verified (sha256=%s, "
                    "%.2fs)", actual_sha[:16], time.perf_counter() - t0)
        return time.perf_counter() - t0

    @staticmethod
    def _v2_shape_allowed(shape: tuple, pre_shape: tuple,
                           dtype: torch.dtype) -> bool:
        """Whitelist of legitimate post-process shape transforms.

        Ground truth measured on W8A8 + MoE (r9: 480 entries, 6 distinct
        transforms). The whitelist intentionally stays a closed set: a
        same-numel shape morph outside this set (B3) is treated as
        manifest tampering and fails loud.
        1. identical: NZ cast and dtype reinterpretation keep the shape;
        2. squeeze/unsqueeze equivalence: identical after stripping all
           size-1 dims -- the quantizer builds (N,1) scale placeholders,
           process squeezes them to (N,). Contiguous storage layouts are
           byte-identical, so this is safe for any dtype;
        3. 2D transpose: W8A8 per-channel int8 weight transposition.
           int8 only -- float weights are never transposed by process
           (r9 E1 probe: an fp16 (16,8)->(8,16) morph is an attack);
        4. 3D dim0-preserving swap of the last two dims: MoE expert
           w13/w2 permute (E,A,B) -> (E,B,A). int8 only.
        """
        if shape == pre_shape:
            return True
        strip = tuple(d for d in shape if d != 1)
        strip_pre = tuple(d for d in pre_shape if d != 1)
        if strip == strip_pre:
            return True
        if dtype == torch.int8:
            if len(pre_shape) == 2 and shape == pre_shape[::-1]:
                return True
            if (len(shape) == 3 and len(pre_shape) == 3
                    and shape[0] == pre_shape[0]
                    and shape[1] == pre_shape[2]
                    and shape[2] == pre_shape[1]):
                return True
        return False

    def restore_weights_v2(self, model: nn.Module, target_device,
                           fingerprint: Optional[dict] = None) -> None:
        """Restore post-process weights from the v2 snapshot; skip
        process_weights_after_loading entirely.

        Each restored tensor is a standalone, contiguous, offset-0 copy:
        aclnn quantized matmuls reject deq_scale passed as a nonzero-
        storage-offset view (561103 NULLPTR) and npu_format_cast on such a
        view sizes its output by the whole underlying storage, so clone-
        then-cast is mandatory. A device-resident blob is therefore never
        kept, giving a footprint identical to the fresh path (and avoiding
        permanently pinning one blob's worth of HBM per rank).

        When VLLM_ASCEND_BIGTENSOR_BULK_H2D=1 the blob is H2D'd in GB-level chunks
        (sorted by offset) and tensors are cloned out of the chunk, which
        keeps peak memory at weights + one chunk instead of 2x weights.
        """
        manifest_path, blob_path = self._v2_snapshot_paths()
        with open(manifest_path) as f:
            manifest = json.load(f)
        rank = torch.distributed.get_rank()
        self._v2_check_fingerprint(
            manifest.get("fingerprint"), fingerprint, manifest_path)

        with open(blob_path, "rb") as f:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                # Fail-loud integrity gate: verify blob bounds/sha256 BEFORE
                # restoring any tensor, so a corrupt/truncated snapshot is
                # never silently loaded into wrong weights. Timed separately
                # from H2D so restore overhead is attributable.
                verify_time = self._v2_verify_blob(mm, manifest, manifest_path)
                h2d_start = time.perf_counter()
                bulk, chunk_size, _, _ = self._v2_plan_bulk(
                    manifest["tensors"], len(mm))
                # Bulk requires offset-ordered consumption; save already
                # writes in offset order, but sort defensively.
                tensor_items = (
                    sorted(manifest["tensors"].items(),
                           key=lambda kv: kv[1]["offset"])
                    if bulk else manifest["tensors"].items())
                param_dict = dict(model.named_parameters(remove_duplicate=False))
                param_dict.update(
                    dict(model.named_buffers(remove_duplicate=False)))
                name_to_module = dict(
                    model.named_modules(remove_duplicate=False))
                pre_shapes_map = manifest.get("pre_shapes") or {}
                # None (key absent) -> legacy manifest, _resolve_param falls
                # back to the pre_shapes heuristic; a set (possibly empty)
                # -> strict allow-list.
                added_params_set = (
                    set(manifest["added_params"])
                    if manifest.get("added_params") is not None else None)

                dev_chunk = None   # current device-side bulk buffer
                chunk_base = 0     # blob offset where dev_chunk starts
                chunk_end = 0      # blob offset where dev_chunk ends
                bulk_h2d = 0.0
                n_chunks = 0
                n_nd = n_nz = 0

                for name, e in tensor_items:
                    dtype = getattr(torch, e["dtype"], None)
                    if dtype is None:
                        raise RuntimeError(
                            f"v2 restore: unsupported dtype {e['dtype']} for {name}")
                    shape = tuple(e["shape"])
                    elem_size = torch.empty((), dtype=dtype).element_size()
                    if e["offset"] % elem_size != 0:
                        raise RuntimeError(
                            f"v2 restore: unaligned offset for {name}")
                    # Descriptor self-consistency: the manifest's nbytes
                    # must match the logical shape * elem_size (for ND) or
                    # the save_shape * save_dtype elem_size (for packed NZ).
                    # This blocks B3-style tampering (same nbytes, different
                    # shape) that sha256 (blob-only) cannot detect.
                    if e["nbytes"] != 0:
                        if "save_dtype" in e:
                            sd = getattr(torch, e["save_dtype"])
                            ss = tuple(e["save_shape"])
                            n_elems = 1
                            for d in ss:
                                n_elems *= int(d)
                            # torch.dtype has no element_size(); go through
                            # a scalar tensor like the ND branch above.
                            expected = n_elems * torch.empty(
                                (), dtype=sd).element_size()
                        else:
                            n_elems = 1
                            for d in shape:
                                n_elems *= int(d)
                            expected = n_elems * elem_size
                        if e["nbytes"] != expected:
                            raise RuntimeError(
                                f"v2 restore: descriptor inconsistency for "
                                f"{name}: nbytes={e['nbytes']} but "
                                f"shape {shape} dtype {e['dtype']} implies "
                                f"{expected} bytes. Manifest may be corrupt "
                                f"or tampered -- re-run convert.")
                    # B3 hardening: a same-numel shape morph (e.g.
                    # [4,4]->[2,8]) is self-consistent in nbytes and the
                    # blob sha256 covers only the blob, so anchor the
                    # manifest's logical shape against pre_shapes: the
                    # post-process shape must be a whitelisted transform
                    # of the recorded pre-process shape (see
                    # _v2_shape_allowed: equal, squeeze/unsqueeze, int8
                    # 2D transpose, int8 MoE 3D swap -- r9 ground truth).
                    # Packed entries (save_dtype) and process-added params
                    # (not in pre_shapes) are exempt.
                    pre_entry = pre_shapes_map.get(name)
                    if pre_entry is not None and "save_dtype" not in e:
                        pre_shape = tuple(pre_entry[0])
                        if not self._v2_shape_allowed(
                                shape, pre_shape, dtype):
                            raise RuntimeError(
                                f"v2 restore: shape whitelist mismatch for "
                                f"{name}: manifest shape {shape} is not a "
                                f"legitimate post-process transform of the "
                                f"recorded pre-process shape {pre_shape} "
                                f"(allowed: equal, squeeze/unsqueeze, int8 "
                                f"2D transpose, int8 MoE 3D swap); the "
                                f"manifest may be tampered -- re-run convert.")
                    param, module, attr_name = self._v2_resolve_target(
                        name, e, param_dict, name_to_module, target_device,
                        pre_shapes_map, added_params_set)
                    kind = e.get("kind")
                    # Stale entry (no matching param in model): skip and let
                    # the reverse coverage check report it after the loop.
                    if param is None and module is None and attr_name is None:
                        continue

                    with torch.no_grad():
                        # Packed low-bit NZ entries store int8 bytes plus the
                        # logical int32 dtype/shape; build the int8 tensor
                        # first, then view back to logical dtype after the
                        # NZ cast (see _v2_is_packed_nz).
                        save_dtype = getattr(
                            torch, e.get("save_dtype", e["dtype"]))
                        save_shape = tuple(
                            e.get("save_shape", e["shape"]))
                        if e["nbytes"] == 0:
                            # Empty placeholder (e.g. Attention.kv_cache
                            # pre-bind, shape [0]): no bytes in the blob, so
                            # torch.frombuffer(count=0) would raise. Reconstruct
                            # an empty tensor directly from the manifest's
                            # dtype/shape/device; the runtime rebinds it.
                            # (Backward-compat for snapshots written before
                            # the save-side numel()==0 skip was added.)
                            tensor = torch.empty(save_shape, dtype=save_dtype)
                            if e.get("device") != "cpu":
                                tensor = tensor.to(target_device)
                                if e["format"] == 29:
                                    tensor = torch_npu.npu_format_cast(tensor, 29)
                                    n_nz += 1
                                else:
                                    n_nd += 1
                        elif bulk and e.get("device") != "cpu":
                            # Ensure this tensor falls within the loaded chunk;
                            # release the previous chunk and load a new one
                            # starting at the tensor when it does not.
                            if (dev_chunk is None
                                    or e["offset"] + e["nbytes"] > chunk_end):
                                dev_chunk = None  # release before realloc
                                chunk_base = e["offset"]
                                # A chunk always covers at least the current
                                # tensor (a single tensor larger than chunk
                                # size wins), bounded by end-of-file.
                                chunk_end = min(
                                    max(chunk_base + chunk_size,
                                        e["offset"] + e["nbytes"]),
                                    len(mm))
                                t0 = time.perf_counter()
                                src = torch.frombuffer(
                                    mm, dtype=torch.uint8,
                                    count=chunk_end - chunk_base,
                                    offset=chunk_base)
                                dev_chunk = torch.empty(
                                    chunk_end - chunk_base,
                                    dtype=torch.uint8, device=target_device)
                                dev_chunk.copy_(src)
                                bulk_h2d += time.perf_counter() - t0
                                n_chunks += 1
                            view = dev_chunk.narrow(
                                0, e["offset"] - chunk_base, e["nbytes"]
                            ).view(save_dtype).view(save_shape)
                            if e["format"] == 29:
                                # clone before cast: cast on a nonzero-offset
                                # view allocates its output against the whole
                                # underlying storage, and transdata's input
                                # must be a standalone tensor anyway.
                                tensor = torch_npu.npu_format_cast(
                                    view.clone(), 29)
                                n_nz += 1
                            else:
                                # clone required: aclnn rejects nonzero-offset
                                # views; final weights must be offset-0.
                                tensor = view.clone()
                                n_nd += 1
                        else:
                            cpu_bytes = torch.frombuffer(
                                mm, dtype=torch.uint8,
                                count=e["nbytes"], offset=e["offset"])
                            tensor = cpu_bytes.view(save_dtype).view(save_shape)
                            if e.get("device") == "cpu":
                                # CPU-resident plain attr: clone out of the
                                # mmap so the mapping can be released.
                                tensor = tensor.clone()
                            else:
                                tensor = tensor.to(target_device)
                                if e["format"] == 29:
                                    # NZ: re-cast from ND-logical bytes.
                                    tensor = torch_npu.npu_format_cast(tensor, 29)
                                    n_nz += 1
                                else:
                                    n_nd += 1
                        if "save_dtype" in e:
                            # Packed low-bit: mirror the fresh path's
                            # ``view(torch.int32).contiguous()`` pack step.
                            # set_ keeps the NZ format metadata.
                            tensor = tensor.view(dtype).contiguous()
                        if kind in ("attr", "obj_attr"):
                            # Verify shape/dtype consistency: a stale
                            # snapshot (e.g. model code changed, same
                            # byte-count but different layout) must fail
                            # loud instead of silently installing a
                            # deformed tensor.
                            current = getattr(module, attr_name, None)
                            if isinstance(current, torch.Tensor) and (
                                    current.shape != tensor.shape
                                    or current.dtype != tensor.dtype):
                                raise RuntimeError(
                                    f"v2 restore: shape/dtype mismatch for "
                                    f"{name}: model has "
                                    f"{tuple(current.shape)}/{current.dtype},"
                                    f" snapshot has "
                                    f"{tuple(tensor.shape)}/{tensor.dtype}."
                                    f" Re-run convert.")
                            setattr(module, attr_name, tensor)
                        elif "save_dtype" in e or dtype != param.dtype:
                            # Packed low-bit entries and dtype-changing
                            # process products: the placeholder param keeps
                            # its create-time dtype (int8, float32, ...) while
                            # the restored tensor carries the post-process
                            # dtype (packed int32 view; per-channel w4a8
                            # reinterprets float32 scale bytes as int64).
                            # set_ forbids the dtype change; the fresh path
                            # uses plain ``.data`` assignment
                            # (layer.w13_weight_scale.data = ...), which
                            # permits it -- mirror that here.
                            # Skip shape check for empty placeholders
                            # (numel==0): process fills them with the real
                            # tensor; the placeholder shape is irrelevant.
                            if param.numel() != 0 and tensor.shape != param.shape:
                                raise RuntimeError(
                                    f"v2 restore: shape mismatch for {name}:"
                                    f" model has {tuple(param.shape)},"
                                    f" snapshot has {tuple(tensor.shape)}."
                                    f" Re-run convert.")
                            param.data = tensor
                        else:
                            # Standard param: if the manifest recorded the
                            # pre-process placeholder shape/dtype, validate
                            # the *current* model placeholder against the
                            # recorded one -- this detects real structure
                            # drift (model code changed) without false-
                            # positive on legitimate process transforms
                            # (W8A8 transpose, NZ cast).
                            # If pre_shapes is absent (old snapshot), fall
                            # back to comparing snapshot tensor vs current
                            # placeholder, skipping empty placeholders and
                            # 2D transposed int8 NZ weights (known legit).
                            pre = manifest.get("pre_shapes", {}).get(name)
                            if pre is not None:
                                pre_shape, pre_dtype = pre[0], pre[1]
                                if (tuple(param.shape) != tuple(pre_shape)
                                        or str(param.dtype).replace(
                                            "torch.", "") != pre_dtype):
                                    raise RuntimeError(
                                        f"v2 restore: placeholder mismatch "
                                        f"for {name}: model has "
                                        f"{tuple(param.shape)}/{param.dtype},"
                                        f" snapshot recorded "
                                        f"{tuple(pre_shape)}/{pre_dtype}."
                                        f" Re-run convert.")
                            elif param.numel() != 0:
                                # No pre_shapes (old snapshot): allow known
                                # legitimate transforms, fail on the rest.
                                is_transpose = (
                                    len(tensor.shape) == 2
                                    and len(param.shape) == 2
                                    and tensor.shape == param.shape[::-1])
                                if (not is_transpose
                                        and (tensor.shape != param.shape
                                             or tensor.dtype != param.dtype)):
                                    raise RuntimeError(
                                        f"v2 restore: shape/dtype mismatch "
                                        f"for {name}: model has "
                                        f"{tuple(param.shape)}/{param.dtype},"
                                        f" snapshot has "
                                        f"{tuple(tensor.shape)}/{tensor.dtype}."
                                        f" Re-run convert.")
                            param.set_(tensor)

                # Every tensor has been cloned into standalone storage; the
                # last chunk can be released before the coverage check.
                dev_chunk = None
                # Replay non-tensor scalar side effects produced by process
                # (e.g. w4a4's aclnn_clip_ratio).
                for name, value in manifest.get("scalar_attrs", {}).items():
                    mod_name, _, attr_name = name.rpartition(".")
                    module = name_to_module.get(mod_name)
                    if module is not None:
                        setattr(module, attr_name, value)
                self._v2_check_coverage(
                    model, manifest["tensors"],
                    set(manifest.get("null_attrs", ())),
                    name_to_module,
                    set(manifest.get("removed_params", ())))
                h2d_time = time.perf_counter() - h2d_start
        logger.info("restore_weights_v2 rank=%d mode=%s nd=%d nz=%d "
                    "chunks=%d bulk_h2d=%.2fs verify=%.2fs h2d=%.2fs",
                    rank, "bulk" if bulk else "per-tensor",
                    n_nd, n_nz, n_chunks, bulk_h2d, verify_time, h2d_time)

    def load_model(self, vllm_config: VllmConfig, model_config: ModelConfig, prefix: str = "") -> nn.Module:
        """Load a model with the given configurations.

        VLLM_ASCEND_BIGTENSOR_V2=1: v2 snapshot mechanism -- snapshot present ->
        restore-only (process skipped); absent -> fresh load + process +
        background snapshot save.  Without the env: plain vLLM default
        load (fresh load + process on every startup), no snapshot I/O.
        """
        device_config = vllm_config.device_config
        load_config = vllm_config.load_config
        load_device = (
            device_config.device if load_config.device is None else load_config.device
        )
        target_device = torch.device(load_device)

        v2_enabled = envs.VLLM_ASCEND_BIGTENSOR_V2
        if v2_enabled:
            fingerprint = self._v2_fingerprint(vllm_config, model_config)
            # Single uniform command: no snapshot -> load+process+async save;
            # snapshot present -> restore-only. The snapshot write runs in a
            # background thread so ~1-2min of disk I/O never blocks engine
            # init (weights are read-only after process; concurrent inference
            # only contends for D2H bandwidth).
            with set_default_torch_dtype(model_config.dtype):
                with target_device:
                    model = initialize_model(vllm_config=vllm_config, model_config=model_config, prefix=prefix)
                if self._has_v2_snapshot():
                    # restore-only path: snapshot is post-process -> skip process
                    self.restore_weights_v2(model, target_device, fingerprint)
                    return model.eval()
                # convert path: full load + process; snapshot save is async
                self.load_weights(model, model_config)
                pre_params = set(model.state_dict().keys())
                # Capture pre-process placeholder shape/dtype per param so
                # restore can validate "model placeholder matches what
                # create_weights produced" rather than "snapshot tensor
                # matches placeholder" -- process_weights_after_loading
                # legitimately changes shape (W8A8 transpose) and dtype
                # (packed int32 view), which is not structure drift.
                pre_shapes = {
                    n: (tuple(t.shape), str(t.dtype).replace("torch.", ""))
                    for n, t in model.state_dict().items()
                }
                pre_scalar = self._v2_capture_scalar_state(model)
                process_weights_after_loading(model, model_config, target_device)
                post_params = set(model.state_dict().keys())
                post_scalar = self._v2_capture_scalar_state(model)
                removed_params = list(pre_params - post_params)
                # Params created by process (post - pre state_dict keys,
                # e.g. W8A8's 96 aclnn_input_scale). restore dynamically
                # registers ONLY entries on this list; a manifest entry
                # missing from the model but not on the list is a ghost
                # and fails loud instead of being silently registered.
                added_params = list(post_params - pre_params)
                scalar_attrs = {k: v for k, v in post_scalar.items()
                                if pre_scalar.get(k) != v}
                # D2H + serialize synchronously (before graph capture starts
                # in warmup). This is the part that touches the device and
                # would collide with aclrtMemcpy-in-capture (EE1016).
                blob_parts, manifest = self.save_weights_v2(
                    model, fingerprint, removed_params, scalar_attrs,
                    pre_shapes, added_params)
                # Disk write is pure CPU/IO -- safe to background.
                threading.Thread(
                    target=self._v2_write_snapshot_async,
                    args=(blob_parts, manifest),
                    name="v2-snapshot-write",
                    daemon=True,
                ).start()
                logger.info("v2 snapshot save dispatched to background thread; "
                            "startup continues without waiting")
            return model.eval()

        with set_default_torch_dtype(model_config.dtype):
            with target_device:
                model = initialize_model(vllm_config=vllm_config, model_config=model_config, prefix=prefix)

            # Quantization does not happen in `load_weights` but after it
            self.load_weights(model, model_config)
            process_weights_after_loading(model, model_config, target_device)

        return model.eval()
