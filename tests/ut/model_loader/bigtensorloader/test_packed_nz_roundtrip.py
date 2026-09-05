# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Round-trip test for packed low-bit NZ snapshot save/restore.

w4a8/w4a16 pack quantized int8 NZ weights via
``int8_nz.view(torch.int32).contiguous()``. The logical int32 dtype no
longer matches the int8 NZ physical layout, so a direct D2H transdata
fails with a fatal EZ9999. The v2 snapshot must save these through the
int8 view (manifest records save_dtype/save_shape) and restore must
rebuild the int8 NZ tensor and view it back to int32, byte-identical to
the fresh path.
"""

import json
import os
import tempfile
import unittest
from unittest.mock import patch

import torch
import torch_npu  # noqa: F401

from vllm.config import LoadConfig

from vllm_ascend.model_loader.bigtensorloader.bigtensorloader import (
    BigTensorLoader)

ACL_FORMAT_FRACTAL_NZ = 29


def _make_packed_int32_nz(shape_int8: tuple) -> torch.Tensor:
    """Build a packed weight exactly as w4a8's process does:
    int8 -> contiguous -> NZ cast -> view(int32).contiguous().

    For 2D (linear) shapes, apply the transpose the quant method does;
    3D shapes are the FusedMoE w13/w2 layout ([E, N, K])."""
    t = torch.randint(-127, 127, shape_int8, dtype=torch.int8)
    if t.dim() == 2:
        t = t.transpose(1, 0).contiguous().transpose(1, 0).contiguous()
    else:
        t = t.contiguous()
    t = t.to("npu")
    t = torch_npu.npu_format_cast(t, ACL_FORMAT_FRACTAL_NZ)
    return t.view(torch.int32).contiguous()


class _PackedModel(torch.nn.Module):

    def __init__(self, packed: torch.Tensor):
        super().__init__()
        self.register_parameter(
            "w_packed",
            torch.nn.Parameter(packed, requires_grad=False))
        # an ordinary bf16 ND param alongside, to exercise mixed entries
        self.register_parameter(
            "w_plain",
            torch.nn.Parameter(
                torch.randn(8, 16, dtype=torch.bfloat16),
                requires_grad=False))


class TestPackedNzSnapshotRoundtrip(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not torch.npu.is_available():
            raise unittest.SkipTest("NPU not available")
        # NZ format casts require internal formats, as in model_runner_v1
        torch.npu.config.allow_internal_format = True
        torch.npu.set_device(0)

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp(prefix="v2_packed_nz_ut_")
        self._old_ckpt = os.environ.get("VLLM_ASCEND_CHECKPOINT_PATH")
        os.environ["VLLM_ASCEND_CHECKPOINT_PATH"] = self._tmpdir
        self._old_verify = os.environ.get("VLLM_ASCEND_BIGTENSOR_VERIFY")
        os.environ.pop("VLLM_ASCEND_BIGTENSOR_VERIFY", None)  # default to "size"
        self.loader = BigTensorLoader(
            LoadConfig(load_format="bigtensorloader"))

    def tearDown(self):
        if self._old_ckpt is None:
            os.environ.pop("VLLM_ASCEND_CHECKPOINT_PATH", None)
        else:
            os.environ["VLLM_ASCEND_CHECKPOINT_PATH"] = self._old_ckpt
        if self._old_verify is None:
            os.environ.pop("VLLM_ASCEND_BIGTENSOR_VERIFY", None)
        else:
            os.environ["VLLM_ASCEND_BIGTENSOR_VERIFY"] = self._old_verify
        for fname in os.listdir(self._tmpdir):
            os.remove(os.path.join(self._tmpdir, fname))
        os.rmdir(self._tmpdir)

    def _roundtrip_once(self, shape_int8: tuple, bulk: bool) -> None:
        packed = _make_packed_int32_nz(shape_int8)
        self.assertEqual(
            int(torch_npu.get_npu_format(packed)), ACL_FORMAT_FRACTAL_NZ)
        self.assertEqual(packed.dtype, torch.int32)

        model = _PackedModel(packed).to("npu")
        with patch("torch.distributed.get_rank", return_value=0), \
                patch("torch.distributed.get_world_size", return_value=1):
            self.loader._save_weights_v2_async(model)

        manifest_path = os.path.join(self._tmpdir, "0.v2.json")
        with open(manifest_path) as f:
            manifest = json.load(f)
        entry = manifest["tensors"]["w_packed"]
        self.assertEqual(entry["save_dtype"], "int8")
        self.assertEqual(entry["dtype"], "int32")
        self.assertEqual(entry["format"], ACL_FORMAT_FRACTAL_NZ)
        self.assertEqual(entry["nbytes"],
                         packed.numel() * packed.element_size())
        # logical int32 shape: last dim divided by 4
        self.assertEqual(entry["shape"][-1], shape_int8[-1] // 4)
        # plain bf16 entry must not carry save_dtype
        self.assertNotIn("save_dtype", manifest["tensors"]["w_plain"])

        # fresh model with empty placeholders, as the serve path
        # creates them when a snapshot exists
        restored = _PackedModel(torch.empty(0, dtype=torch.int32)).to("npu")
        old_bulk = os.environ.get("VLLM_ASCEND_BIGTENSOR_BULK_H2D")
        old_chunk = os.environ.get("VLLM_ASCEND_BIGTENSOR_BULK_CHUNK_MB")
        os.environ["VLLM_ASCEND_BIGTENSOR_BULK_H2D"] = "1" if bulk else "0"
        if bulk:
            # force the bulk branch: dynamic chunk sizing falls back to
            # per-tensor when free device memory leaves <1GB of budget
            os.environ["VLLM_ASCEND_BIGTENSOR_BULK_CHUNK_MB"] = "1"
        try:
            self.loader.restore_weights_v2(restored, torch.device("npu"))
        finally:
            for key, old in (("VLLM_ASCEND_BIGTENSOR_BULK_H2D", old_bulk),
                             ("VLLM_ASCEND_BIGTENSOR_BULK_CHUNK_MB", old_chunk)):
                if old is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = old

        got = restored.w_packed.data
        self.assertEqual(got.dtype, torch.int32)
        self.assertEqual(tuple(got.shape), tuple(packed.shape))
        self.assertEqual(
            int(torch_npu.get_npu_format(got)), ACL_FORMAT_FRACTAL_NZ)
        # byte-level equality against the fresh-path tensor
        self.assertTrue(torch.equal(
            got.view(torch.int8), packed.view(torch.int8)))
        self.assertTrue(torch.equal(
            restored.w_plain.data, model.w_plain.data))

    @patch("torch.distributed.get_world_size", return_value=1)
    @patch("torch.distributed.get_rank", return_value=0)
    def test_packed_int32_nz_roundtrip_per_tensor(self, _r, _w):
        # shapes with last dim not divisible by 16: NZ physical layout has
        # padding, which the int8-view D2H must strip correctly.
        # the 3D shape is the FusedMoE w13/w2 layout.
        for shape_int8 in ((64, 256), (48, 132), (16, 68), (4, 32, 128)):
            with self.subTest(shape_int8=shape_int8):
                self._roundtrip_once(shape_int8, bulk=False)

    @patch("torch.distributed.get_world_size", return_value=1)
    @patch("torch.distributed.get_rank", return_value=0)
    def test_packed_int32_nz_roundtrip_bulk(self, _r, _w):
        # bulk chunked H2D path must rebuild packed tensors identically;
        # the chunk covers the whole blob here, so the packed tensor is
        # cloned out of a nonzero-offset view before the NZ cast
        for shape_int8 in ((64, 256), (4, 32, 128)):
            with self.subTest(shape_int8=shape_int8):
                self._roundtrip_once(shape_int8, bulk=True)

    def test_is_packed_nz_detection(self):
        packed = _make_packed_int32_nz((16, 64))
        self.assertTrue(BigTensorLoader._v2_is_packed_nz(packed))
        # plain int8 NZ: dtype matches its storage, direct D2H is safe
        plain = torch.randint(-127, 127, (16, 64), dtype=torch.int8,
                              device="npu")
        plain = torch_npu.npu_format_cast(plain, ACL_FORMAT_FRACTAL_NZ)
        self.assertFalse(BigTensorLoader._v2_is_packed_nz(plain))
        # int32 ND: not NZ at all
        nd = torch.zeros(4, 8, dtype=torch.int32, device="npu")
        self.assertFalse(BigTensorLoader._v2_is_packed_nz(nd))

    @patch("torch.distributed.get_world_size", return_value=1)
    @patch("torch.distributed.get_rank", return_value=0)
    def test_empty_placeholder_attr_not_saved(self, _r, _w):
        """Empty runtime placeholders (e.g. Attention.kv_cache pre-bind,
        shape [0]) must NOT be written to the snapshot. They carry no data
        and would crash restore's ``torch.frombuffer(count=0)``; the runtime
        rebinds them anyway."""
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_parameter(
                    "w",
                    torch.nn.Parameter(torch.randn(4, 4), requires_grad=False))
                # empty placeholder attr, like Attention.kv_cache before bind
                self.kv_cache = torch.empty(0, dtype=torch.float16, device="npu")

        model = M().to("npu")
        self.loader._save_weights_v2_async(model)

        manifest_path = os.path.join(self._tmpdir, "0.v2.json")
        with open(manifest_path) as f:
            manifest = json.load(f)
        # empty placeholder must not appear in the snapshot
        self.assertNotIn("kv_cache", manifest["tensors"])
        # real weight is captured with nonzero bytes
        self.assertIn("w", manifest["tensors"])
        self.assertGreater(manifest["tensors"]["w"]["nbytes"], 0)

        # restore must not crash; kv_cache stays empty (runtime rebinds it)
        restored = M().to("npu")
        self.loader.restore_weights_v2(restored, torch.device("npu"))
        self.assertEqual(restored.kv_cache.numel(), 0)
        self.assertTrue(torch.equal(restored.w.data, model.w.data))

    @patch("torch.distributed.get_world_size", return_value=1)
    @patch("torch.distributed.get_rank", return_value=0)
    def test_restore_legacy_zero_nbytes_entry(self, _r, _w):
        """Snapshots written before the save-side ``numel()==0`` skip may
        contain ``nbytes=0`` entries. Restore must rebuild an empty tensor
        directly instead of calling ``torch.frombuffer(count=0)`` (which
        raises ``ValueError``)."""
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_parameter(
                    "w",
                    torch.nn.Parameter(torch.randn(4, 4), requires_grad=False))

        model = M().to("npu")
        self.loader._save_weights_v2_async(model)

        manifest_path = os.path.join(self._tmpdir, "0.v2.json")
        with open(manifest_path) as f:
            manifest = json.load(f)

        # Inject a legacy nbytes=0 attr entry (offset is irrelevant for
        # zero-byte entries; use a past-end value to prove no blob read).
        w_entry = manifest["tensors"]["w"]
        manifest["tensors"]["legacy_empty_attr"] = {
            "dtype": "float16",
            "shape": [0],
            "format": 0,
            "offset": w_entry["offset"] + w_entry["nbytes"] + 1024,
            "nbytes": 0,
            "kind": "attr",
            "device": "npu",
        }
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)

        restored = M().to("npu")
        # provide the attr target so _v2_resolve_target can find it
        restored.legacy_empty_attr = torch.empty(
            0, dtype=torch.float16, device="npu")
        # must not raise ValueError from torch.frombuffer(count=0)
        self.loader.restore_weights_v2(restored, torch.device("npu"))
        self.assertEqual(restored.legacy_empty_attr.numel(), 0)
        self.assertTrue(torch.equal(restored.w.data, model.w.data))

    @patch("torch.distributed.get_world_size", return_value=1)
    @patch("torch.distributed.get_rank", return_value=0)
    def test_corrupt_blob_sha256_fails_loud(self, _r, _w):
        """A byte-flipped blob must be rejected by sha256 verification
        (when VLLM_ASCEND_BIGTENSOR_VERIFY=sha256) instead of being silently
        restored into corrupt weights."""
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_parameter(
                    "w",
                    torch.nn.Parameter(torch.randn(8, 8), requires_grad=False))

        model = M().to("npu")
        self.loader._save_weights_v2_async(model)

        blob_path = os.path.join(self._tmpdir, "0.v2.snapshot")
        with open(blob_path, "r+b") as f:
            f.seek(64)
            (b,) = f.read(1)
            f.seek(64)
            f.write(bytes([b ^ 0xFF]))

        restored = M().to("npu")
        old_verify = os.environ.get("VLLM_ASCEND_BIGTENSOR_VERIFY")
        os.environ["VLLM_ASCEND_BIGTENSOR_VERIFY"] = "sha256"
        try:
            with self.assertRaisesRegex(RuntimeError, "sha256 mismatch"):
                self.loader.restore_weights_v2(restored, torch.device("npu"))
        finally:
            if old_verify is None:
                os.environ.pop("VLLM_ASCEND_BIGTENSOR_VERIFY", None)
            else:
                os.environ["VLLM_ASCEND_BIGTENSOR_VERIFY"] = old_verify

    @patch("torch.distributed.get_world_size", return_value=1)
    @patch("torch.distributed.get_rank", return_value=0)
    def test_corrupt_blob_size_mode_passes_silently(self, _r, _w):
        """In default VLLM_ASCEND_BIGTENSOR_VERIFY=size mode, a byte-flip inside
        the bounds is NOT detected (bounds check only catches truncation).
        This documents the trade-off: size is fast but does not catch
        byte-level corruption. Use sha256 for that."""
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_parameter(
                    "w",
                    torch.nn.Parameter(torch.randn(8, 8), requires_grad=False))

        model = M().to("npu")
        self.loader._save_weights_v2_async(model)

        blob_path = os.path.join(self._tmpdir, "0.v2.snapshot")
        with open(blob_path, "r+b") as f:
            f.seek(64)
            (b,) = f.read(1)
            f.seek(64)
            f.write(bytes([b ^ 0xFF]))

        restored = M().to("npu")
        # default mode (size) -- no exception, but weights are wrong
        old_verify = os.environ.get("VLLM_ASCEND_BIGTENSOR_VERIFY")
        os.environ["VLLM_ASCEND_BIGTENSOR_VERIFY"] = "size"
        try:
            self.loader.restore_weights_v2(restored, torch.device("npu"))
            # restored weights are silently corrupt (byte-flipped)
            self.assertFalse(torch.equal(restored.w.data, model.w.data))
        finally:
            if old_verify is None:
                os.environ.pop("VLLM_ASCEND_BIGTENSOR_VERIFY", None)
            else:
                os.environ["VLLM_ASCEND_BIGTENSOR_VERIFY"] = old_verify

    @patch("torch.distributed.get_world_size", return_value=1)
    @patch("torch.distributed.get_rank", return_value=0)
    def test_truncated_blob_bounds_check_fails_loud(self, _r, _w):
        """A half-truncated blob must be rejected by the per-entry bounds
        check with a clear 'truncated' error, not a low-level
        torch.frombuffer ValueError."""
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_parameter(
                    "w",
                    torch.nn.Parameter(torch.randn(64, 64), requires_grad=False))

        model = M().to("npu")
        self.loader._save_weights_v2_async(model)

        blob_path = os.path.join(self._tmpdir, "0.v2.snapshot")
        size = os.path.getsize(blob_path)
        with open(blob_path, "r+b") as f:
            f.truncate(size // 2)

        restored = M().to("npu")
        with self.assertRaisesRegex(RuntimeError, "truncated or corrupt"):
            self.loader.restore_weights_v2(restored, torch.device("npu"))

    @patch("torch.distributed.get_world_size", return_value=1)
    @patch("torch.distributed.get_rank", return_value=0)
    def test_shape_drift_fails_loud(self, _r, _w):
        """A snapshot entry with tampered shape (same byte count, different
        layout) must be rejected at restore time, not silently installed
        as a deformed weight."""
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_parameter(
                    "w",
                    torch.nn.Parameter(torch.randn(4, 4), requires_grad=False))

        model = M().to("npu")
        self.loader._save_weights_v2_async(model)

        # Tamper manifest: change [4,4] to [2,8] (same nbytes, different shape)
        manifest_path = os.path.join(self._tmpdir, "0.v2.json")
        with open(manifest_path) as f:
            manifest = json.load(f)
        manifest["tensors"]["w"]["shape"] = [2, 8]
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)

        restored = M().to("npu")
        with self.assertRaisesRegex(RuntimeError, "shape.*mismatch"):
            self.loader.restore_weights_v2(restored, torch.device("npu"))

    @patch("torch.distributed.get_world_size", return_value=1)
    @patch("torch.distributed.get_rank", return_value=0)
    def test_dtype_drift_fails_loud(self, _r, _w):
        """A snapshot entry with tampered dtype (same byte count, different
        dtype) must be rejected at restore time."""
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_parameter(
                    "w",
                    torch.nn.Parameter(torch.randn(4, 4, dtype=torch.float16),
                                      requires_grad=False))

        model = M().to("npu")
        self.loader._save_weights_v2_async(model)

        # Tamper manifest: change float16 [4,4] (32B) to float32 [8] (32B)
        manifest_path = os.path.join(self._tmpdir, "0.v2.json")
        with open(manifest_path) as f:
            manifest = json.load(f)
        manifest["tensors"]["w"]["dtype"] = "float32"
        manifest["tensors"]["w"]["shape"] = [8]
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)

        restored = M().to("npu")
        with self.assertRaisesRegex(RuntimeError, "shape.*mismatch"):
            self.loader.restore_weights_v2(restored, torch.device("npu"))

    @patch("torch.distributed.get_world_size", return_value=1)
    @patch("torch.distributed.get_rank", return_value=0)
    def test_stale_snapshot_entry_fails_loud(self, _r, _w):
        """A snapshot param entry that no longer exists in the model
        (structure drift) must be rejected by the reverse coverage
        check, not silently registered as a new param."""
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_parameter(
                    "w",
                    torch.nn.Parameter(torch.randn(4, 4), requires_grad=False))

        model = M().to("npu")
        self.loader._save_weights_v2_async(model)

        # Inject a stale param entry into the manifest
        manifest_path = os.path.join(self._tmpdir, "0.v2.json")
        with open(manifest_path) as f:
            manifest = json.load(f)
        manifest["tensors"]["ghost_layer.weight"] = {
            "dtype": "float32", "shape": [2, 2], "format": 0,
            "offset": 0, "nbytes": 16, "kind": "param", "device": "npu"}
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)

        restored = M().to("npu")
        with self.assertRaisesRegex(RuntimeError, "snapshot params not found"):
            self.loader.restore_weights_v2(restored, torch.device("npu"))

    @patch("torch.distributed.get_world_size", return_value=1)
    @patch("torch.distributed.get_rank", return_value=0)
    def test_manifest_nbytes_inconsistency_fails_loud(self, _r, _w):
        """B3b: tampering manifest shape so nbytes no longer matches
        prod(shape)*elem_size must be caught by descriptor self-
        consistency. (Not the B3 same-numel morph -- see
        test_manifest_same_numel_shape_morph_fails_loud.)"""
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_parameter(
                    "w",
                    torch.nn.Parameter(torch.randn(4, 4), requires_grad=False))

        model = M().to("npu")
        self.loader._save_weights_v2_async(model)

        # Tamper: change shape [4,4] -> [3,8] (nbytes stays 64 but
        # 3*8*4=96 != 64 -- descriptor inconsistency detected)
        manifest_path = os.path.join(self._tmpdir, "0.v2.json")
        with open(manifest_path) as f:
            manifest = json.load(f)
        entry = manifest["tensors"]["w"]
        entry["shape"] = [3, 8]  # nbytes=64 but 3*8*4=96
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)

        restored = M().to("npu")
        with self.assertRaisesRegex(RuntimeError,
                                    "descriptor inconsistency"):
            self.loader.restore_weights_v2(restored, torch.device("npu"))

    @patch("torch.distributed.get_world_size", return_value=1)
    @patch("torch.distributed.get_rank", return_value=0)
    def test_manifest_same_numel_shape_morph_fails_loud(self, _r, _w):
        """B3 (r7/r8 original attack): a same-numel shape morph
        ([4,4] -> [2,8], nbytes identical, blob untouched) passes
        descriptor self-consistency and blob sha256; it must be
        rejected by the pre_shapes shape whitelist (equal or 2D
        transpose only) instead of being silently installed."""
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_parameter(
                    "w",
                    torch.nn.Parameter(torch.randn(4, 4, dtype=torch.float16),
                                      requires_grad=False))

        model = M().to("npu")
        # Record pre_shapes like the load_model convert path does.
        pre_shapes = {"w": ((4, 4), "float16")}
        self.loader._save_weights_v2_async(model, None, None, None,
                                           pre_shapes)

        # Tamper: [4,4] -> [2,8] fp16 (same numel=16, same nbytes=32,
        # self-consistent -- only the whitelist can catch it)
        manifest_path = os.path.join(self._tmpdir, "0.v2.json")
        with open(manifest_path) as f:
            manifest = json.load(f)
        self.assertEqual(manifest["tensors"]["w"]["shape"], [4, 4])
        manifest["tensors"]["w"]["shape"] = [2, 8]
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)

        restored = M().to("npu")
        with self.assertRaisesRegex(RuntimeError,
                                    "shape whitelist mismatch"):
            self.loader.restore_weights_v2(restored, torch.device("npu"))

    @patch("torch.distributed.get_world_size", return_value=1)
    @patch("torch.distributed.get_rank", return_value=0)
    def test_ghost_entry_on_existing_module_fails_loud(self, _r, _w):
        """B2b: a ghost manifest entry attached to an EXISTING module
        (root) must NOT be dynamically registered. Only entries on the
        save-side added_params list may register; new manifests carry
        an explicit (possibly empty) added_params list, so ghosts fail
        loud via the reverse coverage check."""
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_parameter(
                    "w",
                    torch.nn.Parameter(torch.randn(4, 4), requires_grad=False))

        model = M().to("npu")
        self.loader._save_weights_v2_async(model)

        manifest_path = os.path.join(self._tmpdir, "0.v2.json")
        with open(manifest_path) as f:
            manifest = json.load(f)
        # Ghost entry on the root module (module exists, attribute
        # missing, and it is NOT on the added_params list).
        manifest["tensors"]["ghost_w"] = {
            "dtype": "float32", "shape": [2, 2], "format": 0,
            "offset": 0, "nbytes": 16, "kind": "param", "device": "npu"}
        # New save writes added_params=[] (strict mode).
        self.assertEqual(manifest.get("added_params"), [])
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)

        restored = M().to("npu")
        with self.assertRaisesRegex(RuntimeError,
                                    "snapshot params not found"):
            self.loader.restore_weights_v2(restored, torch.device("npu"))
        # The ghost must not have been registered on the model.
        self.assertFalse(any("ghost_w" in n
                             for n, _ in restored.named_parameters()))

    @patch("torch.distributed.get_world_size", return_value=1)
    @patch("torch.distributed.get_rank", return_value=0)
    def test_process_added_param_registers_and_fills(self, _r, _w):
        """Positive control for added_params: an entry that the save side
        recorded as process-added (e.g. W8A8 aclnn_input_scale) IS
        dynamically registered on the module and filled from the blob."""
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_parameter(
                    "w",
                    torch.nn.Parameter(torch.randn(4, 4), requires_grad=False))

        model = M().to("npu")
        self.loader._save_weights_v2_async(model)

        manifest_path = os.path.join(self._tmpdir, "0.v2.json")
        with open(manifest_path) as f:
            manifest = json.load(f)
        # Simulate a process-added param: a distinct entry sharing w's
        # bytes, explicitly recorded on the added_params list.
        manifest["tensors"]["scale_added"] = dict(manifest["tensors"]["w"])
        manifest["added_params"] = ["scale_added"]
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)

        restored = M().to("npu")  # fresh model: no scale_added attribute
        self.loader.restore_weights_v2(restored, torch.device("npu"))
        self.assertTrue(hasattr(restored, "scale_added"))
        self.assertEqual(tuple(restored.scale_added.shape), (4, 4))
        self.assertTrue(torch.equal(restored.scale_added, restored.w))

    def test_shape_whitelist_transform_table(self):
        """r9 ground truth: the six legitimate W8A8+MoE transforms
        (480 manifest entries) must pass; known attack morphs must
        fail. Direct helper-level table, mirrors the r9 report."""
        ok = self.loader._v2_shape_allowed
        # squeeze/unsqueeze equivalence (384 scale entries, bfloat16)
        self.assertTrue(ok((5120,), (5120, 1), torch.bfloat16))
        self.assertTrue(ok((2048,), (2048, 1), torch.bfloat16))
        self.assertTrue(ok((128, 1536), (128, 1536, 1), torch.bfloat16))
        self.assertTrue(ok((128, 2048), (128, 2048, 1), torch.bfloat16))
        # MoE expert 3D last-two-dims swap (96 entries, int8)
        self.assertTrue(ok((128, 2048, 1536), (128, 1536, 2048), torch.int8))
        self.assertTrue(ok((128, 768, 2048), (128, 2048, 768), torch.int8))
        # 2D transpose (int8)
        self.assertTrue(ok((2048, 5120), (5120, 2048), torch.int8))
        # identical (any dtype)
        self.assertTrue(ok((4, 4), (4, 4), torch.float16))
        # attacks: B3 same-numel morph, E1 float transpose, float MoE3
        # swap, dim0-changing 3D swap
        self.assertFalse(ok((2, 8), (4, 4), torch.float16))
        self.assertFalse(ok((8, 16), (16, 8), torch.float16))
        self.assertFalse(ok((128, 2048, 1536), (128, 1536, 2048),
                            torch.bfloat16))
        self.assertFalse(ok((96, 2048, 1536), (128, 1536, 2048), torch.int8))

    @patch("torch.distributed.get_world_size", return_value=1)
    @patch("torch.distributed.get_rank", return_value=0)
    def test_fp16_transpose_morph_fails_loud(self, _r, _w):
        """E1: a float16 (16,8)->(8,16) morph is exactly the 2D
        transpose with identical numel/nbytes -- descriptor and blob
        checks cannot catch it. Legitimate transposes only occur on
        int8 weights, so the whitelist must reject it."""
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_parameter(
                    "w",
                    torch.nn.Parameter(
                        torch.randn(16, 8, dtype=torch.float16),
                        requires_grad=False))

        model = M().to("npu")
        pre_shapes = {"w": ((16, 8), "float16")}
        self.loader._save_weights_v2_async(model, None, None, None,
                                           pre_shapes)

        # Tamper to the exact transpose (nbytes 16*8*2 == 8*16*2)
        manifest_path = os.path.join(self._tmpdir, "0.v2.json")
        with open(manifest_path) as f:
            manifest = json.load(f)
        manifest["tensors"]["w"]["shape"] = [8, 16]
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)

        restored = M().to("npu")
        with self.assertRaisesRegex(RuntimeError,
                                    "shape whitelist mismatch"):
            self.loader.restore_weights_v2(restored, torch.device("npu"))

    @patch("torch.distributed.get_world_size", return_value=1)
    @patch("torch.distributed.get_rank", return_value=0)
    def test_squeeze_transform_restores(self, _r, _w):
        """Positive control for the squeeze rule: quantizer-style (N,1)
        bfloat16 placeholder, manifest records the process-squeezed (N,)
        shape -- restore must succeed and install the (N,) view."""
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_parameter(
                    "w",
                    torch.nn.Parameter(
                        torch.randn(64, 1, dtype=torch.bfloat16),
                        requires_grad=False))

        model = M().to("npu")
        pre_shapes = {"w": ((64, 1), "bfloat16")}
        self.loader._save_weights_v2_async(model, None, None, None,
                                           pre_shapes)

        # Simulate the process squeeze in the manifest (nbytes equal:
        # 64*1*2 == 64*2)
        manifest_path = os.path.join(self._tmpdir, "0.v2.json")
        with open(manifest_path) as f:
            manifest = json.load(f)
        manifest["tensors"]["w"]["shape"] = [64]
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)

        restored = M().to("npu")
        self.loader.restore_weights_v2(restored, torch.device("npu"))
        self.assertEqual(tuple(restored.w.shape), (64,))

    @patch("torch.distributed.get_world_size", return_value=1)
    @patch("torch.distributed.get_rank", return_value=0)
    def test_stale_tmp_residue_cleaned_on_write(self, _r, _w):
        """r10 operational finding: a convert killed mid-disk-write leaves
        .tmp residue. The next disk write must remove it (a startup that
        reuses an existing snapshot never rewrites, so the junk would
        squat on disk indefinitely) and still produce a valid snapshot."""
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_parameter(
                    "w",
                    torch.nn.Parameter(torch.randn(4, 4), requires_grad=False))

        # Simulate residue from an interrupted convert
        with open(os.path.join(self._tmpdir, "0.v2.snapshot.tmp"),
                  "wb") as f:
            f.write(b"junk-junk-junk")
        with open(os.path.join(self._tmpdir, "0.v2.json.tmp"), "w") as f:
            f.write("{junk")

        model = M().to("npu")
        self.loader._save_weights_v2_async(model)

        self.assertFalse(
            os.path.exists(os.path.join(self._tmpdir, "0.v2.snapshot.tmp")))
        self.assertFalse(
            os.path.exists(os.path.join(self._tmpdir, "0.v2.json.tmp")))
        # A valid snapshot was still written
        self.assertTrue(
            os.path.exists(os.path.join(self._tmpdir, "0.v2.json")))
        self.assertTrue(
            os.path.exists(os.path.join(self._tmpdir, "0.v2.snapshot")))
        restored = M().to("npu")
        self.loader.restore_weights_v2(restored, torch.device("npu"))
        self.assertTrue(torch.equal(restored.w, model.w))


if __name__ == "__main__":
    unittest.main()
