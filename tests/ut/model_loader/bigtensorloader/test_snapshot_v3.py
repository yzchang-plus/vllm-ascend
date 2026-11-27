# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""v3 snapshot schema: container capture, alias dedup, schema gate, fallback.

These tests pin the reviewer-driven correctness upgrades of PR #16734:

* containers (list/tuple/dict, also nested / inside carrier objects) are
  captured with JSON-serializable access chains and recreated on restore --
  v2 silently skipped them;
* aliased tensors (tied weights, re-referenced derived tensors) are stored
  once and re-shared on restore -- v2 duplicated them and broke tying; stride
  is part of the dedup identity so a transposed view of a square weight is
  stored separately instead of aliasing its base;
* manifests older than schema v3 are rejected (regenerated via fallback);
* per-tensor spot hashes catch in-bounds tampering of small tensors;
* restore failures make load_model preserve shared snapshots for concurrent
  readers and fall back to a fresh load + process instead of crashing or
  serving garbage.
"""

import hashlib
import json
import os
import shutil
import tempfile
import threading
import time
import unittest
import weakref
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import torch
import torch_npu  # noqa: F401
from vllm.config import LoadConfig

from vllm_ascend.model_loader.bigtensorloader import bigtensorloader as bigtensorloader_module
from vllm_ascend.model_loader.bigtensorloader.bigtensorloader import ACL_FORMAT_FRACTAL_NZ_C0_16, BigTensorLoader
from vllm_ascend.quantization.nz import fp4_packed_nz


class _Carrier:
    """Plain-object tensor carrier (MLA impl-style)."""

    def __init__(self, w):
        self.w_uv = w
        self._scratch = torch.randn(1)


class _ContainerModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_parameter("w", torch.nn.Parameter(torch.randn(4, 4), requires_grad=False))
        self.caches = [torch.randn(2, 3), {"a.b": torch.randn(3, 2)}]
        self.frozen = (torch.randn(5),)


class _TiedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(8, 4)
        self.head = torch.nn.Linear(4, 8, bias=False)
        self.head.weight = self.emb.weight  # tied


class _CarrierModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_parameter("w", torch.nn.Parameter(torch.randn(4, 4), requires_grad=False))
        self.impl = _Carrier(torch.randn(6, 2))


class _BareModel(torch.nn.Module):
    """Restore target: only the param; all extra attrs must be recreated."""

    def __init__(self):
        super().__init__()
        self.register_parameter("w", torch.nn.Parameter(torch.randn(4, 4), requires_grad=False))


class TestSnapshotV3(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            if not torch.npu.is_available():
                raise unittest.SkipTest("NPU not available")
            torch.npu.config.allow_internal_format = True
            torch.npu.set_device(0)
            torch.zeros(1, device="npu")  # probe real device access
        except unittest.SkipTest:
            raise
        except Exception as e:
            raise unittest.SkipTest(f"NPU not available: {e}") from e

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp(prefix="snapshot_v3_ut_")
        self.loader = self._make_loader()

    def tearDown(self):
        for fname in os.listdir(self._tmpdir):
            path = os.path.join(self._tmpdir, fname)
            if os.path.isdir(path):
                # fingerprint-namespaced snapshot subdirectories (e.g.
                # load_model tests that pass an explicit fingerprint)
                shutil.rmtree(path)
            else:
                os.remove(path)
        os.rmdir(self._tmpdir)

    def _make_loader(self, **overrides: Any) -> BigTensorLoader:
        extra = {"checkpoint_path": self._tmpdir, **overrides}
        return BigTensorLoader(LoadConfig(load_format="bigtensorloader", model_loader_extra_config=extra))

    # -- helpers -----------------------------------------------------------

    def _save(self, model, fingerprint=None):
        with (
            patch("torch.distributed.get_rank", return_value=0),
            patch("torch.distributed.get_world_size", return_value=1),
        ):
            self.loader._save_snapshot_async(model, fingerprint)

    def _restore(self, model):
        with (
            patch("torch.distributed.get_rank", return_value=0),
            patch("torch.distributed.get_world_size", return_value=1),
        ):
            self.loader.restore_snapshot(model, torch.device("npu"))

    @staticmethod
    def _as_nd(t: torch.Tensor) -> torch.Tensor:
        """Logical ND host bytes of a transposed NZ view (aclnn has no equal for NZ)."""
        return torch_npu.npu_format_cast(t.transpose(0, 1), 2).to("cpu")

    def _manifest(self) -> dict:
        with open(os.path.join(self._tmpdir, "0.json")) as f:
            return json.load(f)

    def _blob_size(self) -> int:
        return os.path.getsize(os.path.join(self._tmpdir, "0.snapshot"))

    # -- container capture -------------------------------------------------

    def test_containers_captured_with_paths_and_defs(self):
        model = _ContainerModel().to("npu")
        self._save(model)
        manifest = self._manifest()

        self.assertEqual(manifest["schema_version"], 3)
        tensors = manifest["tensors"]
        self.assertIn("caches.0", tensors)
        self.assertIn("caches.1.a.b", tensors)
        self.assertIn("frozen.0", tensors)
        # dotted dict keys must survive the (dotted) name and the path
        self.assertEqual(tensors["caches.1.a.b"]["path"], [{"i": 1}, {"k": "a.b"}])
        self.assertEqual(tensors["caches.0"]["path"], [{"i": 0}])
        self.assertEqual(tensors["frozen.0"]["path"], [{"i": 0}])
        # container defs: type + length, keyed module|chain
        defs = manifest["containers"]
        self.assertIn('|[{"a": "caches"}]', defs)
        self.assertEqual(defs['|[{"a": "caches"}]'], {"type": "list", "len": 2})
        self.assertIn('|[{"a": "frozen"}]', defs)
        self.assertEqual(defs['|[{"a": "frozen"}]'], {"type": "tuple", "len": 1})
        self.assertEqual(defs['|[{"a": "caches"}, {"i": 1}]'], {"type": "dict"})

    def test_container_roundtrip_recreates_missing_containers(self):
        model = _ContainerModel().to("npu")
        self._save(model)
        # the (tiny) tensors make the spot-hash threshold irrelevant here
        restored = _BareModel().to("npu")
        self._restore(restored)

        self.assertTrue(torch.equal(restored.caches[0], model.caches[0]))
        self.assertTrue(torch.equal(restored.caches[1]["a.b"], model.caches[1]["a.b"]))
        # tuples were recreated as lists and sealed back to tuples
        self.assertIsInstance(restored.frozen, tuple)
        self.assertTrue(torch.equal(restored.frozen[0], model.frozen[0]))
        self.assertTrue(torch.equal(restored.w.data, model.w.data))

    def test_container_restored_into_existing_containers(self):
        model = _ContainerModel().to("npu")
        self._save(model)
        restored = _ContainerModel().to("npu")  # containers already exist
        self._restore(restored)
        self.assertTrue(torch.equal(restored.caches[0], model.caches[0]))
        self.assertTrue(torch.equal(restored.caches[1]["a.b"], model.caches[1]["a.b"]))
        self.assertTrue(torch.equal(restored.frozen[0], model.frozen[0]))
        self.assertIsInstance(restored.frozen, tuple)  # existing tuple untouched

    def test_nested_container_recreation(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_parameter("w", torch.nn.Parameter(torch.randn(2, 2), requires_grad=False))
                self.grid = [[torch.randn(1), torch.randn(1)], torch.randn(1)]

        model = M().to("npu")
        self._save(model)
        manifest = self._manifest()
        self.assertEqual(manifest["tensors"]["grid.0.1"]["path"], [{"i": 0}, {"i": 1}])
        self.assertIn('|[{"a": "grid"}, {"i": 0}]', manifest["containers"])

        restored = _BareModel().to("npu")
        self._restore(restored)
        self.assertTrue(torch.equal(restored.grid[0][1], model.grid[0][1]))
        self.assertTrue(torch.equal(restored.grid[1], model.grid[1]))

    def test_nested_tuple_sealed_innermost_first(self):
        """Nested tuple containers are sealed innermost-first; inner tuples must not stay lists."""

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_parameter("w", torch.nn.Parameter(torch.randn(4, 4), requires_grad=False))
                self.pair = ((torch.randn(1), torch.randn(1)),)

        model = M().to("npu")
        self._save(model)
        restored = _BareModel().to("npu")
        self._restore(restored)
        self.assertIsInstance(restored.pair, tuple)
        # The inner tuple must survive the roundtrip instead of degrading to a list.
        self.assertIsInstance(restored.pair[0], tuple)
        self.assertTrue(torch.equal(restored.pair[0][0], model.pair[0][0]))
        self.assertTrue(torch.equal(restored.pair[0][1], model.pair[0][1]))

    def test_private_attrs_skipped_like_netloader(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_parameter("w", torch.nn.Parameter(torch.randn(2, 2), requires_grad=False))
                self._runtime_cache = [torch.randn(2)]

        self._save(M().to("npu"))
        manifest = self._manifest()
        self.assertNotIn("_runtime_cache.0", manifest["tensors"])

    # -- carrier objects ----------------------------------------------------

    def test_carrier_tensor_roundtrip(self):
        model = _CarrierModel().to("npu")
        self._save(model)
        manifest = self._manifest()
        self.assertIn("impl.w_uv", manifest["tensors"])
        self.assertEqual(manifest["tensors"]["impl.w_uv"]["path"], [{"a": "w_uv"}])
        # netloader policy: private members of scanned objects are skipped
        self.assertNotIn("impl._scratch", manifest["tensors"])

        restored = _CarrierModel().to("npu")
        self._restore(restored)
        self.assertTrue(torch.equal(restored.impl.w_uv, model.impl.w_uv))
        # _scratch untouched by restore
        self.assertFalse(torch.equal(restored.impl._scratch, model.impl._scratch))

    def test_missing_carrier_fails_loud(self):
        model = _CarrierModel().to("npu")
        self._save(model)
        restored = _BareModel().to("npu")  # no impl carrier
        with self.assertRaisesRegex(RuntimeError, "missing"):
            self._restore(restored)

    # -- alias dedup --------------------------------------------------------

    def test_tied_weights_dedup_and_restay_tied(self):
        model = _TiedModel().to("npu")
        self._save(model)
        manifest = self._manifest()

        emb_entry = manifest["tensors"]["emb.weight"]
        head_entry = manifest["tensors"]["head.weight"]
        self.assertGreater(emb_entry["nbytes"], 0)
        self.assertEqual(head_entry.get("alias_of"), "emb.weight")
        self.assertEqual(head_entry["nbytes"], 0)
        # the tied tensor is stored exactly once
        self.assertEqual(self._blob_size(), emb_entry["nbytes"] + emb_entry["offset"])

        restored = _TiedModel().to("npu")
        # break the tie so restore is what re-establishes it
        with torch.no_grad():
            restored.head.weight = torch.nn.Parameter(
                torch.randn_like(restored.head.weight.data),
                requires_grad=False,
            )
        self._restore(restored)
        self.assertTrue(torch.equal(restored.emb.weight.data, model.emb.weight.data))
        self.assertTrue(torch.equal(restored.head.weight.data, restored.emb.weight.data))
        self.assertEqual(restored.head.weight.data_ptr(), restored.emb.weight.data_ptr())

    def test_alias_tensor_in_two_containers_stored_once(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_parameter("w", torch.nn.Parameter(torch.randn(2, 2), requires_grad=False))
                shared = torch.randn(3)
                self.left = [shared]
                self.right = {"key": shared}

        model = M().to("npu")
        self._save(model)
        manifest = self._manifest()
        self.assertGreater(manifest["tensors"]["left.0"]["nbytes"], 0)
        self.assertEqual(manifest["tensors"]["right.key"]["alias_of"], "left.0")

        restored = _BareModel().to("npu")
        self._restore(restored)
        self.assertTrue(torch.equal(restored.left[0], model.left[0]))
        self.assertTrue(torch.equal(restored.right["key"], model.left[0]))
        # aliasing (shared storage) survives the roundtrip
        self.assertEqual(restored.right["key"].data_ptr(), restored.left[0].data_ptr())

    def test_transposed_square_view_stored_separately(self):
        """A square transposed view must not dedup onto its base tensor.

        data_ptr/dtype/shape/format are identical between a square weight and
        its transpose; only the stride differs. Without stride in the dedup
        key the transpose is recorded as an alias and silently restored as a
        copy of the base, changing the weight.
        """

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_parameter("w", torch.nn.Parameter(torch.randn(4, 4), requires_grad=False))

        model = M().to("npu")
        with torch.no_grad():
            model.w_t = model.w.data.t()  # same ptr/dtype/shape/format, different stride
        self.assertFalse(torch.equal(model.w_t, model.w.data))  # non-symmetric base

        self._save(model)
        manifest = self._manifest()
        # the transpose is its own entry, not an alias of the base
        self.assertNotIn("alias_of", manifest["tensors"]["w_t"])
        self.assertGreater(manifest["tensors"]["w_t"]["nbytes"], 0)

        restored = M().to("npu")  # fresh model: no w_t attribute
        self._restore(restored)
        self.assertTrue(torch.equal(restored.w_t, model.w_t))
        self.assertFalse(torch.equal(restored.w_t, restored.w.data))
        # standalone storage: not the base's tensor installed a second time
        self.assertNotEqual(restored.w_t.data_ptr(), restored.w.data_ptr())

    # -- schema gate ----------------------------------------------------------

    def test_schema_gate_rejects_v2(self):
        model = _BareModel().to("npu")
        self._save(model)
        manifest_path = os.path.join(self._tmpdir, "0.json")
        manifest = self._manifest()
        manifest["schema_version"] = 2
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)

        with self.assertRaisesRegex(RuntimeError, "schema_version"):
            self._restore(_BareModel().to("npu"))

    def test_missing_schema_version_rejected(self):
        model = _BareModel().to("npu")
        self._save(model)
        manifest_path = os.path.join(self._tmpdir, "0.json")
        manifest = self._manifest()
        del manifest["schema_version"]
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)

        with self.assertRaisesRegex(RuntimeError, "schema_version"):
            self._restore(_BareModel().to("npu"))

    # -- drift detection --------------------------------------------------------

    def test_attr_drift_shape_mismatch_fails_loud(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_parameter("w", torch.nn.Parameter(torch.randn(4, 4), requires_grad=False))
                self.scale = torch.randn(8)

        model = M().to("npu")
        self._save(model)

        class Drifted(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_parameter("w", torch.nn.Parameter(torch.randn(4, 4), requires_grad=False))
                self.scale = torch.randn(16)  # shape drifted

        with self.assertRaisesRegex(RuntimeError, "shape/dtype mismatch"):
            self._restore(Drifted().to("npu"))

    # -- spot hash ------------------------------------------------------------

    def test_spot_hash_small_tensor_only(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_parameter(
                    "small",
                    torch.nn.Parameter(torch.randn(4), requires_grad=False),
                )
                self.register_parameter(
                    "big",
                    torch.nn.Parameter(
                        torch.empty(BigTensorLoader._SPOT_HASH_MAX_BYTES + 1, dtype=torch.int8),
                        requires_grad=False,
                    ),
                )

        self._save(M().to("npu"))
        manifest = self._manifest()
        self.assertIn("sha256", manifest["tensors"]["small"])
        self.assertNotIn("sha256", manifest["tensors"]["big"])

    def test_alias_entries_are_not_spot_hashed_twice(self):
        model = _TiedModel().to("npu")
        self._save(model)
        manifest = self._manifest()
        self.assertNotIn("sha256", manifest["tensors"]["head.weight"])

    # -- bulk mode with containers -----------------------------------------------

    def test_bulk_restore_with_containers(self):
        model = _ContainerModel().to("npu")
        self._save(model)
        self.loader = self._make_loader(bulk_h2d=True, bulk_chunk_mb=1)
        restored = _BareModel().to("npu")
        self._restore(restored)
        self.assertTrue(torch.equal(restored.caches[0], model.caches[0]))
        self.assertTrue(torch.equal(restored.frozen[0], model.frozen[0]))
        self.assertTrue(torch.equal(restored.w.data, model.w.data))

    # -- fallback ------------------------------------------------------------------

    @patch("torch.distributed.get_world_size", return_value=1)
    @patch("torch.distributed.get_rank", return_value=0)
    def test_load_model_falls_back_on_truncated_blob(self, _r, _w):
        # fast_restore must be enabled: the test exercises snapshot-restore
        # fallback after truncating the blob, which only happens when the
        # restore branch is entered in load_model.
        self.loader = self._make_loader(fast_restore=True)
        model = _BareModel().to("npu").to("npu")

        created = []
        processed = []

        def fake_process(model, model_config, target_device):
            processed.append(model)

        vllm_config = SimpleNamespace(
            device_config=SimpleNamespace(device="npu"),
            load_config=SimpleNamespace(device=None),
            parallel_config=SimpleNamespace(
                tensor_parallel_size=1,
                pipeline_parallel_size=1,
                enable_expert_parallel=False,
                data_parallel_size=1,
            ),
            additional_config={},
        )
        # a registration owned by a live co-resident model: must survive the fallback
        foreign_module = object()
        forward_context = {"live.layers.0.self_attn.attn": foreign_module}
        moe_layers = ["live.layers.0.mlp.experts"]
        vllm_config.compilation_config = SimpleNamespace(
            static_forward_context=forward_context,
            static_all_moe_layers=moe_layers,
        )
        model_config = SimpleNamespace(model="test-model", dtype=torch.float32, quantization=None)
        fingerprint = self.loader._fingerprint(vllm_config, model_config)
        self._save(model, fingerprint)
        _, blob_path = self.loader._snapshot_paths(fingerprint)
        size = os.path.getsize(blob_path)
        with open(blob_path, "r+b") as f:
            f.truncate(size // 10)
        manifest_path, _ = self.loader._snapshot_paths(fingerprint)
        original_manifest_inode = os.stat(manifest_path).st_ino

        def fake_initialize_model(**kwargs):
            created.append(_BareModel().to("npu").to("npu"))
            # simulate attention/MoE layers registering themselves as built
            forward_context["model.layers.0.self_attn.attn"] = created[-1]
            moe_layers.append("model.layers.0.mlp.experts")
            return created[-1]

        with (
            patch(
                "vllm_ascend.model_loader.bigtensorloader.bigtensorloader.initialize_model",
                side_effect=fake_initialize_model,
            ),
            patch(
                "vllm_ascend.model_loader.bigtensorloader.bigtensorloader.process_weights_after_loading",
                side_effect=fake_process,
            ),
            patch.object(BigTensorLoader, "load_weights", return_value=None),
        ):
            returned = self.loader.load_model(vllm_config, model_config)

        # restore attempt + fallback fresh load
        self.assertEqual(len(created), 2)
        self.assertIs(returned, created[1])
        # the abandoned model's registration was dropped (and the fresh model
        # re-registered the same prefix after the reset -- no "Duplicate layer
        # name"), while the live model's registration survived untouched
        self.assertIs(forward_context["model.layers.0.self_attn.attn"], created[1])
        self.assertIs(forward_context["live.layers.0.self_attn.attn"], foreign_module)
        # MoE registry rolled back to the pre-load baseline, then the fresh
        # model appended its own entry
        self.assertEqual(moe_layers, ["live.layers.0.mlp.experts", "model.layers.0.mlp.experts"])
        # fresh path ran process_weights exactly once, on the fresh model
        self.assertEqual(processed, [created[1]])
        # The old manifest is already schema v3, so wait for the atomic
        # replacement itself rather than the schema value. Otherwise tearDown
        # can remove the directory while the background writer is still active.
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if os.stat(manifest_path).st_ino != original_manifest_inode:
                break
            time.sleep(0.1)
        self.assertNotEqual(os.stat(manifest_path).st_ino, original_manifest_inode)
        with open(manifest_path) as manifest_file:
            self.assertEqual(json.load(manifest_file)["schema_version"], self.loader._SCHEMA_VERSION)
        # fallback path returns fresh weights (not the corrupt snapshot's)
        self.assertTrue(torch.equal(returned.w.data, created[1].w.data))

    @patch("torch.distributed.get_world_size", return_value=1)
    @patch("torch.distributed.get_rank", return_value=0)
    def test_load_model_restores_without_process_when_snapshot_valid(self, _r, _w):
        # fast_restore must be enabled: the test verifies the restore-only
        # path (no fresh process call), which requires fast_restore=true.
        self.loader = self._make_loader(fast_restore=True)
        model = _BareModel().to("npu").to("npu")

        created = []
        processed = []

        def fake_initialize_model(**kwargs):
            created.append(_BareModel().to("npu").to("npu"))
            return created[-1]

        def fake_process(model, model_config, target_device):
            processed.append(model)

        vllm_config = SimpleNamespace(
            device_config=SimpleNamespace(device="npu"),
            load_config=SimpleNamespace(device=None),
            parallel_config=SimpleNamespace(
                tensor_parallel_size=1,
                pipeline_parallel_size=1,
                enable_expert_parallel=False,
                data_parallel_size=1,
            ),
            additional_config={},
        )
        model_config = SimpleNamespace(model="test-model", dtype=torch.float32, quantization=None)
        fingerprint = self.loader._fingerprint(vllm_config, model_config)
        self._save(model, fingerprint)

        with (
            patch(
                "vllm_ascend.model_loader.bigtensorloader.bigtensorloader.initialize_model",
                side_effect=fake_initialize_model,
            ),
            patch(
                "vllm_ascend.model_loader.bigtensorloader.bigtensorloader.process_weights_after_loading",
                side_effect=fake_process,
            ),
            patch.object(BigTensorLoader, "load_weights", return_value=None),
        ):
            returned = self.loader.load_model(vllm_config, model_config)

        # restore-only: one init, no process, snapshot weights installed
        self.assertEqual(len(created), 1)
        self.assertEqual(processed, [])
        self.assertTrue(torch.equal(returned.w.data, model.w.data))

    def test_nz_c0_variant_preserves_cast_spec(self):
        """NZ_C0 variants need the original cast recipe on both restore paths."""

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_parameter(
                    "w",
                    torch.nn.Parameter(torch.empty(16, 32, dtype=torch.int8), requires_grad=False),
                )

        model = M().to("npu")
        source = torch.arange(16 * 32, dtype=torch.int32).reshape(16, 32).to(torch.int8).to("npu")
        # Keep a view between the recorded cast and the serialized parameter:
        # this is how post-processing stores transposed expert weights.
        model.w.data = fp4_packed_nz().cast(source).transpose(0, 1)
        original_format = int(torch_npu.get_npu_format(model.w.data))
        self.assertEqual(original_format, ACL_FORMAT_FRACTAL_NZ_C0_16)
        self.assertTrue(BigTensorLoader._is_packed_nz(fp4_packed_nz().cast(source).view(torch.int32)))

        self._save(model)
        entry = self._manifest()["tensors"]["w"]
        # The manifest's own format/dtype are enough to rebuild the layout: no
        # extra cast recipe has to be persisted, and older snapshots stay valid.
        self.assertEqual(entry["format"], ACL_FORMAT_FRACTAL_NZ_C0_16)
        self.assertNotIn("nz_cast_spec", entry)
        # w4a8_mxfp4 keeps this weight as a transposed view of the cast result;
        # the blob stores the contiguous base and the view is replayed on load.
        self.assertEqual(entry["transpose"], [0, 1])
        self.assertFalse(model.w.data.is_contiguous())
        self.assertEqual(
            BigTensorLoader._nz_cast_kwargs(original_format, torch.int8),
            fp4_packed_nz().cast_kwargs(),
        )

        for bulk_h2d in (False, True):
            with self.subTest(bulk_h2d=bulk_h2d):
                self.loader = self._make_loader(bulk_h2d=bulk_h2d, bulk_chunk_mb=1)
                restored = M().to("npu")
                self._restore(restored)
                self.assertEqual(int(torch_npu.get_npu_format(restored.w.data)), original_format)
                self.assertFalse(restored.w.data.is_contiguous())
                # Compare the ND base of both views: that is the layout the
                # operators consume, and it round-trips element for element.
                self.assertTrue(torch.equal(self._as_nd(model.w.data), self._as_nd(restored.w.data)))


class TestSnapshotIsolationCPU(unittest.TestCase):
    """CPU regression tests for P/D, speculative decoding, and save failures."""

    def setUp(self):
        self._tempdir = tempfile.TemporaryDirectory(prefix="bigtensor-isolation-")
        self.loader = BigTensorLoader(
            LoadConfig(
                load_format="bigtensorloader",
                model_loader_extra_config={"fast_restore": True, "checkpoint_path": self._tempdir.name},
            )
        )

    def tearDown(self):
        self._tempdir.cleanup()

    @staticmethod
    def _model_config(runner_type="generate", num_hidden_layers=61, model_type="deepseek_v3"):
        return SimpleNamespace(
            model="/models/shared-model",
            dtype=torch.float32,
            quantization=None,
            runner_type=runner_type,
            hf_text_config=SimpleNamespace(
                model_type=model_type,
                architectures=["TestForCausalLM"],
                num_hidden_layers=num_hidden_layers,
                num_nextn_predict_layers=1 if runner_type == "generate" else None,
            ),
        )

    @staticmethod
    def _vllm_config(kv_role=None):
        return SimpleNamespace(
            device_config=SimpleNamespace(device="cpu"),
            load_config=SimpleNamespace(device=None),
            parallel_config=SimpleNamespace(
                tensor_parallel_size=1,
                pipeline_parallel_size=1,
                enable_expert_parallel=False,
                data_parallel_size=1,
            ),
            additional_config={},
            kv_transfer_config=SimpleNamespace(kv_role=kv_role) if kv_role is not None else None,
        )

    def test_pd_roles_use_separate_reusable_snapshot_directories(self):
        model_config = self._model_config()
        with (
            patch("torch.distributed.get_world_size", return_value=1),
            patch("torch.distributed.get_rank", return_value=0),
        ):
            producer_fingerprint = self.loader._fingerprint(self._vllm_config("kv_producer"), model_config)
            consumer_fingerprint = self.loader._fingerprint(self._vllm_config("kv_consumer"), model_config)
            producer_paths = self.loader._snapshot_paths(producer_fingerprint)
            repeated_producer_paths = self.loader._snapshot_paths(producer_fingerprint)
            consumer_paths = self.loader._snapshot_paths(consumer_fingerprint)

        self.assertEqual(producer_paths, repeated_producer_paths)
        self.assertNotEqual(os.path.dirname(producer_paths[0]), os.path.dirname(consumer_paths[0]))
        self.assertNotEqual(os.path.dirname(producer_paths[0]), self._tempdir.name)
        self.assertEqual([os.path.basename(path) for path in producer_paths], ["0.json", "0.snapshot"])
        self.assertEqual([os.path.basename(path) for path in consumer_paths], ["0.json", "0.snapshot"])

    def test_mtp_target_and_drafter_use_separate_snapshot_directories(self):
        vllm_config = self._vllm_config("kv_producer")
        with (
            patch("torch.distributed.get_world_size", return_value=1),
            patch("torch.distributed.get_rank", return_value=0),
        ):
            target_fingerprint = self.loader._fingerprint(vllm_config, self._model_config())
            drafter_fingerprint = self.loader._fingerprint(
                vllm_config,
                self._model_config(runner_type="draft", num_hidden_layers=1, model_type="deepseek_v3_mtp"),
            )
            target_paths = self.loader._snapshot_paths(target_fingerprint)
            drafter_paths = self.loader._snapshot_paths(drafter_fingerprint)

        self.assertEqual(target_fingerprint["model"], drafter_fingerprint["model"])
        self.assertEqual(target_fingerprint["model_role"], "target")
        self.assertEqual(drafter_fingerprint["model_role"], "draft")
        self.assertNotEqual(target_fingerprint["num_hidden_layers"], drafter_fingerprint["num_hidden_layers"])
        self.assertNotEqual(os.path.dirname(target_paths[0]), os.path.dirname(drafter_paths[0]))

    def test_concurrent_writers_use_independent_temporary_files(self):
        fingerprint = {"model": "shared", "model_role": "target", "kv_role": "kv_producer"}
        manifest = {
            "world_rank": 0,
            "tensors": [],
            "sha256": "a" * 64,
            "fingerprint": fingerprint,
        }
        barrier = threading.Barrier(2)
        threads = [
            threading.Thread(target=self.loader._write_snapshot_async, args=([b"same bytes"], manifest))
            for _ in range(2)
        ]
        with (
            patch("torch.distributed.get_rank", return_value=0),
            patch("os.fsync", side_effect=lambda _fd: barrier.wait(timeout=5)),
            patch("vllm_ascend.model_loader.bigtensorloader.bigtensorloader.logger.exception") as log_exception,
        ):
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=10)
            manifest_path, blob_path = self.loader._snapshot_paths(fingerprint)
            snapshot_files = set(os.listdir(os.path.dirname(manifest_path)))

        self.assertFalse(any(thread.is_alive() for thread in threads))
        log_exception.assert_not_called()
        self.assertEqual(snapshot_files, {"0.json", "0.snapshot"})
        with open(blob_path, "rb") as blob_file:
            self.assertEqual(blob_file.read(), b"same bytes")

    def test_blob_writer_streams_with_bounded_buffer(self):
        """Chunks bigger than the queue must stream out, not pile up in host RAM."""
        fingerprint = {"model": "stream", "model_role": "target", "kv_role": "kv_producer"}
        manifest = {"world_rank": 0, "tensors": {}, "sha256": "", "fingerprint": fingerprint}
        chunks = [bytes([i]) * 512 for i in range(40)]

        with (
            patch.object(bigtensorloader_module, "_SNAPSHOT_BUFFER_BYTES", 1024),
            patch("torch.distributed.get_rank", return_value=0),
            patch("torch.distributed.get_world_size", return_value=1),
        ):
            writer = self.loader._make_snapshot_writer(fingerprint)
            for chunk in chunks:
                writer.append(chunk)
            writer.finish(manifest)
            writer.wait()
            manifest_path, blob_path = self.loader._snapshot_paths(fingerprint)

        with open(blob_path, "rb") as blob_file:
            self.assertEqual(blob_file.read(), b"".join(chunks))
        with open(manifest_path) as manifest_file:
            self.assertEqual(json.load(manifest_file)["world_rank"], 0)
        self.assertEqual(writer.hexdigest(), hashlib.sha256(b"".join(chunks)).hexdigest())

    def test_blob_writer_stops_the_producer_when_the_path_is_unwritable(self):
        """A read-only checkpoint path must fail before gigabytes of D2H copies."""
        fingerprint = {"model": "unwritable", "model_role": "target", "kv_role": "kv_producer"}
        with (
            patch("os.makedirs", side_effect=PermissionError("read-only file system")),
            patch("torch.distributed.get_rank", return_value=0),
            patch("torch.distributed.get_world_size", return_value=1),
            patch("vllm_ascend.model_loader.bigtensorloader.bigtensorloader.logger.exception"),
        ):
            writer = self.loader._make_snapshot_writer(fingerprint)
            raised = False
            deadline = time.time() + 5
            while time.time() < deadline:
                try:
                    writer.append(b"x" * 16)
                except RuntimeError:
                    raised = True
                    break
                time.sleep(0.01)
            writer.abort()
            writer.wait()

        self.assertTrue(raised, "append should surface the writer failure")
        self.assertFalse(writer._thread.is_alive())

    def test_restore_failure_preserves_snapshot_for_concurrent_reader(self):
        vllm_config = self._vllm_config("kv_consumer")
        model_config = self._model_config()
        fresh_model = _BareModel()
        with (
            patch("torch.distributed.get_world_size", return_value=1),
            patch("torch.distributed.get_rank", return_value=0),
        ):
            fingerprint = self.loader._fingerprint(vllm_config, model_config)
            manifest = {
                "world_rank": 0,
                "tensors": {},
                "sha256": "a" * 64,
                "fingerprint": fingerprint,
            }
            self.loader._write_snapshot_async([b"shared snapshot"], manifest)
            manifest_path, blob_path = self.loader._snapshot_paths(fingerprint)
            with open(manifest_path, "rb") as manifest_file:
                original_manifest = manifest_file.read()
            with open(blob_path, "rb") as blob_file:
                original_blob = blob_file.read()

            with (
                patch(
                    "vllm_ascend.model_loader.bigtensorloader.bigtensorloader.initialize_model",
                    return_value=fresh_model,
                ),
                patch("vllm_ascend.model_loader.bigtensorloader.bigtensorloader.process_weights_after_loading"),
                patch.object(BigTensorLoader, "load_weights", return_value=None),
                patch.object(
                    self.loader,
                    "restore_snapshot",
                    side_effect=RuntimeError("injected restore failure"),
                ) as restore_mock,
                patch("vllm_ascend.model_loader.bigtensorloader.bigtensorloader.threading.Thread") as thread_mock,
            ):
                returned = self.loader.load_model(vllm_config, model_config)

            with open(manifest_path, "rb") as manifest_file:
                self.assertEqual(manifest_file.read(), original_manifest)
            with open(blob_path, "rb") as blob_file:
                self.assertEqual(blob_file.read(), original_blob)

        self.assertIs(returned, fresh_model)
        restore_mock.assert_called_once()
        thread_mock.assert_called_once()

    def test_restore_failure_releases_abandoned_model_before_fresh_initialize(self):
        vllm_config = self._vllm_config("kv_consumer")
        model_config = self._model_config()
        fresh_model = _BareModel()
        abandoned_ref = None

        def fake_initialize_model(**kwargs):
            nonlocal abandoned_ref
            if abandoned_ref is None:
                abandoned_model = _BareModel()
                abandoned_ref = weakref.ref(abandoned_model)
                return abandoned_model
            # load_model must leave the exception handler and release its
            # traceback before constructing the replacement model.
            self.assertIsNone(abandoned_ref())
            return fresh_model

        with (
            patch("torch.distributed.get_world_size", return_value=1),
            patch("torch.distributed.get_rank", return_value=0),
        ):
            fingerprint = self.loader._fingerprint(vllm_config, model_config)
            # The real restore path raises on the old schema without any mock
            # retaining the abandoned model in its call history.
            self.loader._write_snapshot_async(
                [b"stale snapshot"],
                {
                    "world_rank": 0,
                    "tensors": {},
                    "sha256": "a" * 64,
                    "fingerprint": fingerprint,
                    "schema_version": 0,
                },
            )
            with (
                patch(
                    "vllm_ascend.model_loader.bigtensorloader.bigtensorloader.initialize_model",
                    side_effect=fake_initialize_model,
                ) as initialize_mock,
                patch("vllm_ascend.model_loader.bigtensorloader.bigtensorloader.process_weights_after_loading"),
                patch("vllm_ascend.model_loader.bigtensorloader.bigtensorloader.logger.exception"),
                patch.object(BigTensorLoader, "load_weights", return_value=None),
                patch("vllm_ascend.model_loader.bigtensorloader.bigtensorloader.threading.Thread"),
            ):
                returned = self.loader.load_model(vllm_config, model_config)

        self.assertIs(returned, fresh_model)
        self.assertEqual(initialize_mock.call_count, 2)
        self.assertIsNone(abandoned_ref())

    def test_synchronous_save_failure_returns_loaded_model(self):
        model = _BareModel()
        vllm_config = self._vllm_config()
        model_config = self._model_config()
        with (
            patch("torch.distributed.get_world_size", return_value=1),
            patch("torch.distributed.get_rank", return_value=0),
            patch(
                "vllm_ascend.model_loader.bigtensorloader.bigtensorloader.initialize_model",
                return_value=model,
            ),
            patch(
                "vllm_ascend.model_loader.bigtensorloader.bigtensorloader.process_weights_after_loading"
            ) as process_mock,
            patch.object(BigTensorLoader, "load_weights", return_value=None) as load_weights_mock,
            patch.object(self.loader, "save_snapshot", side_effect=RuntimeError("injected D2H failure")) as save_mock,
            patch("vllm_ascend.model_loader.bigtensorloader.bigtensorloader.threading.Thread"),
            patch.object(bigtensorloader_module._SnapshotBlobWriter, "abort") as abort_mock,
            patch.object(bigtensorloader_module._SnapshotBlobWriter, "finish") as finish_mock,
            patch("vllm_ascend.model_loader.bigtensorloader.bigtensorloader.logger.exception") as log_exception,
        ):
            returned = self.loader.load_model(vllm_config, model_config)

        self.assertIs(returned, model)
        self.assertFalse(returned.training)
        load_weights_mock.assert_called_once()
        process_mock.assert_called_once()
        save_mock.assert_called_once()
        # a synchronous serialization failure aborts the writer and never publishes
        abort_mock.assert_called_once()
        finish_mock.assert_not_called()
        log_exception.assert_called_once()

    def test_non_contiguous_view_saves_base_order(self):
        """Non-contiguous views are stored as their contiguous base.

        A strided D2H copy goes through the aicpu Transpose kernel, which faults
        for some dtypes (float8_e8m0fnu). Saving the metadata-only permutation
        of the base and re-applying it on restore avoids that kernel entirely.
        """
        base = torch.randn(4, 8)
        tensor = base.transpose(0, 1)  # non-contiguous view
        self.assertFalse(tensor.is_contiguous())

        blob_parts: list = []
        entry, offset = self.loader._append_tensor(tensor, blob_parts, 0)

        expected = base.contiguous().numpy().tobytes()
        self.assertEqual(entry["shape"], [8, 4])
        self.assertEqual(entry["transpose"], [0, 1])
        self.assertEqual(entry["nbytes"], len(expected))
        self.assertEqual(offset, len(expected))
        self.assertEqual(b"".join(blob_parts), expected)

    def test_higher_rank_view_recovers_contiguous_base(self):
        """3-D expert weights are kept as a permutation of a contiguous base.

        ``(E, H, 2I)`` transposed to ``(E, 2I, H)`` has no single swap that
        makes it contiguous, so the save must find the full permutation and
        record the inverse to re-apply after restore.
        """
        base = torch.randn(2, 3, 4)
        view = base.permute(0, 2, 1)
        self.assertFalse(view.is_contiguous())

        perm = BigTensorLoader._base_permutation(view)
        self.assertTrue(view.permute(perm).is_contiguous())
        self.assertEqual(BigTensorLoader._invert_permutation(perm), (0, 2, 1))

        # Round trip: permuted base -> inverse permutation -> original view
        restored = view.permute(perm).clone().permute(BigTensorLoader._invert_permutation(perm))
        self.assertEqual(tuple(restored.shape), tuple(view.shape))
        self.assertEqual(restored.numpy().tobytes(), view.contiguous().numpy().tobytes())

    def test_mx_scale_layout_is_saved_as_its_base(self):
        """4-D MX block scales are a permuted view; the save must record it.

        Real shape/stride from a DeepSeek-V4.1 w4a8 checkpoint:
        (48, 80, 4608, 2) / (737280, 2, 160, 1), i.e. a permute(0, 2, 1, 3) of
        the contiguous (48, 4608, 80, 2) base.
        """
        base = torch.randn(2, 4, 3, 2)
        view = base.permute(0, 2, 1, 3)
        self.assertEqual(tuple(view.shape), (2, 3, 4, 2))
        self.assertFalse(view.is_contiguous())

        blob_parts: list = []
        entry, _ = self.loader._append_tensor(view, blob_parts, 0)

        restore_perm = entry["permute"]
        self.assertEqual(restore_perm, [0, 2, 1, 3])
        self.assertEqual(entry["save_shape"], [2, 4, 3, 2])
        self.assertEqual(entry["shape"], [2, 3, 4, 2])
        self.assertEqual(b"".join(blob_parts), base.contiguous().numpy().tobytes())

        # Restore: base-order bytes -> save_shape -> re-apply the view
        rebuilt = torch.frombuffer(bytearray(b"".join(blob_parts)), dtype=base.dtype)
        rebuilt = rebuilt.view(base.shape).permute(*restore_perm)
        self.assertEqual(tuple(rebuilt.shape), tuple(view.shape))
        self.assertEqual(tuple(rebuilt.stride()), tuple(view.stride()))
        self.assertEqual(rebuilt.contiguous().numpy().tobytes(), view.contiguous().numpy().tobytes())

    def test_invert_permutation_of_rotation(self):
        self.assertEqual(BigTensorLoader._invert_permutation((1, 2, 0)), (2, 0, 1))
        self.assertEqual(BigTensorLoader._invert_permutation((0, 1)), (0, 1))

    def test_layout_without_contiguous_base_returns_none(self):
        """A slice is not a permutation of a base; it keeps the strided copy path."""
        tensor = torch.randn(4, 8)[:, ::2]
        self.assertIsNone(BigTensorLoader._base_permutation(tensor))


if __name__ == "__main__":
    unittest.main()
