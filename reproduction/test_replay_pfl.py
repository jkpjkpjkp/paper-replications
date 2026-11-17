"""Prevent reuse of incompatible or corrupted research checkpoints."""
import tempfile
import unittest
from pathlib import Path

import torch

from replay_pfl import digest, model_error, validate_receipt


class PFLReuseTests(unittest.TestCase):
    def test_different_tensor_values_are_detected(self):
        self.assertEqual(model_error({"w": torch.tensor([1., 2.])},
                                     {"w": torch.tensor([1., 3.])}), 1.)

    def test_nonfinite_parameters_cannot_pass_equivalence(self):
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            model_error({"w": torch.tensor([float("nan")])}, {"w": torch.tensor([0.])})

    def test_architecture_mismatch_cannot_broadcast(self):
        with self.assertRaisesRegex(ValueError, "shape"):
            model_error({"w": torch.zeros(2, 1)}, {"w": torch.zeros(2)})

    def test_cache_requires_model_data_protocol_and_source_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            model = Path(folder) / "model.pt"
            model.write_bytes(b"original checkpoint")
            receipt = {"model_sha256": digest(model), "data_sha256": "data-v1",
                       "protocol_sha256": "config-v1", "source_sha256": {"train.py": "source-v1"}}
            validate_receipt(receipt, model, "data-v1", "config-v1", {"train.py": "source-v1"})
            for data, config, source in [("data-v2", "config-v1", "source-v1"),
                                         ("data-v1", "config-v2", "source-v1"),
                                         ("data-v1", "config-v1", "source-v2")]:
                with self.subTest(data=data, config=config, source=source):
                    with self.assertRaisesRegex(ValueError, "provenance mismatch"):
                        validate_receipt(receipt, model, data, config, {"train.py": source})
            model.write_bytes(b"different checkpoint")
            with self.assertRaisesRegex(ValueError, "provenance mismatch"):
                validate_receipt(receipt, model, "data-v1", "config-v1", {"train.py": "source-v1"})


if __name__ == "__main__":
    unittest.main()
