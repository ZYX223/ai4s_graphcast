import json
import tempfile
import unittest
from pathlib import Path

from graphcast_bench.validation.audit import run_with_audit


class AuditFinallyTest(unittest.TestCase):
    def test_primary_error_survives_secondary_audit_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audit.json"

            def train():
                raise ValueError("primary training failure")

            def audit(_):
                raise RuntimeError("secondary audit failure")

            with self.assertRaisesRegex(ValueError, "primary training failure"):
                run_with_audit(train, audit, path)
            self.assertEqual(
                json.loads(path.read_text())["status"], "incomplete_after_training_error"
            )

    def test_audit_error_after_success_is_not_suppressed(self):
        def audit(_):
            raise RuntimeError("missing required dispatch")

        with self.assertRaisesRegex(RuntimeError, "missing required dispatch"):
            run_with_audit(lambda: None, audit, Path("unused"))

    def test_success_returns_value_and_saves_audit(self):
        seen = []
        self.assertEqual(run_with_audit(lambda: 7, seen.append, Path("audit")), 7)
        self.assertEqual(seen, [Path("audit")])

    def test_optional_memory_diagnostic_cannot_mask_primary_or_skip_audit(self):
        seen = []

        def train():
            raise ValueError("allocation failed")

        def diagnostic(error):
            self.assertIsInstance(error, ValueError)
            raise RuntimeError("memory stats unavailable")

        with self.assertRaisesRegex(ValueError, "allocation failed"):
            run_with_audit(train, seen.append, Path("audit"), on_failure=diagnostic)
        self.assertEqual(seen, [Path("audit")])
