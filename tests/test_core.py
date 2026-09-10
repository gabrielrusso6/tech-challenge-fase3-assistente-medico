"""Testes executáveis com a biblioteca padrão, inclusive sem LangGraph instalado."""
import json
import tempfile
import unittest
from pathlib import Path

from tc3.dataset import analyze, prepare, redact
from tc3.storage import Repository

ROOT = Path(__file__).resolve().parents[1]


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.repo = Repository(Path(self.tmp.name) / "test.sqlite")
        self.repo.seed(ROOT / "data/patients.json")

    def test_patient_isolation(self):
        self.assertEqual(self.repo.patient("SYN-001")["id"], "SYN-001")
        self.assertEqual(self.repo.patient("SYN-002")["id"], "SYN-002")

    def test_sql_injection_does_not_return_patient(self):
        self.assertIsNone(self.repo.patient("' OR 1=1 --"))

    def test_missing_patient(self):
        self.assertIsNone(self.repo.patient("missing"))

    def test_seed_is_idempotent(self):
        self.repo.seed(ROOT / "data/patients.json")
        with self.repo.connect() as db:
            self.assertEqual(db.execute("SELECT count(*) FROM patients").fetchone()[0], 3)

    def test_audit_is_scoped(self):
        self.repo.log("a", "loaded", {"count": 1})
        self.repo.log("b", "loaded", {"count": 2})
        self.assertEqual(self.repo.events("a"), [{"event": "loaded", "details": {"count": 1}}])

    def test_decisions_persist(self):
        self.repo.decision("run", "SYN-001", "reviewer-demo", False)
        with Repository(self.repo.path).connect() as db:
            self.assertEqual(db.execute("SELECT approved FROM decisions").fetchone()[0], 0)

    def test_real_data_rejected(self):
        p = Path(self.tmp.name) / "bad.json"
        p.write_text(json.dumps([{"id": "X", "synthetic": False}]))
        with self.assertRaises(ValueError):
            self.repo.seed(p)


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.rows = json.loads((ROOT / "data/seed_qa.json").read_text(encoding="utf-8"))

    def test_seed_valid_and_all_splits_present(self):
        rows = prepare(self.rows)
        self.assertEqual(set(analyze(rows)["splits"]), {"train", "validation", "test"})

    def test_duplicate_question_rejected(self):
        with self.assertRaises(ValueError):
            prepare(self.rows + [{**self.rows[0], "question": self.rows[0]["question"].upper()}])

    def test_group_leakage_rejected(self):
        row = {**self.rows[0], "question": "Outra pergunta", "split": "test"}
        with self.assertRaises(ValueError):
            prepare(self.rows + [row])

    def test_redaction(self):
        self.assertEqual(redact("Contato fake@example.com CPF 123.456.789-00"),
                         "Contato [EMAIL] CPF [CPF]")

    def test_empty_answer_rejected(self):
        self.rows[0]["answer"] = " "
        with self.assertRaises(ValueError):
            prepare(self.rows)

    def test_non_synthetic_rejected(self):
        self.rows[0]["synthetic"] = False
        with self.assertRaises(ValueError):
            prepare(self.rows)

    def test_missing_source_rejected(self):
        self.rows[0]["source_ids"] = []
        with self.assertRaises(ValueError):
            prepare(self.rows)

    def test_empty_dataset_analysis_rejected(self):
        with self.assertRaises(ValueError):
            analyze([])


if __name__ == "__main__":
    unittest.main()
