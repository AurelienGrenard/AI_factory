"""Check artifact publication, interruption recovery and preservation of edits."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.datasets import artifact_publication as publication


class PublicationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.work = self.root / "run/work"
        self.journal = self.root / "run/publication.json"
        self.items = []
        for relative in ("datasets/example.json", "catalog/example/generation.yaml"):
            source = self.work / relative
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_text("new " + relative)
            self.items.append({"path": relative, "sha256": publication.digest(source),
                               "previous_sha256": None})

    def publish(self):
        publication.publish_pair(self.root, self.work, self.journal, self.items)

    def test_publish_and_idempotent_resume(self):
        self.publish()
        self.publish()
        self.assertEqual(json.loads(self.journal.read_text())["state"], "complete")
        for item in self.items:
            self.assertEqual(publication.digest(self.root / item["path"]), item["sha256"])

    def test_resume_interrupted_pair(self):
        original_replace = publication.os.replace

        def interrupt(source, destination):
            if Path(destination) == self.root / self.items[1]["path"]:
                raise InterruptedError("simulated interruption between the two renames")
            original_replace(source, destination)

        with patch.object(publication.os, "replace", side_effect=interrupt):
            with self.assertRaises(InterruptedError):
                self.publish()
        self.assertEqual(json.loads(self.journal.read_text())["state"], "publishing")
        self.publish()
        self.assertEqual(json.loads(self.journal.read_text())["state"], "complete")

    def test_external_edit_blocks_before_either_rename(self):
        path = self.root / self.items[1]["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("user edit")
        with self.assertRaisesRegex(ValueError, "immutable"):
            self.publish()
        self.assertFalse((self.root / self.items[0]["path"]).exists())

    def test_changed_staged_file_blocks_pair(self):
        (self.work / self.items[1]["path"]).write_text("partial output")
        with self.assertRaisesRegex(ValueError, "missing or changed"):
            self.publish()
        self.assertFalse(self.journal.exists())

    def test_external_edit_between_renames_is_preserved(self):
        original_replace = publication.os.replace
        changed = self.root / self.items[1]["path"]

        def edit_after_first_rename(source, destination):
            original_replace(source, destination)
            if Path(destination) == self.root / self.items[0]["path"]:
                changed.parent.mkdir(parents=True, exist_ok=True)
                changed.write_text("new user edit")

        with patch.object(publication.os, "replace", side_effect=edit_after_first_rename):
            with self.assertRaisesRegex(ValueError, "during publication"):
                self.publish()
        self.assertEqual(changed.read_text(), "new user edit")
        self.assertEqual(json.loads(self.journal.read_text())["state"], "publishing")

    def test_missing_output_requires_a_real_staged_checksum(self):
        self.items[0]["sha256"] = None
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            self.publish()
        self.assertFalse(self.journal.exists())

    def test_published_artifacts_are_immutable(self):
        self.publish()
        self.publish()
        self.journal.unlink()
        source = self.work / self.items[0]["path"]
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text("different bytes")
        self.items[0]["sha256"] = publication.digest(source)
        with self.assertRaisesRegex(ValueError, "immutable"):
            self.publish()

    def test_paths_cannot_escape_or_follow_symlinks(self):
        for relative in ("../outside", "/tmp/outside", ""):
            with self.assertRaises(ValueError):
                publication.contained_path(self.root, relative)
        (self.root / "redirect").symlink_to(self.work, target_is_directory=True)
        with self.assertRaises(ValueError):
            publication.contained_path(self.root, "redirect/output.json")


if __name__ == "__main__":
    unittest.main()
