"""Check the read-only display for dataset campaign progress."""

import unittest

from tools.datasets.watch_generation_progress import format_progress


class GenerationProgressTest(unittest.TestCase):
    def test_sample_progress_names_the_json_writing_phase(self) -> None:
        job = {
            "kind": "samples",
            "state": "running",
            "rows": 3_000_000,
            "dataset": "datasets/model/x/samples/samples_01.json",
        }
        waiting = format_progress(job, {}, 100.0)
        self.assertIn("en attente de la phase d'écriture", waiting)
        self.assertNotIn("Prix calculés", waiting)
        writing = format_progress(
            job,
            {
                "state": "running",
                "completed_samples": 1_500_000,
                "total_samples": 3_000_000,
                "samples_per_second": 100_000.0,
                "elapsed_seconds": 15.0,
                "estimated_seconds_remaining": 15.0,
            },
            100.0,
        )
        self.assertIn("Échantillons écrits : 1 500 000 / 3 000 000", writing)
        self.assertIn("50,00 % de l'écriture", writing)
        self.assertIn("Temps restant estimé (écriture JSON)", writing)


if __name__ == "__main__":
    unittest.main()
