import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from segmentation import MODEL, RESULTS, ROOT, assign_segments, read_customers, write_results


class SegmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = json.loads(MODEL.read_text(encoding="utf-8"))
        cls.batches = [read_customers(path) for path in
                       sorted((ROOT / "simulator_batches/incoming").glob("batch_*.csv"))[:2]]

    def test_two_batches_keep_their_reference_labels(self):
        self.assertEqual(len(self.batches), 2)
        columns = ["CustomerID", "segment_id", "customer_segment", "rq2_match"]
        reference = pd.read_csv(RESULTS / "assignments.csv", dtype={"CustomerID": "string"}).set_index("CustomerID")
        for batch in self.batches:
            scored = assign_segments(batch, self.model)[columns].set_index("CustomerID")
            pd.testing.assert_frame_equal(scored, reference.loc[scored.index], check_dtype=False)

    def test_unknown_churn_is_excluded_and_does_not_change_assignment(self):
        batch = self.batches[0].copy()
        original = assign_segments(batch, self.model)
        batch["Churned"] = float("nan")
        batch.loc[batch.index[0], "Churned"] = 1
        with tempfile.TemporaryDirectory() as folder:
            write_results(batch, self.model, Path(folder))
            profiles = pd.read_csv(Path(folder) / "profiles.csv")
            group = profiles.loc[profiles.segment_id.eq(original.segment_id.iloc[0])].iloc[0]
            self.assertEqual(profiles.known_churn.sum(), 1)
            self.assertEqual(group.churn_share, 1)
            self.assertTrue(profiles.loc[profiles.known_churn.eq(0), "churn_share"].isna().all())
            labels = pd.read_csv(Path(folder) / "assignments.csv")
            self.assertEqual(labels.segment_id.tolist(), original.segment_id.tolist())

    def test_rq2_cutoffs_include_ties_and_require_frequency(self):
        limits = self.model["thresholds"]
        freq, gap, eng = limits["frequent"], limits["long_gap"], limits["low_engagement"]
        frame = pd.DataFrame([[freq, gap, eng + .1], [freq, gap - 1, eng],
                              [freq - 1, gap, eng], [freq, gap - 1, eng + .1]],
                             columns=self.model["features"])
        self.assertEqual(assign_segments(frame, self.model).rq2_match.tolist(),
                         [True, True, False, False])


if __name__ == "__main__":
    unittest.main()
