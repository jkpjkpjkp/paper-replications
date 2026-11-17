"""Guard against false replication claims from partial runs or metric scaling."""
import unittest

import numpy as np
import pandas as pd

from compare_published import SHARPE_FACTOR, additional_tables, compare, comparison_summary


class PublishedComparisonTests(unittest.TestCase):
    def setUp(self):
        self.seeds = [16, 37, 47, 214, 1947]
        self.reference = pd.DataFrame([{"market": "euro50", "rho_percent": 10,
            "method": "dfl", "optimizer": "oscar", "paper_daily_gross_sharpe_mean": .955,
            "paper_seed_sd": .014}])
        self.panels = {"euro50": {"n_assets": 47}}

    def runs(self, seeds, sr):
        return pd.DataFrame([{"dataset": "euro50", "method": "dfl", "k_percent": 10,
            "selector": "oscar", "seed": seed, "alpha": .5, "verified_daily_sr": sr,
            "observed_actual_holdings_max": 47, "observed_gross_exposure_max": 3.}
            for seed in seeds])

    def result(self, runs):
        return compare(self.reference, runs, self.panels, self.seeds, "oscar").iloc[0]

    def test_one_seed_is_incomplete_even_when_mean_matches(self):
        result = self.result(self.runs(self.seeds[:1], .955))
        self.assertFalse(result.all_required_repeats_complete)
        self.assertFalse(result.exact_published_result_replicated)
        self.assertEqual(result.status, "incomplete_seed_repeats")

    def test_annualized_agreement_cannot_be_called_daily_agreement(self):
        result = self.result(self.runs(self.seeds, .955/np.sqrt(252)))
        self.assertAlmostEqual(result.annualized_sr_minus_paper_display, 0.)
        self.assertFalse(result.rounded_display_numbers_agree)
        self.assertFalse(result.exact_published_result_replicated)
        self.assertTrue(result.annualized_mean_match)
        self.assertFalse(result.annualized_sd_match)
        self.assertFalse(result.annualized_mean_and_sd_match)

    def test_annualization_scales_seed_sd_and_preserves_paper_values(self):
        offsets = np.arange(-2., 3.)
        values = (.955 + offsets / offsets.std(ddof=1) * .014) / SHARPE_FACTOR
        runs = self.runs(self.seeds, 0.)
        runs["verified_daily_sr"] = values
        result = self.result(runs)
        self.assertAlmostEqual(result.released_annualized_gross_sr_mean, .955)
        self.assertAlmostEqual(result.released_annualized_seed_sd, .014)
        self.assertAlmostEqual(result.released_daily_seed_sd, .014 / SHARPE_FACTOR)
        self.assertEqual(result.paper_reported_sharpe_mean, .955)
        self.assertEqual(result.paper_seed_sd, .014)
        self.assertTrue(result.annualized_mean_and_sd_match)
        self.assertFalse(result.daily_mean_and_sd_match)
        self.assertFalse(result.exact_published_result_replicated)

    def test_incomplete_seed_set_cannot_match_annualized_statistics(self):
        runs = self.runs(self.seeds[:2], 0.)
        runs["verified_daily_sr"] = np.array([.955-.014/np.sqrt(2), .955+.014/np.sqrt(2)]) / SHARPE_FACTOR
        result = self.result(runs)
        self.assertAlmostEqual(result.released_annualized_seed_sd, .014)
        self.assertFalse(result.annualized_mean_and_sd_match)

    def test_error_summary_excludes_incomplete_cells(self):
        frame = compare(self.reference, self.runs(self.seeds[:1], .955/SHARPE_FACTOR), self.panels, self.seeds, "oscar")
        summary = comparison_summary(frame).iloc[0]
        self.assertEqual(summary.completed_cells, 0)
        self.assertTrue(np.isnan(summary.annualized_sharpe_mae))

    def test_alpha_sharpe_is_scaled_but_drawdown_is_not(self):
        runs = self.runs(self.seeds, .2)
        runs["verified_daily_sr"] = [.1, .15, .2, .25, .3]
        runs["verified_test_mdd"] = [.1, .2, .3, .4, .5]
        tables = additional_tables(runs, self.seeds, "oscar")
        sr = tables[2].query("market == 'euro50' and alpha == .5 and rho_percent == 10").iloc[0]
        self.assertAlmostEqual(sr.released_annualized_gross_sr_mean, .2*SHARPE_FACTOR)
        self.assertAlmostEqual(sr.released_annualized_seed_sd, np.std([.1,.15,.2,.25,.3],ddof=1)*SHARPE_FACTOR)
        mdd = tables[3].query("market == 'euro50' and method == 'dfl' and rho_percent == 10").iloc[0]
        self.assertAlmostEqual(mdd.released_mean, .3)
        self.assertAlmostEqual(mdd.released_seed_sd, np.std([.1,.2,.3,.4,.5],ddof=1))
        self.assertEqual(mdd.metric, "maximum_drawdown_unscaled")

    def test_unplanned_seed_cannot_replace_missing_seed(self):
        result = self.result(self.runs(self.seeds[:4]+[999], .955))
        self.assertEqual(result.completed_seed_count, 4)
        self.assertFalse(result.all_required_repeats_complete)

    def test_duplicate_seed_cannot_count_as_another_repeat(self):
        with self.assertRaisesRegex(ValueError, "Duplicate runs"):
            self.result(self.runs(self.seeds[:4]+[16], .955))

    def test_dense_source_holdings_are_reported_without_hard_support_claim(self):
        result = self.result(self.runs(self.seeds, .955))
        self.assertEqual(result.current_k, 5)
        self.assertEqual(result.observed_actual_holdings_max, 47)
        self.assertFalse(result.exact_published_result_replicated)

    def test_absent_market_stays_in_comparison(self):
        self.panels = {"euro50": None}
        result = self.result(pd.DataFrame())
        self.assertEqual(result.status, "data_unavailable")
        self.assertEqual(result.required_seed_count, 5)

    def test_alpha_sensitivity_cannot_pollute_main_table(self):
        main = self.runs(self.seeds, .955)
        extra = self.runs(self.seeds, 1.2)
        extra["alpha"] = .9
        result = self.result(pd.concat([main, extra], ignore_index=True))
        self.assertEqual(result.completed_seed_count, 5)
        self.assertAlmostEqual(result.released_daily_gross_sr_mean, .955)

    def test_hard_baseline_mapping_does_not_change_dfl(self):
        reference = pd.concat([self.reference, self.reference.assign(method="pfl")], ignore_index=True)
        dfl = self.runs(self.seeds, .12)
        soft = dfl.assign(method="pfl", alpha=0., verified_daily_sr=.21)
        hard = soft.assign(selector="oscar_hard", verified_daily_sr=.34)
        result = compare(reference, pd.concat([dfl, soft, hard]), self.panels, self.seeds, "oscar_hard")
        self.assertEqual(result.iloc[0].release_selector, "oscar")
        self.assertAlmostEqual(result.iloc[0].released_daily_gross_sr_mean, .12)
        self.assertEqual(result.iloc[1].release_selector, "oscar_hard")
        self.assertAlmostEqual(result.iloc[1].released_daily_gross_sr_mean, .34)

    def test_undefined_drawdown_cannot_be_silently_omitted(self):
        runs = self.runs(self.seeds, .955)
        runs["verified_test_mdd"] = np.nan
        appendix = additional_tables(runs, self.seeds, "oscar")[3]
        row = appendix[(appendix.market == "euro50") & (appendix.method == "dfl") & (appendix.rho_percent == 10)].iloc[0]
        self.assertEqual(row.completed_seed_count, 5)
        self.assertEqual(row.defined_metric_count, 0)
        self.assertFalse(row.all_required_repeats_complete)
        self.assertEqual(row.status, "undefined_metric_in_some_runs")


if __name__ == "__main__":
    unittest.main()
