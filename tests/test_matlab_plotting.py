"""Regression tests for the paper-aligned four-figure suite."""

from pathlib import Path
import unittest


PROJECT = Path(__file__).resolve().parents[1]


class MatlabPlottingTests(unittest.TestCase):
    def test_four_figure_suite_uses_only_paper_aligned_results(self) -> None:
        matlab = (
            PROJECT
            / "matlab_paper_aligned/run_four_figures_paper_aligned.m"
        ).read_text(encoding="utf-8")
        python = (
            PROJECT
            / "experiments/plot_four_publication_figures_paper_aligned.py"
        ).read_text(encoding="utf-8")
        for source in (matlab, python):
            self.assertIn("deadline_curves_paper_aligned_50seeds.csv", source)
            self.assertIn("deployment_scaling_paper_aligned.csv", source)
            self.assertIn("bos_scaling_paper_aligned.csv", source)
            self.assertNotIn("v6_1", source)
        for name in (
            "Figure3_deadline_success.png",
            "Figure4_macro_cost_delay.png",
            "Figure5_transfer_cpu_ram.png",
            "Figure6_runtime_bos_acceptance.png",
        ):
            self.assertIn(name, matlab)
            self.assertIn(name.removesuffix(".png"), python)
        self.assertNotIn("exportgraphics", matlab)
        self.assertNotIn(".pdf", matlab.lower())
        self.assertNotIn("bar(", matlab.lower())
        self.assertIn("global_deadline_success", matlab)
        self.assertIn("first_round_accepted_sfc_ratio", matlab)

    def test_python_publication_script_uses_no_bars_or_splines(self) -> None:
        source = (
            PROJECT
            / "experiments/plot_four_publication_figures_paper_aligned.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn(".bar(", source)
        self.assertNotIn("make_interp_spline", source)
        self.assertNotIn("scipy.interpolate", source)
        self.assertIn("plot_curve", source)
        self.assertIn("common_global_success", source)
        self.assertIn("CPU and RAM utilization", source)
        self.assertIn("first_round_accepted_sfc_ratio", source)

    def test_figures_four_and_five_start_exactly_at_two(self) -> None:
        matlab = (
            PROJECT
            / "matlab_paper_aligned/run_four_figures_paper_aligned.m"
        ).read_text(encoding="utf-8")
        python = (
            PROJECT
            / "experiments/plot_four_publication_figures_paper_aligned.py"
        ).read_text(encoding="utf-8")
        self.assertEqual(matlab.count("formatMacroAxis(ax,firstValid,lastValid,2.0)"), 3)
        self.assertEqual(python.count("axis_lower = 2.0"), 2)
        self.assertNotIn("formatMacroAxis(ax,firstValid,lastValid,1.5)", matlab)
        self.assertNotIn("axis_lower = 1.5", python)

    def test_figure_three_is_complete_and_legend_is_inside(self) -> None:
        matlab = (
            PROJECT
            / "matlab_paper_aligned/run_four_figures_paper_aligned.m"
        ).read_text(encoding="utf-8")
        python = (
            PROJECT
            / "experiments/plot_four_publication_figures_paper_aligned.py"
        ).read_text(encoding="utf-8")
        self.assertIn("T.deadline_factor>=0.8 & T.deadline_factor<=3.0", matlab)
        self.assertIn("xlim(ax,[0.8 3.0])", matlab)
        self.assertIn("data.deadline_factor.between(0.8, 3.0)", python)
        self.assertIn("ax.set_xlim(0.8, 3.0)", python)
        self.assertIn("'Location','southeast','NumColumns',2", matlab)
        self.assertNotIn("'Location','southoutside'", matlab)
        self.assertIn("axes[0].legend(", python)
        self.assertIn('loc="lower right"', python)
        self.assertNotIn('fig.tight_layout(rect=(0, 0.16, 1, 1))', python)

    def test_figure_six_legend_uses_blank_lower_area(self) -> None:
        matlab = (
            PROJECT
            / "matlab_paper_aligned/run_four_figures_paper_aligned.m"
        ).read_text(encoding="utf-8")
        python = (
            PROJECT
            / "experiments/plot_four_publication_figures_paper_aligned.py"
        ).read_text(encoding="utf-8")
        self.assertIn("'Location','southwest','NumColumns',2", matlab)
        self.assertIn("axes[1].legend(", python)
        self.assertIn('loc="lower left"', python)
        self.assertNotIn('loc="upper left"', python)


if __name__ == "__main__":
    unittest.main()
