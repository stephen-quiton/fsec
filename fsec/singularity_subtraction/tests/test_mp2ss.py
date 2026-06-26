import unittest
from contextlib import redirect_stdout
from io import StringIO
import numpy as np
from pyscf.pbc import df
from pyscf.pbc import gto as pbcgto
from pyscf.pbc import mp
from pyscf.pbc import scf as pbcscf

from fsec.singularity_subtraction import MP2SS
from fsec.singularity_subtraction.mp2ss import (
    MP2DirectFourthOrderSS,
    MP2DirectFullSS,
    MP2DirectSecondOrderSS,
    MP2ExchangeSS,
)


class PrintResultsTests(unittest.TestCase):
    @staticmethod
    def _calculator(calculator_class, integral, quadrature, correction):
        calculator = calculator_class.__new__(calculator_class)
        calculator.integral_term = integral
        calculator.quadrature_term = quadrature
        calculator.correction = correction
        return calculator

    def _mp2ss(self, separate):
        mp2ss = MP2SS.__new__(MP2SS)
        mp2ss.with_df_ints = True
        mp2ss.auxfunc_direct = "direct-model"
        mp2ss.auxfunc_exchange = "exchange-model"
        mp2ss.edi_uncorr = -1.0
        mp2ss.exi_uncorr = 0.25
        mp2ss.emp2_uncorr = -0.75
        mp2ss.emp2ss_direct = -1.1
        mp2ss.emp2ss_exchange = 0.3
        mp2ss.emp2ss = -0.8
        mp2ss.correct_q2_q4_separately = separate
        mp2ss.exchange_correction = self._calculator(MP2ExchangeSS, 4.0, 1.5, 2.5)
        return mp2ss

    def test_print_results_with_full_direct_term(self):
        mp2ss = self._mp2ss(separate=False)
        mp2ss.direct_correction = self._calculator(MP2DirectFullSS, 3.0, 2.0, 1.0)

        output = StringIO()
        with redirect_stdout(output):
            mp2ss.print_results()
        result = output.getvalue()

        self.assertIn("Uncorrected MP2 Energies:", result)
        self.assertIn("Direct Term:\n", result)
        self.assertNotIn("Direct Term (2nd order):", result)
        self.assertIn("Exchange Term:\n", result)
        self.assertIn("Final Energies:", result)
        self.assertIn(" Integral term (hartree)    = 3.0", result)
        self.assertIn(" Total correction (hartree) = 1.0", result)

    def test_print_results_with_separate_direct_terms(self):
        mp2ss = self._mp2ss(separate=True)
        mp2ss.direct_second_order_correction = self._calculator(
            MP2DirectSecondOrderSS, 2.0, 1.5, 0.5
        )
        mp2ss.direct_fourth_order_correction = self._calculator(
            MP2DirectFourthOrderSS, 1.0, 0.5, 0.5
        )
        mp2ss.direct_correction = self._calculator(MP2DirectFullSS, 3.0, 2.0, 1.0)
        mp2ss.direct_correction.results_title = "Direct Term (Full: 2nd + 4th)"

        output = StringIO()
        with redirect_stdout(output):
            mp2ss.print_results()
        result = output.getvalue()

        self.assertIn("Direct Term (2nd order):", result)
        self.assertIn("Direct Term (4th order):", result)
        self.assertIn("Direct Term (Full: 2nd + 4th):", result)
        self.assertNotIn("Direct Term:\n", result)
        self.assertIn("Exchange Term:\n", result)
        self.assertIn("Final Energies:", result)


class KnownValues(unittest.TestCase):
    TOL = 1e-6
    REFERENCES = {
        "emp2ss": -0.012377126997773956,
        "mp2ss_total_correction": 0.004131436113473142,
        "mp2ss_direct_correction": -0.004741925103280841,
        "mp2ss_exchange_correction": 0.008873361216753983,
        "direct_integral_term": -0.0291517748423867,
        "direct_quadrature_term": -0.02440984973910586,
        "exchange_integral_term": 0.018810916823196567,
        "exchange_quadrature_term": 0.009937555606442584,
        "direct_integral_term_q2": -0.02237394483611901,
        "direct_quadrature_term_q2": -0.02000979264030079,
        "direct_total_correction_q2": -0.0023641521958182206,
        "direct_integral_term_dG0": -0.00677783000626769,
        "direct_quadrature_term_dG0": -0.00440005709880507,
        "direct_total_correction_dG0": -0.00237777290746262,
        "direct_total_correction_q2_dG0": -0.004741925103280841,
        "emp2_uncorr": -0.016508563111247095,
        "edi_uncorr": -0.0311807129223964,
        "exi_uncorr": 0.014672149811149306,
        "emp2ss_direct": -0.035922638025677245,
        "emp2ss_exchange": 0.02354551102790329,
    }

    @classmethod
    def setUpClass(cls):
        cls.kmf, cls.kmp, cls.t2 = cls._run_mp2()

    @staticmethod
    def _build_h2_cell():
        cell = pbcgto.Cell()
        cell.unit = "Bohr"
        cell.atom = """
            H 0.00 0.00 0.00
            H 0.00 0.00 1.80
        """
        cell.a = np.eye(3) * 6.0
        cell.verbose = 0
        cell.spin = 0
        cell.charge = 0
        cell.basis = "gth-szv"
        cell.pseudo = "gth-hf"
        cell.ke_cutoff = 100
        cell.max_memory = 1000
        cell.precision = 1e-8
        cell.build()
        kpts = cell.make_kpts((1, 1, 2), wrap_around=True, with_gamma_point=True)
        return cell, kpts

    @classmethod
    def _run_mp2(cls):
        cell, kpts = cls._build_h2_cell()
        kmf = pbcscf.KRHF(cell, kpts)
        kmf.exxdiv = "ewald"
        kmf.with_df = df.GDF(cell, kpts).build()
        kmf.kernel()
        if not kmf.converged:
            raise RuntimeError("Reference KRHF calculation did not converge")

        kmp = mp.KMP2(kmf)
        kmp.with_df_ints = True
        _, t2 = kmp.kernel(with_t2=True)
        return kmf, kmp, t2

    def test_mp2ss_stacked_singularity_split_q2_q4(self):
        mp2ss = MP2SS(
            kmf=self.kmf,
            kmp=self.kmp,
            t2=self.t2,
        )
        self.assertEqual(mp2ss.options.auxfunc_direct, "XNGauss")
        self.assertEqual(mp2ss.options.auxfunc_direct_q2, "XNGaussStackedSingularity")
        self.assertEqual(mp2ss.options.auxfunc_direct_dG0, "XNGauss")
        self.assertEqual(mp2ss.options.auxfunc_exchange, "XNGaussStackedSingularity")
        self.assertEqual(mp2ss.options.qG_norm_cutoff, 4.0)
        self.assertEqual(mp2ss.options.fit_method, "scipy_least_squares")
        self.assertTrue(mp2ss.options.fit_with_coul)
        self.assertTrue(mp2ss.options.correct_q2_q4_separately)
        self.assertEqual(mp2ss.options.t2_store_type, "kikjka")
        correction = mp2ss.compute_correction(direct=True, exchange=True)

        actual = {
            "emp2ss": mp2ss.emp2ss,
            "mp2ss_total_correction": mp2ss.mp2ss_total_correction,
            "mp2ss_direct_correction": mp2ss.mp2ss_direct_correction,
            "mp2ss_exchange_correction": mp2ss.mp2ss_exchange_correction,
            "direct_integral_term": mp2ss.direct_integral_term,
            "direct_quadrature_term": mp2ss.direct_quadrature_term,
            "exchange_integral_term": mp2ss.exchange_integral_term,
            "exchange_quadrature_term": mp2ss.exchange_quadrature_term,
            "direct_integral_term_q2": mp2ss.direct_integral_term_q2,
            "direct_quadrature_term_q2": mp2ss.direct_quadrature_term_q2,
            "direct_total_correction_q2": mp2ss.direct_total_correction_q2,
            "direct_integral_term_dG0": mp2ss.direct_integral_term_dG0,
            "direct_quadrature_term_dG0": mp2ss.direct_quadrature_term_dG0,
            "direct_total_correction_dG0": mp2ss.direct_total_correction_dG0,
            "direct_total_correction_q2_dG0": mp2ss.direct_total_correction_q2_dG0,
            "emp2_uncorr": mp2ss.emp2_uncorr,
            "edi_uncorr": mp2ss.edi_uncorr,
            "exi_uncorr": mp2ss.exi_uncorr,
            "emp2ss_direct": mp2ss.emp2ss_direct,
            "emp2ss_exchange": mp2ss.emp2ss_exchange,
        }

        for quantity, reference in self.REFERENCES.items():
            self.assertAlmostEqual(
                actual[quantity],
                reference,
                delta=self.TOL,
                msg=quantity,
            )

        self.assertAlmostEqual(float(correction), self.REFERENCES["mp2ss_total_correction"], delta=self.TOL)
        self.assertAlmostEqual(
            correction.mp2ss_direct_correction,
            self.REFERENCES["mp2ss_direct_correction"],
            delta=self.TOL,
        )
        self.assertAlmostEqual(
            correction.mp2ss_exchange_correction,
            self.REFERENCES["mp2ss_exchange_correction"],
            delta=self.TOL,
        )


if __name__ == "__main__":
    unittest.main()
