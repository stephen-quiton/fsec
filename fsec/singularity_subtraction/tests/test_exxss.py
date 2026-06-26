import unittest
import numpy as np
from pyscf.pbc import df
from pyscf.pbc import dft as pbcdft
from pyscf.pbc import gto as pbcgto
from pyscf.pbc import scf as pbcscf

from fsec.singularity_subtraction import ExxSS, ExxSSQuarticExponential


class KnownValues(unittest.TestCase):
    TOL = 1e-6
    REFERENCES = {
        "pbe0_contracted_gaussian": {
            "Ek_ss": -0.5260391467546657,
            "correction": -0.25938972319027775,
            "integral_term": -0.4703482793474786,
            "quadrature_term": 0.21095855615720083,
        },
        "hf_contracted_gaussian": {
            "Ek_ss": -0.5260391467530449,
            "correction": -0.25938972319001474,
            "integral_term": -0.47034827934593426,
            "quadrature_term": 0.21095855615591952,
        },
        "pbe0_quartic_exponential": {
            "Ek_ss": -0.525063933771805,
            "correction": -0.2584145102074171,
            "integral_term": -0.5391775947159534,
            "quadrature_term": 0.28076308450853626,
        },
        "hf_quartic_exponential": {
            "Ek_ss": -0.5250639337242369,
            "correction": -0.2584145101612068,
            "integral_term": -0.539177573428092,
            "quadrature_term": 0.2807630632668852,
        },
    }

    @classmethod
    def setUpClass(cls):
        cls.pbe0_mf = cls._run_pbe0()
        cls.hf_mf = cls._run_hf()

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
    def _run_pbe0(cls):
        cell, kpts = cls._build_h2_cell()
        mf = pbcdft.KRKS(cell, kpts)
        mf.xc = "PBE0"
        return cls._run_scf(mf, cell, kpts)

    @classmethod
    def _run_hf(cls):
        cell, kpts = cls._build_h2_cell()
        mf = pbcscf.KRHF(cell, kpts)
        return cls._run_scf(mf, cell, kpts)

    @staticmethod
    def _run_scf(mf, cell, kpts):
        mf.exxdiv = "ewald"
        mf.with_df = df.GDF(cell, kpts).build()
        mf.kernel()
        if not mf.converged:
            raise RuntimeError("Reference mean-field calculation did not converge")
        return mf

    def _assert_exxss_reference(self, label, mf, exxss_cls, **kwargs):
        exxss = exxss_cls(
            mf,
            fit_method="scipy_least_squares",
            fit_with_coul=True,
            **kwargs,
        )
        exxss.compute_correction()

        actual = {
            "Ek_ss": exxss.Ek_ss,
            "correction": exxss.correction,
            "integral_term": exxss.integral_term,
            "quadrature_term": exxss.quadrature_term,
        }
        for quantity, reference in self.REFERENCES[label].items():
            self.assertAlmostEqual(
                actual[quantity],
                reference,
                delta=self.TOL,
                msg=f"{label} {quantity}",
            )

    def test_pbe0_exxss_contracted_gaussian(self):
        self._assert_exxss_reference(
            "pbe0_contracted_gaussian",
            self.pbe0_mf,
            ExxSS,
        )

    def test_hf_exxss_contracted_gaussian(self):
        self._assert_exxss_reference(
            "hf_contracted_gaussian",
            self.hf_mf,
            ExxSS,
        )

    def test_pbe0_exxss_quartic_exponential(self):
        self._assert_exxss_reference(
            "pbe0_quartic_exponential",
            self.pbe0_mf,
            ExxSSQuarticExponential,
            qG_norm_cutoff_sigma=2.0,
        )

    def test_hf_exxss_quartic_exponential(self):
        self._assert_exxss_reference(
            "hf_quartic_exponential",
            self.hf_mf,
            ExxSSQuarticExponential,
            qG_norm_cutoff_sigma=2.0,
        )


if __name__ == "__main__":
    unittest.main()
