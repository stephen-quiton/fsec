import unittest

import numpy as np

from singularity_subtraction.model_function.exx_modfunc import (
    ContractedGaussianModel,
    ExpoLorentzianModel,
    ExponentialModel,
    QuarticExponentialModel,
)
from singularity_subtraction.model_function.mp2_direct_modfunc import (
    XNExpAbs,
    XNExpAbs2,
    XNExpAbsStackedSingularity,
    XNExpAbsStackedSingularityQMesh,
    XNExponential,
    XNGauss,
    XNGaussStackedSingularity,
    XNGaussStackedSingularityQMesh,
    XNQuarticExponential,
)
from singularity_subtraction.model_function.mp2_exchange_modfunc import (
    XNExpAbs2StackedSingularityExchange,
    XNExpAbsStackedSingularityExchange,
    XNExponentialStackedSingularityExchange,
    XNGaussStackedSingularityExchange,
    XNQuarticExponentialStackedSingularityExchange,
)


class CubicCell:
    def __init__(self, lattice_constant):
        self.lattice_constant = lattice_constant
        self.reciprocal_lattice_constant = 2 * np.pi / lattice_constant

    def get_scaled_kpts(self, kpts):
        return np.asarray(kpts) / self.reciprocal_lattice_constant

    def get_abs_kpts(self, scaled_kpts):
        return np.asarray(scaled_kpts) * self.reciprocal_lattice_constant


def build_cubic_reciprocal_lattice_grid(mesh, reciprocal_lattice_constant):
    gx, gy, gz = np.meshgrid(
        *[np.fft.fftfreq(n, d=1 / n) for n in mesh],
        indexing="ij",
    )
    return (
        np.hstack((gx.reshape(-1, 1), gy.reshape(-1, 1), gz.reshape(-1, 1)))
        * reciprocal_lattice_constant
    )


class CoulombIntegralDefaultParameterTests(unittest.TestCase):
    cell = CubicCell(lattice_constant=6.0)
    deltaGs = build_cubic_reciprocal_lattice_grid(
        mesh=(29, 29, 29),
        reciprocal_lattice_constant=cell.reciprocal_lattice_constant,
    )
    qGrid = np.array([[0.0, 0.0, 0.0]])
    q2s = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
        ]
    )

    def _assert_coulomb_integral(self, model, expected):
        actual = model.coulomb_integral()
        self.assertTrue(np.isfinite(actual))
        self.assertAlmostEqual(actual, expected, places=12)

    def test_exx_model_coulomb_integrals_with_default_parameters(self):
        cases = [
            (
                "ContractedGaussianModel",
                ContractedGaussianModel(num_gaussians=1, isotropic=True),
                0.7978845608028654,
            ),
            (
                "ExponentialModel",
                ExponentialModel(num_primitives=1, isotropic=True),
                0.8633719706943167,
            ),
            (
                "ExpoLorentzianModel",
                ExpoLorentzianModel(num_primitives=1),
                0.7285878253497297,
            ),
            (
                "QuarticExponentialModel",
                QuarticExponentialModel(num_primitives=1),
                0.6456928457638528,
            ),
        ]

        for name, model, expected in cases:
            with self.subTest(model=name):
                self._assert_coulomb_integral(model, expected)

    def test_mp2_direct_model_coulomb_integrals_with_default_parameters(self):
        cases = [
            ("XNGauss", XNGauss(), -15.749609945722417),
            ("XNExpAbs", XNExpAbs(parameters=[1.0, 1.0]), -25.132741228718345),
            (
                "XNExpAbs2",
                XNExpAbs2(parameters=[1.0, 1.0, 1.0]),
                -2.0009560178328294,
            ),
            (
                "XNExponential",
                XNExponential(parameters=[1.0, 1.0, 1.0]),
                -55.50276939602421,
            ),
            (
                "XNQuarticExponential",
                XNQuarticExponential(parameters=[1.0, 1.0, 1.0, 1.0]),
                -7.358879567252773,
            ),
            (
                "XNGaussStackedSingularity",
                XNGaussStackedSingularity(deltaGs=self.deltaGs),
                -216.00000000000043,
            ),
            (
                "XNGaussStackedSingularityQMesh",
                XNGaussStackedSingularityQMesh(
                    qGrid=self.qGrid,
                    cell=self.cell,
                    deltaGs=self.deltaGs,
                ),
                -216.00000000000043,
            ),
            (
                "XNExpAbsStackedSingularity",
                XNExpAbsStackedSingularity(deltaGs=self.deltaGs),
                -549.9897072725538,
            ),
            (
                "XNExpAbsStackedSingularityQMesh",
                XNExpAbsStackedSingularityQMesh(
                    qGrid=self.qGrid,
                    cell=self.cell,
                    deltaGs=self.deltaGs,
                ),
                -549.9897072725538,
            ),
        ]

        for name, model, expected in cases:
            with self.subTest(model=name):
                self._assert_coulomb_integral(model, expected)

    def test_mp2_exchange_model_coulomb_integrals_with_default_parameters(self):
        cases = [
            (
                "XNGaussStackedSingularityExchange",
                XNGaussStackedSingularityExchange(
                    parameters=[1.0, 1.0],
                    q2s=self.q2s,
                    dvol=1.0,
                ),
                248.05021344239853,
            ),
            (
                "XNExponentialStackedSingularityExchange",
                XNExponentialStackedSingularityExchange(
                    parameters=[1.0, 1.0, 1.0],
                    q2s=self.q2s,
                    dvol=1.0,
                ),
                3080.557410628242,
            ),
            (
                "XNExpAbsStackedSingularityExchange",
                XNExpAbsStackedSingularityExchange(
                    parameters=[1.0, 1.0],
                    q2s=self.q2s,
                    dvol=1.0,
                ),
                631.6546816697189,
            ),
            (
                "XNExpAbs2StackedSingularityExchange",
                XNExpAbs2StackedSingularityExchange(
                    parameters=[1.0, 1.0, 1.0],
                    q2s=self.q2s,
                    dvol=1.0,
                ),
                4.0038249853014145,
            ),
            (
                "XNQuarticExponentialStackedSingularityExchange",
                XNQuarticExponentialStackedSingularityExchange(
                    parameters=[1.0, 1.0, 1.0, 1.0],
                    q2s=self.q2s,
                    dvol=1.0,
                ),
                54.15310848533036,
            ),
        ]

        for name, model, expected in cases:
            with self.subTest(model=name):
                self._assert_coulomb_integral(model, expected)


if __name__ == "__main__":
    unittest.main()
