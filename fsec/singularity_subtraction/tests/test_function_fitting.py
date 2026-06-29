import unittest

import numpy as np

from fsec.singularity_subtraction.function_fitting import (
    ExxScipyLeastSquares,
    ExxScipyMinimize,
    MP2ScipyLeastSquares,
    MP2ScipyMinimize,
)
from fsec.singularity_subtraction.model_function.exx_modfunc import (
    ContractedGaussianModel,
    QuarticExponentialModel,
)
from fsec.singularity_subtraction.model_function.mp2_direct_modfunc import (
    XNGauss,
    XNGaussStackedSingularityQMesh,
)
from fsec.singularity_subtraction.model_function.mp2_exchange_modfunc import (
    XNGaussStackedSingularityExchange,
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


class KnownValues(unittest.TestCase):
    QG_20 = np.array(
        [
            [0.0, 0.0, 0.0],
            [-1.0471975511965976, 0.0, 0.0],
            [0.0, -1.0471975511965976, 0.0],
            [0.0, 0.0, -1.0471975511965976],
            [0.0, 0.0, 1.0471975511965976],
            [0.0, 1.0471975511965976, 0.0],
            [1.0471975511965976, 0.0, 0.0],
            [-1.0471975511965976, -1.0471975511965976, 0.0],
            [-1.0471975511965976, 0.0, -1.0471975511965976],
            [-1.0471975511965976, 0.0, 1.0471975511965976],
            [-1.0471975511965976, 1.0471975511965976, 0.0],
            [0.0, -1.0471975511965976, -1.0471975511965976],
            [0.0, -1.0471975511965976, 1.0471975511965976],
            [0.0, 1.0471975511965976, -1.0471975511965976],
            [0.0, 1.0471975511965976, 1.0471975511965976],
            [1.0471975511965976, -1.0471975511965976, 0.0],
            [1.0471975511965976, 0.0, -1.0471975511965976],
            [1.0471975511965976, 0.0, 1.0471975511965976],
            [1.0471975511965976, 1.0471975511965976, 0.0],
            [-1.0471975511965976, -1.0471975511965976, -1.0471975511965976],
        ]
    )
    REFERENCE_EXX_20 = np.array(
        [
            1.0000000040758938,
            0.32744907530353146,
            0.3274490753036006,
            0.13484040769431635,
            0.13484040769431635,
            0.3274490753036006,
            0.32744907530353146,
            0.1329367270735302,
            0.055346487300111935,
            0.055346487300113774,
            0.1329367270735332,
            0.055346487300126236,
            0.055346487300125986,
            0.055346487300125986,
            0.055346487300126236,
            0.1329367270735332,
            0.055346487300113774,
            0.055346487300111935,
            0.1329367270735302,
            0.026606903933605867,
        ]
    )
    REFERENCE_MP2_DIRECT_20 = np.array(
        [
            -1.0893511193700196e-21,
            -1.054426172919673e-21,
            -1.0543974002128198e-21,
            -0.00026261641191463426,
            -0.00026261641191463426,
            -1.0543974002128198e-21,
            -1.054426172919673e-21,
            -1.0205123510313412e-21,
            -0.00011402023793530142,
            -0.00011402023793529546,
            -1.0204582352612464e-21,
            -0.00011402023793531614,
            -0.00011402023793530657,
            -0.00011402023793530657,
            -0.00011402023793531614,
            -1.0204582352612464e-21,
            -0.00011402023793529546,
            -0.00011402023793530142,
            -1.0205123510313412e-21,
            -5.774656738478592e-05,
        ]
    )
    REFERENCE_MP2_Q4_20 = np.array(
        [
            -7.717470607835843e-41,
            -7.230553951086863e-41,
            -7.23015934826672e-41,
            -4.485211663249214e-06,
            -4.485211663249214e-06,
            -7.23015934826672e-41,
            -7.230553951086863e-41,
            -6.772916892974211e-41,
            -8.454795391735684e-07,
            -8.454795391734801e-07,
            -6.77219860299934e-41,
            -8.454795391737869e-07,
            -8.45479539173645e-07,
            -8.45479539173645e-07,
            -8.454795391737869e-07,
            -6.77219860299934e-41,
            -8.454795391734801e-07,
            -8.454795391735684e-07,
            -6.772916892974211e-41,
            -2.168660470894743e-07,
        ]
    )
    REFERENCE_MP2_EXCHANGE_20 = np.array(
        [
            5.446755596850098e-22,
            5.272130864598365e-22,
            5.271987001064099e-22,
            0.00013130820595731713,
            0.00013130820595731713,
            5.271987001064099e-22,
            5.272130864598365e-22,
            5.102561755156706e-22,
            5.701011896765071e-05,
            5.701011896764773e-05,
            5.102291176306232e-22,
            5.701011896765807e-05,
            5.7010118967653286e-05,
            5.7010118967653286e-05,
            5.701011896765807e-05,
            5.102291176306232e-22,
            5.701011896764773e-05,
            5.701011896765071e-05,
            5.102561755156706e-22,
            2.887328369239296e-05,
        ]
    )

    @classmethod
    def setUpClass(cls):
        cls.cell = CubicCell(lattice_constant=6.0)
        cls.gamma_qGrid = np.array([[0.0, 0.0, 0.0]])
        cls.mp2_deltaGs = build_cubic_reciprocal_lattice_grid(
            mesh=(29, 29, 29),
            reciprocal_lattice_constant=cls.cell.reciprocal_lattice_constant,
        )
        cls.exx_qG = cls.QG_20
        cls.exx_SqG = cls.REFERENCE_EXX_20

        cls.mp2_qG = cls.QG_20
        cls.SqG_full_direct = cls.REFERENCE_MP2_DIRECT_20
        cls.SqG_full_q4 = cls.REFERENCE_MP2_Q4_20
        cls.SqG_full_exchange = cls.REFERENCE_MP2_EXCHANGE_20

    def _assert_parameters(self, fit_cls, model, input_data, output_data, expected, **fit_kwargs):
        fit_method = fit_cls(model, fit_with_coul=False)
        fitted = fit_method.fit_model(input_data, output_data, **fit_kwargs)
        self.assertEqual(len(fitted), len(expected))
        for actual, reference in zip(fitted, expected):
            self.assertAlmostEqual(actual, reference, places=3)

    def test_exx_scipy_minimize_contracted_gaussian(self):
        model = ContractedGaussianModel(num_gaussians=1, isotropic=True)
        expected = [1.0, 1.0e-8]
        self._assert_parameters(
            ExxScipyMinimize,
            model,
            self.exx_qG,
            self.exx_SqG,
            expected,
        )

    def test_exx_scipy_least_squares_contracted_gaussian(self):
        model = ContractedGaussianModel(num_gaussians=1, isotropic=True)
        expected = [1.0, 0.6485758873718301]
        self._assert_parameters(
            ExxScipyLeastSquares,
            model,
            self.exx_qG,
            self.exx_SqG,
            expected,
        )

    def test_exx_scipy_minimize_quartic_exponential(self):
        model = QuarticExponentialModel(num_primitives=1)
        expected = [1.0, 0.7388025253272993, 3.0887137273423044, 1.4083534745912776]
        self._assert_parameters(
            ExxScipyMinimize,
            model,
            self.exx_qG,
            self.exx_SqG,
            expected,
            force_positive_params=True,
        )

    def test_exx_scipy_least_squares_quartic_exponential(self):
        model = QuarticExponentialModel(num_primitives=1)
        expected = [1.0, 2.1081797177657133, 2.7564722224352383, 0.40869039302204624]
        self._assert_parameters(
            ExxScipyLeastSquares,
            model,
            self.exx_qG,
            self.exx_SqG,
            expected,
            force_positive_params=True,
        )

    def test_mp2_scipy_minimize_direct_xngauss_stacked_singularity(self):
        model = XNGaussStackedSingularityQMesh(
            qGrid=self.gamma_qGrid,
            cell=self.cell,
            parameters=[1.0e-4, 1.0],
            deltaGs=self.mp2_deltaGs,
        )
        expected = [2.7391986894019372e-05, 0.8193662600962808]
        self._assert_parameters(
            MP2ScipyMinimize,
            model,
            self.mp2_qG,
            self.SqG_full_direct,
            expected,
            fit_multipliers=[1.0e4, 1.0],
        )

    def test_mp2_scipy_least_squares_direct_xngauss_stacked_singularity(self):
        model = XNGaussStackedSingularityQMesh(
            qGrid=self.gamma_qGrid,
            cell=self.cell,
            parameters=[1.0e-4, 1.0],
            deltaGs=self.mp2_deltaGs,
        )
        expected = [2.7391967036276372e-05, 0.81936636144651]
        self._assert_parameters(
            MP2ScipyLeastSquares,
            model,
            self.mp2_qG,
            self.SqG_full_direct,
            expected,
        )

    def test_mp2_scipy_minimize_dg0_xngauss(self):
        model = XNGauss(parameters=[1.0e-4, 1.0], negative=True, deg=4)
        expected = [0.5696568538141716, 0.20509585399570057]
        self._assert_parameters(
            MP2ScipyMinimize,
            model,
            self.mp2_qG,
            self.SqG_full_q4,
            expected,
            fit_multipliers=[1.0e4, 1.0],
        )

    def test_mp2_scipy_least_squares_dg0_xngauss(self):
        model = XNGauss(parameters=[1.0e-4, 1.0], negative=True, deg=4)
        expected = [1.3088397673385366e-05, 0.48253452987121476]
        self._assert_parameters(
            MP2ScipyLeastSquares,
            model,
            self.mp2_qG,
            self.SqG_full_q4,
            expected,
        )

    def test_mp2_scipy_minimize_exchange_xngauss_stacked_singularity_exchange(self):
        model = XNGaussStackedSingularityExchange(
            parameters=[1.0e-4, 1.0], q2s=self.mp2_qG, dvol=1.0
        )
        expected = [1.7642180397497834e-05, 0.8193663363312604]
        self._assert_parameters(
            MP2ScipyMinimize,
            model,
            self.mp2_qG,
            self.SqG_full_exchange,
            expected,
            fit_multipliers=[1.0e4, 1.0],
        )

    def test_mp2_scipy_least_squares_exchange_xngauss_stacked_singularity_exchange(self):
        model = XNGaussStackedSingularityExchange(
            parameters=[1.0e-4, 1.0], q2s=self.mp2_qG, dvol=1.0
        )
        expected = [1.7642177723423226e-05, 0.8193663658499301]
        self._assert_parameters(
            MP2ScipyLeastSquares,
            model,
            self.mp2_qG,
            self.SqG_full_exchange,
            expected,
        )


class IsotropicInputShapeTests(unittest.TestCase):
    def setUp(self):
        self.model = QuarticExponentialModel(
            num_primitives=1, parameters=[1.0, 1.0, 1.0, 1.0]
        )
        self.fit_method = ExxScipyLeastSquares(self.model, fit_with_coul=False)
        self.params = np.array(self.model.parameters, dtype=float)

    def test_abs_diff_accepts_nx3_input_by_converting_to_r(self):
        qg = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 3.0, 4.0],
            ]
        )
        r = np.linalg.norm(qg, axis=1)
        output = self.model.eval_model_r(r)

        residual = self.fit_method.abs_diff(self.params, qg, output)
        self.assertTrue(np.allclose(residual, 0.0, atol=1e-12))

    def test_abs_diff_accepts_nx1_input(self):
        r_col = np.array([[0.0], [1.0], [2.0]])
        output = self.model.eval_model_r(r_col[:, 0])

        residual = self.fit_method.abs_diff(self.params, r_col, output)
        self.assertTrue(np.allclose(residual, 0.0, atol=1e-12))

    def test_abs_diff_rejects_invalid_isotropic_shape(self):
        bad_input = np.array([[1.0, 2.0], [3.0, 4.0]])
        output = np.array([0.0, 0.0])

        with self.assertRaises(ValueError):
            _ = self.fit_method.abs_diff(self.params, bad_input, output)


if __name__ == "__main__":
    unittest.main()
