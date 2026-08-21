import unittest

import numpy as np
from scipy.integrate import dblquad, quad

from fsec.singularity_subtraction.model_function import ModelFunction
from fsec.singularity_subtraction.model_function.mp2_exchange_modfunc import (
    XNGaussAnisotropicStackedSingularityExchange,
    XNGaussStackedSingularityExchange,
)
from fsec.singularity_subtraction.mp2ss import (
    EXCHANGE_MODEL_SPECS,
    ExchangeCorrectionConfig,
    MP2ExchangeSS,
)


class AnisotropicGaussianExchangeTests(unittest.TestCase):
    def test_integral_and_evaluation_match_independent_expressions(self):
        sx, sy, sz, sigma = parameters = [0.7, 1.3, 2.1, 0.8]
        q2s = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.2, -0.3],
                [-0.4, 0.8, 1.1],
                [0.5, -1.2, 0.7],
            ]
        )
        coords = np.array(
            [
                [0.2, -0.5, 0.7],
                [1.1, 0.4, -0.2],
                [-0.6, 0.3, 0.9],
            ]
        )
        dvol = 0.37
        model = XNGaussAnisotropicStackedSingularityExchange(
            parameters=parameters, q2s=q2s, dvol=dvol
        )

        radial, _ = quad(
            lambda r: r**2 * np.exp(-r**2 / (2 * sigma**2)), 0, np.inf
        )
        angular, _ = dblquad(
            lambda phi, theta: (
                sx * np.sin(theta) ** 2 * np.cos(phi) ** 2
                + sy * np.sin(theta) ** 2 * np.sin(phi) ** 2
                + sz * np.cos(theta) ** 2
            )
            * np.sin(theta),
            0,
            np.pi,
            lambda _theta: 0,
            lambda _theta: 2 * np.pi,
        )
        self.assertAlmostEqual(model.coulomb_integral(), (radial * angular) ** 2)

        q2_norm_squared = np.sum(q2s**2, axis=1)
        nonzero = q2_norm_squared >= 1e-8
        q2_quadratic = np.sum(q2s[nonzero] ** 2 * parameters[:3], axis=1)
        expected_q2 = dvol * np.sum(
            q2_quadratic
            / q2_norm_squared[nonzero]
            * np.exp(-q2_norm_squared[nonzero] / (2 * sigma**2))
        )
        coord_quadratic = np.sum(coords**2 * parameters[:3], axis=1)
        expected = (
            coord_quadratic
            * np.exp(-np.sum(coords**2, axis=1) / (2 * sigma**2))
            * expected_q2
        )
        np.testing.assert_allclose(model.eval_model(coords), expected)

    def test_isotropic_reduction_matches_existing_gaussian(self):
        c0 = 0.36
        sigma = 1.2
        s = np.sqrt(c0)
        q2s = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.5, -0.1, 0.4],
                [-0.7, 0.8, 0.2],
                [1.0, 0.3, -0.6],
            ]
        )
        coords = np.array(
            [[0.2, 0.4, -0.3], [0.9, -0.5, 0.1], [-0.6, 0.7, 1.1]]
        )
        dvol = 0.23
        isotropic = XNGaussStackedSingularityExchange(
            parameters=[c0, sigma], q2s=q2s, dvol=dvol
        )
        anisotropic = XNGaussAnisotropicStackedSingularityExchange(
            parameters=[s, s, s, sigma], q2s=q2s, dvol=dvol
        )

        self.assertAlmostEqual(
            anisotropic.uncorrected_q2_quad,
            s * isotropic.uncorrected_q2_quad,
        )
        np.testing.assert_allclose(
            anisotropic.eval_model(coords), isotropic.eval_model(coords)
        )
        self.assertAlmostEqual(
            anisotropic.coulomb_integral(), isotropic.coulomb_integral()
        )

    def test_registry_and_exchange_factory_configuration(self):
        spec = EXCHANGE_MODEL_SPECS["GaussAnisotropic"]
        self.assertIs(
            ModelFunction.get_class(spec["cls_name"]),
            XNGaussAnisotropicStackedSingularityExchange,
        )

        config = ExchangeCorrectionConfig(
            auxfunc_exchange="GaussAnisotropic",
            fit_class=None,
            fit_with_coul=False,
            qG_norm_cutoff=None,
        )
        calculator = MP2ExchangeSS(config)
        model, initial_params, multipliers = calculator._create_exchange_model(
            "GaussAnisotropic", np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]), 0.5
        )

        self.assertIsInstance(model, XNGaussAnisotropicStackedSingularityExchange)
        np.testing.assert_array_equal(initial_params, [1e-2, 1e-2, 1e-2, 1.0])
        self.assertEqual(model.num_params, 4)
        self.assertEqual(multipliers, [1e2, 1e2, 1e2, 1.0])


if __name__ == "__main__":
    unittest.main()
