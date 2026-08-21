import numpy as np
from scipy.integrate import quad

from fsec.singularity_subtraction.model_function import ModelFunction


class MP2StackedSingularityExchange(ModelFunction):
    """Auxiliary model for MP2 exchange correction."""

    def __init__(self, parameters=None, q2s=None, dvol=None):
        super().__init__(parameters=parameters, is_contraction=False)
        self.q2s = None if q2s is None else np.asarray(q2s, dtype=float)
        self.dvol = dvol
        self.uncorrected_q2_quad = None
        if self.q2s is not None:
            self.compute_uncorrected_q2_quad()

    def decay_func_r(self, r: np.ndarray):
        pass

    def default_parameters(self):
        pass

    def set_parameters(self, parameters):
        self.parameters = parameters
        if self.q2s is not None:
            self.compute_uncorrected_q2_quad()

    def set_q2s(self, q2s, dvol=None):
        self.q2s = np.asarray(q2s, dtype=float)
        if dvol is not None:
            self.dvol = dvol
        self.compute_uncorrected_q2_quad()

    def compute_uncorrected_q2_quad(self):
        squared_norm_q2s = np.sum(self.q2s**2, axis=1)
        squared_norm_q2s[squared_norm_q2s < 1e-8] = np.inf
        uncorrected_q2_quad = self.dvol * np.sum(self.decay_func_r(np.sqrt(squared_norm_q2s)))
        self.uncorrected_q2_quad = uncorrected_q2_quad
        return uncorrected_q2_quad

    def eval_model(self, coords: np.ndarray):
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError("coords must be an (N,3) array: [q_x, q_y, q_z]")
        if self.q2s is None:
            raise ValueError("q2s must be set on MP2StackedSingularityExchange before evaluation")
        if self.q2s.ndim != 2 or self.q2s.shape[1] != 3:
            raise ValueError("q2s must be an (M,3) array")
        if self.dvol is None:
            raise ValueError("dvol must be set on MP2StackedSingularityExchange before evaluation")
        if self.c0 is None:
            raise ValueError("c0 must be set on MP2StackedSingularityExchange before evaluation")

        c0 = self.c0

        uncorrected_q2_quad = self.uncorrected_q2_quad

        q_norm = np.linalg.norm(coords, axis=1)
        q_norm_squared = q_norm**2

        decay_func_r = self.decay_func_r(q_norm)
        results = c0 * q_norm_squared * decay_func_r * uncorrected_q2_quad

        return results

    def coulomb_integral(self):
        """Compute the coulomb integral as a 1d scipy.integrate.quad integral."""

        def integrand(r):
            return r**2 * self.decay_func_r(r)

        integral, abserror = quad(integrand, 0, np.inf)
        prefactor = 4 * np.pi
        print("Computed integral with scipy.integrate.quad. Estimated error: ", abserror)
        return self.c0 * (prefactor * integral) ** 2


class XNGaussStackedSingularityExchange(MP2StackedSingularityExchange):
    def __init__(self, parameters=None, q2s=None, dvol=None):
        self.sigma = parameters[1]
        self.c0 = parameters[0]
        super().__init__(parameters=parameters, q2s=q2s, dvol=dvol)

    def default_parameters(self):
        return [1.0, 1.0]

    def set_parameters(self, parameters):
        self.sigma = parameters[1]
        self.c0 = parameters[0]
        super().set_parameters(parameters)

    def decay_func_r(self, r: np.ndarray):
        if np.isclose(self.sigma, 0):
            return np.zeros_like(r)
        return np.exp(-r**2 / (2 * self.sigma**2))

    def coulomb_integral(self):
        c0, sigma = self.c0, self.sigma
        if self.c0 is None or self.sigma is None:
            raise ValueError("c0 and sigma must be set on XNGaussStackedSingularityExchange before evaluation")
        if np.isclose(sigma, 0):
            return 0.0
        return c0 * (2 * sigma**2 * np.pi) ** 3


class XNGaussAnisotropicStackedSingularityExchange(MP2StackedSingularityExchange):
    r"""Anisotropic Gaussian exchange model.

    The parameters are ``[sx, sy, sz, sigma]`` and define the quadratic form
    ``q.T @ diag(sx, sy, sz) @ q`` with an isotropic Gaussian decay.
    """

    def __init__(self, parameters=None, q2s=None, dvol=None):
        # Delay attaching q2s until the anisotropic parameters exist because
        # the base class computes the cached q2 quadrature immediately.
        super().__init__(parameters=parameters, q2s=None, dvol=dvol)
        self.set_parameters(self.parameters)
        if q2s is not None:
            self.set_q2s(q2s)

    @staticmethod
    def _quadratic_form(coords, diagonal):
        return np.sum(coords**2 * diagonal, axis=1)

    def default_parameters(self):
        return [1.0, 1.0, 1.0, 1.0]

    def set_parameters(self, parameters):
        parameters = np.asarray(parameters, dtype=float)
        self.parameters = parameters
        self.sx, self.sy, self.sz, self.sigma = parameters
        self.S = np.diag(parameters[:3])
        self.num_params = 4
        if self.q2s is not None:
            self.compute_uncorrected_q2_quad()

    def decay_func_r(self, r: np.ndarray):
        if np.isclose(self.sigma, 0):
            return np.zeros_like(r, dtype=float)
        return np.exp(-r**2 / (2 * self.sigma**2))

    def compute_uncorrected_q2_quad(self):
        squared_norm_q2s = np.sum(self.q2s**2, axis=1)
        squared_norm_q2s[squared_norm_q2s < 1e-8] = np.inf
        quadratic_form = self._quadratic_form(self.q2s, self.parameters[:3])
        angular_weight = quadratic_form / squared_norm_q2s
        self.uncorrected_q2_quad = self.dvol * np.sum(
            angular_weight * self.decay_func_r(np.sqrt(squared_norm_q2s))
        )
        return self.uncorrected_q2_quad

    def eval_model(self, coords: np.ndarray):
        coords = np.asarray(coords)
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError("coords must be an (N,3) array: [q_x, q_y, q_z]")
        if self.q2s is None:
            raise ValueError(
                "q2s must be set on XNGaussAnisotropicStackedSingularityExchange "
                "before evaluation"
            )
        if self.dvol is None:
            raise ValueError(
                "dvol must be set on XNGaussAnisotropicStackedSingularityExchange "
                "before evaluation"
            )

        quadratic_form = self._quadratic_form(coords, self.parameters[:3])
        return (
            quadratic_form
            * self.decay_func_r(np.linalg.norm(coords, axis=1))
            * self.uncorrected_q2_quad
        )

    def coulomb_integral(self):
        if np.isclose(self.sigma, 0):
            return 0.0
        mean_s = (self.sx + self.sy + self.sz) / 3
        return (2 * np.pi * self.sigma**2) ** 3 * mean_s**2


class XNExponentialStackedSingularityExchange(MP2StackedSingularityExchange):
    def __init__(self, parameters=None, q2s=None, dvol=None):
        self.c0 = parameters[0]
        self.alpha = parameters[1]
        self.gamma = parameters[2]
        super().__init__(parameters=parameters, q2s=q2s, dvol=dvol)

    def default_parameters(self):
        return [1.0, 1.0, 1.0]

    def set_parameters(self, parameters):
        self.alpha = parameters[1]
        self.gamma = parameters[2]
        self.c0 = parameters[0]
        super().set_parameters(parameters)

    def decay_func_r(self, r: np.ndarray):
        if np.isclose(self.alpha, 0):
            return np.zeros_like(r)
        return np.exp(-self.alpha * (np.sqrt(1 + self.gamma * r**2) - 1))


class XNExpAbsStackedSingularityExchange(MP2StackedSingularityExchange):
    def __init__(self, parameters=None, q2s=None, dvol=None):
        self.c0 = parameters[0]
        self.alpha = parameters[1]
        super().__init__(parameters=parameters, q2s=q2s, dvol=dvol)

    def default_parameters(self):
        return [1.0, 1.0]

    def set_parameters(self, parameters):
        self.alpha = parameters[1]
        self.c0 = parameters[0]
        super().set_parameters(parameters)

    def decay_func_r(self, r: np.ndarray):
        if np.isclose(self.alpha, 0):
            return np.zeros_like(r)
        return np.exp(-self.alpha * r)

    def coulomb_integral(self):
        c0, alpha = self.c0, self.alpha
        if self.c0 is None or self.alpha is None:
            raise ValueError("c0 and alpha must be set on XNExpAbsStackedSingularityExchange before evaluation")
        if np.isclose(alpha, 0):
            return 0.0
        return c0 * (8 * np.pi / (alpha**3)) ** 2


class XNExpAbs2StackedSingularityExchange(MP2StackedSingularityExchange):
    def __init__(self, parameters=None, q2s=None, dvol=None):
        self.c0 = parameters[0]
        self.alpha = parameters[1]
        self.beta = parameters[2]
        super().__init__(parameters=parameters, q2s=q2s, dvol=dvol)

    def default_parameters(self):
        return [1.0, 1.0, 1.0]

    def set_parameters(self, parameters):
        self.alpha = parameters[1]
        self.beta = parameters[2]
        self.c0 = parameters[0]
        super().set_parameters(parameters)

    def decay_func_r(self, r: np.ndarray):
        if np.isclose(self.alpha, 0):
            return np.zeros_like(r)
        return np.exp(-self.alpha * r - self.beta * r**2)


class XNQuarticExponentialStackedSingularityExchange(MP2StackedSingularityExchange):
    def __init__(self, parameters=None, q2s=None, dvol=None):
        self.c0 = parameters[0]
        self.alpha = parameters[1]
        self.beta = parameters[2]
        self.kappa = parameters[3]
        super().__init__(parameters=parameters, q2s=q2s, dvol=dvol)

    def default_parameters(self):
        return [1.0, 1.0, 1.0, 1.0]

    def set_parameters(self, parameters):
        self.alpha = parameters[1]
        self.beta = parameters[2]
        self.kappa = parameters[3]
        self.c0 = parameters[0]
        super().set_parameters(parameters)

    def decay_func_r(self, r: np.ndarray):
        if np.isclose(self.alpha, 0):
            return np.zeros_like(r)
        gamma1 = self.beta / self.alpha
        gamma2 = self.kappa / self.alpha
        return np.exp(-self.alpha * (np.sqrt(1 + gamma1 * r**2 + gamma2 * r**4) - 1))
