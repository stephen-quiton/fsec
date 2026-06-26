from abc import ABC, abstractmethod
import numpy as np
from singularity_subtraction.model_function import ModelFunction, IsotropicModel
from scipy.optimize import least_squares, minimize

class FitMethod(ABC):
    def __init__(self, model_function: ModelFunction, **kwargs):
        self.model_function = model_function
        if isinstance(model_function, IsotropicModel):
            self.abs_diff = self.abs_diff_r
        else:
            self.abs_diff = self.abs_diff_xyz
        self.initial_guess = kwargs.get('initial_guess')
        print("Using initial_guess: ", self.initial_guess)
        if self.initial_guess is None:
            self.initial_guess = self.default_initial_guess()
        self.fit_with_coul = kwargs.get('fit_with_coul', False)
        self.coul_deg = kwargs.get('coul_deg', 2)
        self.dump_flags()

    @abstractmethod
    def fit_model(self, input_data: np.ndarray, output_data: np.ndarray,fit_multipliers=None):
        """
        Fit the model based on input_data and output_data. Output data should be normalized (i.e.S(0)=1 for exx)
        Should return the optimized parameters.
        """
        pass

    @abstractmethod
    def dump_flags(self):
        """
        For printing the fitting method used
        """
        pass

    def residuals(self, params_input, input_data: np.ndarray, output_data: np.ndarray, pow=2):
        """
        Compute the residuals between the model and the data.
        """
        result = np.sum(self.abs_diff(params_input, input_data, output_data) ** pow)
        return result

    def abs_diff_r(self, params_input, input_data: np.ndarray, output_data: np.ndarray):
        """
        Compute the residuals between the model and the data.
        """
        input_data = self._prepare_input_data(input_data)
        self.model_function.set_parameters(params_input)
        f_fit = self.model_function.eval_model_r(input_data)
        abs_diff = np.abs(f_fit - output_data)
        if self.fit_with_coul:
            abs_diff *= 1./input_data**self.coul_deg
            abs_diff = abs_diff[input_data > 1e-8]
        return abs_diff

    def abs_diff_xyz(self, params_input, input_data: np.ndarray, output_data: np.ndarray):
        """
        Compute the residuals between the model and the data.
        """
        self.model_function.set_parameters(params_input)
        f_fit = self.model_function.eval_model(input_data)
        abs_diff = f_fit - output_data
        if self.fit_with_coul:
            abs_diff *= 1./np.sum(input_data**self.coul_deg, axis=1)
            abs_diff = abs_diff[np.linalg.norm(input_data, axis=1) > 1e-8] # take out the singular point
        return abs_diff

    def _prepare_input_data(self, input_data: np.ndarray):
        """
        Normalize isotropic model inputs to radial data.
        """
        input_data = np.asarray(input_data)
        if not isinstance(self.model_function, IsotropicModel):
            return input_data

        if input_data.ndim == 1:
            return input_data

        if input_data.ndim == 2 and input_data.shape[1] == 1:
            return input_data[:, 0]

        if input_data.ndim == 2 and input_data.shape[1] == 3:
            return np.linalg.norm(input_data, axis=1)

        raise ValueError(
            "For IsotropicModel, input_data must have shape (N,), (N, 1), or (N, 3). "
            f"Got {input_data.shape}."
        )


    def default_initial_guess(self):
        """
        Return an initial guess for the model parameters.
        """
        return np.ones_like(self.model_function.default_parameters())
    
    
class MP2ScipyMinimize(FitMethod):
    def __init__(self, model_function, **kwargs):
        self.is_contraction = kwargs.get('is_contraction', False)
        super().__init__(model_function, **kwargs)
    
    def fit_model(self, input_data: np.ndarray, output_data: np.ndarray, force_positive_params=True, fit_multipliers=None, x0_points=None,fixed_params=None):
        """
        Finds optimal parameters for the model using scipy.optimize.minimize
        """
        input_data = self._prepare_input_data(input_data)
        
        if fit_multipliers is None:
            fit_multipliers = np.ones(len(self.model_function.parameters))
        if x0_points is None:
            if input_data.ndim == 1:
                x0_points = [0.0]
            else:
                x0_points = [np.array([0.,0.,0.])]
            
        lb = 1e-8 if force_positive_params else -np.inf
        
        normalization = np.max(np.abs(output_data)) # To avoid numerical issues with very small residuals

            
        def residuals(params, input_data, output_data):
            self.model_function.set_parameters(params / fit_multipliers)
            if isinstance(self.model_function, IsotropicModel):
                f_fit = self.model_function.eval_model_r(input_data)
            else:
                f_fit = self.model_function.eval_model(input_data)
            abs_diff = np.abs(f_fit - output_data)
            if self.fit_with_coul:
                input_data_updated = input_data.copy()
                for x0 in x0_points:
                    if input_data_updated.ndim == 1:
                        abs_diff *= 1./(input_data_updated - x0)**2
                        mask = np.abs(input_data_updated - x0) > 1e-8
                    else:
                        abs_diff *= 1./np.sum((input_data_updated - x0)**2, axis=1)
                        mask = np.linalg.norm(input_data_updated-x0, axis=1) > 1e-8
                    abs_diff = abs_diff[mask] # take out the singular points
                    input_data_updated = input_data_updated[mask] # take out the singular points
            return np.sum(abs_diff**2)/normalization**2
        
        if fixed_params is not None:
            def residuals_to_fit(params, input_data, output_data):
                params_updated = params.copy()
                params_updated[fixed_params] = self.model_function.parameters[fixed_params]
                return residuals(params_updated, input_data, output_data)
        else:
            residuals_to_fit = residuals
            
        input_initial_guess = self.initial_guess * fit_multipliers
        constraints = ()
        bounds = [(lb, np.inf)] * len(self.model_function.parameters)
        result = minimize(residuals_to_fit, input_initial_guess, args=(input_data, output_data), constraints=constraints,
                          bounds=bounds, options={'disp':True, 'ftol':1e-10, 'gtol':1e-10})
        if not result.success:
            print("WARNING:Fitting failed. Result: ", result)
            raise ValueError("Fitting failed. Result: ", result)
        self.fitted_parameters = result.x / fit_multipliers
        return self.fitted_parameters
    
    def dump_flags(self):
        """
        Dump the flags used in the fitting.
        """
        print("<class MP2ScipyMinimize> with model function", self.model_function)
        print("Fitting with coul: ", self.fit_with_coul)


class ExxScipyMinimize(FitMethod):
    def __init__(self, model_function : ModelFunction, **kwargs):
        self.is_contraction = kwargs.get('is_contraction', False)
        super().__init__(model_function, **kwargs)

    def residuals_set_c0(self, params_input, input_data: np.ndarray, output_data: np.ndarray):
        """
        Manually set first parameter to 1. To avoid using `constraint` object.
        """
        params = np.zeros(len(params_input)+1)
        params[1:] = params_input
        params[0] = 1
        # self.model_function.set_parameters(params)
        # return self.residuals(params, input_data, output_data, pow)
        return self.residuals(params, input_data, output_data)

    def fit_model(self, input_data: np.ndarray, output_data: np.ndarray, force_positive_params=True, fit_multipliers=None):
        """
        Finds optimal parameters for the model using scipy.optimize.minimize
        """
        constraints = ()
        lb = 1e-8 if force_positive_params else -np.inf
        bounds = [(lb, np.inf)] * len(self.model_function.parameters)


        input_initial_guess = self.initial_guess
        residuals = self.residuals
        is_contraction = self.model_function.is_contraction
        if is_contraction:
            # Constraint where all c_i must be positive and sum to 1
            if self.model_function.num_primitives > 1:
                def normalization(params):
                    return np.sum(params[::self.model_function.num_primitive_params]) - 1
                constraints = [
                    {'type': 'eq', 'fun': normalization},
                ]
                bounds[::self.model_function.num_primitive_params] = [(0.0, np.inf)]
            elif self.model_function.num_primitives == 1:
                # Try to avoid setting a `constraint` to use LBFGS instead of SLSQP
                input_initial_guess = input_initial_guess[1:]
                bounds = bounds[1:]
                residuals = self.residuals_set_c0
            else:
                raise ValueError("num_primitives must be > 0")


        result = minimize(residuals, input_initial_guess, args=(input_data, output_data), constraints=constraints,
                          bounds=bounds,options={'disp':True, 'ftol':1e-8, 'gtol':1e-8})
        self.fitted_parameters = result.x
        if not result.success:
            print("WARNING:Fitting failed. Result: ", result)
            raise ValueError("Fitting failed. Result: ", result)

        if is_contraction and self.model_function.num_primitives == 1:
            params = np.zeros(len(result.x)+1)
            params[1:] = result.x
            params[0] = 1.0
            self.fitted_parameters = params

        print("Fitting complete. Parameters: ", self.fitted_parameters)
        print("Sum residual: ", result.fun)
        print("Force positive params: ", force_positive_params)
        return self.fitted_parameters

    def dump_flags(self):
        """
        Dump the flags used in the fitting.
        """
        print("<class ExxScipyMinimize> with model function", self.model_function)
        print("Fitting with coul: ", self.fit_with_coul)


class ExxScipyLeastSquares(FitMethod):
    def __init__(self, model_function : ModelFunction, **kwargs):
        super().__init__(model_function, **kwargs)

    def abs_diff_set_c0(self, params_input, input_data: np.ndarray, output_data: np.ndarray):
        """
        Instead of setting constraint \sum_i c_i = 1, we force c_0 = 1 - \sum_i c_i and remove c_0 from
        the optimization.
        """
        num_primitive_params = self.model_function.num_primitive_params
        params = np.zeros(len(params_input)+1)
        params[1:] = params_input
        params[0] = 1 - np.sum(params_input[num_primitive_params-1::num_primitive_params])
        self.model_function.set_parameters(params)
        # return self.residuals(params, input_data, output_data, pow)
        return self.abs_diff(params, input_data, output_data)

    def fit_model(self, input_data: np.ndarray, output_data: np.ndarray, force_positive_params=False, fit_multipliers=None):
        """
        Finds optimal parameters for the model using scipy.optimize.least_squares
        """
        # Setup bounds for the parameters
        is_contraction = self.model_function.is_contraction
        lb = 0. if force_positive_params else -np.inf
        upper_bounds = [np.inf]*len(self.model_function.parameters)
        lower_bounds = [lb]*len(self.model_function.parameters)
        input_initial_guess = self.initial_guess
        if is_contraction:
            lower_bounds[::self.model_function.num_primitive_params] = [0.0]*self.model_function.num_primitives

            # Remove the first parameter (c_i), because we are fitting with normalization
            upper_bounds = upper_bounds[1:]
            lower_bounds = lower_bounds[1:]

            input_initial_guess = self.initial_guess[1:]
        bounds = (lower_bounds,upper_bounds)


        abs_diff = self.abs_diff_set_c0 if is_contraction else self.abs_diff
        result = least_squares(abs_diff, input_initial_guess, args=(input_data, output_data), bounds=bounds, verbose=2)
        if not result.success:
            print("WARNING: Fitting failed. Result: ", result)
            raise ValueError("Fitting failed. Result: ", result)
        
        # Check if fit is sigificantly incorrect.
        normalization = np.sum(np.abs(output_data))
        npoints = len(input_data)
        abs_rel_error = np.sum(abs_diff(result.x, input_data, output_data))/normalization/npoints
        if abs_rel_error > 0.5:
            print("WARNING: Relative error (", abs_rel_error, ") is too large.")
            
        if is_contraction:
            params = np.zeros(len(result.x)+1)
            params[1:] = result.x
            params[0] = 1 - \
                np.sum(result.x[self.model_function.num_primitive_params-1::self.model_function.num_primitives])
            self.fitted_parameters = params
        else:
            self.fitted_parameters = result.x

        print("Fitting complete. Parameters: ", self.fitted_parameters)
        print("Sum residual: ", np.sum((result.fun)**2))
        print("Force positive params: ", force_positive_params)
        return self.fitted_parameters

    def dump_flags(self):
        """
        Dump the flags used in the fitting.
        """
        print("<class MP2ScipyLeastSquares> with model function", self.model_function)
        print("Fitting with coul: ", self.fit_with_coul)


class MP2ScipyLeastSquares(FitMethod):
    def __init__(self, model_function : ModelFunction, **kwargs):
        super().__init__(model_function, **kwargs)
        
    def fit_model(self, input_data: np.ndarray, output_data: np.ndarray, force_positive_params=False, fit_multipliers=None,
                  fixed_params=None,jac='2-point',x_scale=1.0,max_nfev=None):
        """
        Finds optimal parameters for the model using scipy.optimize.least_squares
        """
        # Setup bounds for the parameters
        normalization = np.sum(np.abs(output_data)) # To avoid numerical issues with very small residuals

        lb = 1e-14 * normalization if force_positive_params else -np.inf
        upper_bounds = [np.inf]*len(self.model_function.parameters)
        lower_bounds = [lb]*len(self.model_function.parameters)
        input_initial_guess = self.initial_guess
        bounds = (lower_bounds,upper_bounds)
        
        
        def abs_diff(params, input_data, output_data):
            return self.abs_diff(params, input_data, output_data)/normalization

        result = least_squares(abs_diff, input_initial_guess, args=(input_data, output_data), bounds=bounds, verbose=2, jac=jac,
                               x_scale=x_scale, max_nfev=max_nfev,ftol=1e-12,gtol=1e-12)
        if not result.success:
            print("WARNING:Fitting failed. Result: ", result)
            raise ValueError("Fitting failed. Result: ", result)
        
        # Check if fit is sigificantly incorrect.
        npoints = len(input_data)
        abs_rel_error = np.sum(np.abs(abs_diff(result.x, input_data, output_data)))#/normalization/npoints # dont normalize by normalization/npoints?
        if abs_rel_error > 0.5:
            print("WARNING: Relative error (", abs_rel_error, ") is too large.")

        self.fitted_parameters = result.x


        print("Fitting complete. Parameters: ", self.fitted_parameters)
        print("Sum residual: ", np.sum((result.fun)**2))
        print("Relative error: ", abs_rel_error)
        print("Force positive params: ", force_positive_params)
        return self.fitted_parameters

    def dump_flags(self):
        """
        Dump the flags used in the fitting.
        """
        print("<class MP2ScipyLeastSquares> with model function", self.model_function)
        print("Fitting with coul: ", self.fit_with_coul)
