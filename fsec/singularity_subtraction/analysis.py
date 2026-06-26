import inspect

import numpy as np
import scipy.special

import singularity_subtraction.model_function as model_function
from singularity_subtraction.function_fitting import MP2ScipyLeastSquares


class OriginDiagnostics:
    """Estimates finite-size origin-sampling errors in the MP2 structure factor.

    Methods here examine how sensitive fitted model parameters and Coulomb
    integrals are to removing the q-point(s) closest to the origin along each
    reciprocal lattice vector.
    """

    def __init__(self, fit_class, curvature_n_nearest=50, mp2_structure_factor=None):
        self.fit_class = fit_class
        self.curvature_n_nearest = curvature_n_nearest
        self.mp2_structure_factor = mp2_structure_factor

    def estimate_origin_error(self, SqG, qG_full, Lvec_recip, auxfunc,
                              coul_deg=2, fit_multipliers=None):
        """Estimate the origin-error contribution to the first fit coefficient ``c0``
        and to the model's Coulomb integral."""
        model_name = auxfunc.__class__.__name__
        print(f"OriginDiagnostics: estimate_origin_error: estimating origin error from {model_name} fit "
              f"(fit_with_coul=True, coul_deg={coul_deg})...")

        qG_full = np.asarray(qG_full)
        SqG = np.asarray(SqG)
        Lvec_recip = np.asarray(Lvec_recip)

        norms = np.linalg.norm(qG_full, axis=1)
        tol_align = 1e-8
        tol_origin = 1e-8
        drop_indices = set()
        for i in range(Lvec_recip.shape[0]):
            b_i = Lvec_recip[i]
            cross_norms = np.linalg.norm(np.cross(qG_full, b_i), axis=1)
            is_parallel = (cross_norms < tol_align) & (norms > tol_origin)
            if not np.any(is_parallel):
                print(f"  WARNING: no qG points found along reciprocal lattice vector b_{i}")
                continue
            masked_norms = np.where(is_parallel, norms, np.inf)
            drop_indices.add(int(np.argmin(masked_norms)))

        drop_indices = sorted(drop_indices)
        print(f"  Removing {len(drop_indices)} qG point(s) closest to origin along reciprocal lattice vectors:")
        for idx in drop_indices:
            print(f"    qG[{idx}] = {qG_full[idx]}, |qG| = {norms[idx]:.6e}")

        keep_mask = np.ones(len(qG_full), dtype=bool)
        if drop_indices:
            keep_mask[np.array(drop_indices)] = False

        initial_params = np.asarray(auxfunc.parameters).copy()

        try:
            ci_sig = inspect.signature(auxfunc.coulomb_integral)
            ci_accepts_coul_deg = 'coul_deg' in ci_sig.parameters
        except (TypeError, ValueError):
            ci_accepts_coul_deg = False

        def _coulomb_integral():
            if ci_accepts_coul_deg:
                return auxfunc.coulomb_integral(coul_deg=coul_deg)
            return auxfunc.coulomb_integral()

        def _fit(qG_pts, SqG_vals, label):
            auxfunc.set_parameters(initial_params.copy())
            fit_method = self.fit_class(auxfunc, fit_with_coul=True, coul_deg=coul_deg)
            fit_method.initial_guess = initial_params
            print(f"  Fitting {model_name} [{label}] with {len(qG_pts)} points...")
            fitted_params = fit_method.fit_model(
                qG_pts, SqG_vals,
                fit_multipliers=fit_multipliers,
                force_positive_params=True,
                max_nfev=1000 * auxfunc.num_params,
            )
            auxfunc.set_parameters(fitted_params)
            integral = _coulomb_integral()
            return fitted_params[0], fitted_params, integral

        c0_full, params_full, integral_full = _fit(qG_full, SqG, label='all points')
        c0_dropped, params_dropped, integral_dropped = _fit(
            qG_full[keep_mask], SqG[keep_mask], label='closest-along-recip removed'
        )

        auxfunc.set_parameters(initial_params.copy())

        delta_c0 = c0_full - c0_dropped
        delta_integral = integral_full - integral_dropped

        print(f"  {model_name} params (all points)         : {params_full}")
        print(f"  {model_name} params (closest removed)    : {params_dropped}")
        print(f"  c0 (all points)                          : {c0_full}")
        print(f"  c0 (closest along b_i removed)           : {c0_dropped}")
        print(f"  delta c0 (all - dropped)                 : {delta_c0}")
        print(f"  coulomb_integral (all points)            : {integral_full}")
        print(f"  coulomb_integral (closest along b_i removed): {integral_dropped}")
        print(f"  delta coulomb_integral (all - dropped)   : {delta_integral}")

        self.origin_error_delta_c0 = delta_c0
        self.origin_error_c0_full = c0_full
        self.origin_error_c0_dropped = c0_dropped
        self.origin_error_delta_integral = delta_integral
        self.origin_error_integral_full = integral_full
        self.origin_error_integral_dropped = integral_dropped

        return (delta_c0, c0_full, c0_dropped,
                delta_integral, integral_full, integral_dropped)

    def estimate_origin_derivative_error(self, SqG, qG_full, Lvec_recip, deriv_order,
                                         tol=1e-6):
        """Estimate the change in the n-th derivative of S(q) at the origin."""
        print(f"OriginDiagnostics: estimate_origin_derivative_error: estimating change in the "
              f"{deriv_order}-th derivative at q=0 via Fornberg "
              f"(averaged over reciprocal lattice vectors)...")

        qG_full = np.asarray(qG_full)
        SqG = np.asarray(SqG)
        Lvec_recip = np.asarray(Lvec_recip)

        norms = np.linalg.norm(qG_full, axis=1)
        tol_align = 1e-8
        tol_origin = 1e-8
        drop_indices = set()
        for i in range(Lvec_recip.shape[0]):
            b_i = Lvec_recip[i]
            cross_norms = np.linalg.norm(np.cross(qG_full, b_i), axis=1)
            is_parallel = (cross_norms < tol_align) & (norms > tol_origin)
            if not np.any(is_parallel):
                print(f"  WARNING: no qG points found along reciprocal lattice vector b_{i}")
                continue
            masked_norms = np.where(is_parallel, norms, np.inf)
            drop_indices.add(int(np.argmin(masked_norms)))

        drop_indices = sorted(drop_indices)
        print(f"  Removing {len(drop_indices)} qG point(s) closest to origin along reciprocal lattice vectors:")
        for idx in drop_indices:
            print(f"    qG[{idx}] = {qG_full[idx]}, |qG| = {norms[idx]:.6e}")

        keep_mask = np.ones(len(qG_full), dtype=bool)
        if drop_indices:
            keep_mask[np.array(drop_indices)] = False

        A_cols = np.asarray(Lvec_recip).T

        def _symmetrize_dedup(qG, S, dedup_tol=1e-8):
            qG = np.asarray(qG)
            S = np.asarray(S)
            local_norms = np.linalg.norm(qG, axis=1)
            nonzero = local_norms > tol_origin
            qG_aug = np.concatenate([qG, -qG[nonzero]], axis=0)
            S_aug = np.concatenate([S, S[nonzero]], axis=0)
            if len(qG_aug) == 0:
                return qG_aug, S_aug
            from scipy.spatial import KDTree
            kdtree = KDTree(qG_aug)
            keep = np.ones(len(qG_aug), dtype=bool)
            for i in range(len(qG_aug)):
                if not keep[i]:
                    continue
                dups = kdtree.query_ball_point(qG_aug[i], r=dedup_tol)
                for j in dups:
                    if j > i:
                        keep[j] = False
            return qG_aug[keep], S_aug[keep]

        def _avg_deriv(qG, S, label):
            qG_sym, S_sym = _symmetrize_dedup(qG, S)
            derivs = self.curvature_along_lattice(
                qG_sym, S_sym, A_cols,
                n_nearest=max(deriv_order + 1, 3),
                tol=tol,
                output_order=deriv_order,
            )
            avg = float(np.mean(derivs))
            print(f"  [{label}] (symmetrized: {len(qG_sym)} pts): "
                  f"per-axis {deriv_order}-th deriv = {derivs}, avg = {avg}")
            return avg, derivs

        deriv_full, derivs_full = _avg_deriv(qG_full, SqG, label='all points')
        deriv_dropped, derivs_dropped = _avg_deriv(
            qG_full[keep_mask], SqG[keep_mask], label='closest along recip removed'
        )

        delta_deriv = deriv_full - deriv_dropped
        delta_per_axis = derivs_full - derivs_dropped

        print(f"  avg {deriv_order}-th deriv (all points)             : {deriv_full}")
        print(f"  avg {deriv_order}-th deriv (closest along b_i dropped): {deriv_dropped}")
        print(f"  delta avg {deriv_order}-th deriv (all - dropped)    : {delta_deriv}")
        print(f"  per-axis delta {deriv_order}-th deriv               : {delta_per_axis}")

        self.origin_error_deriv_order = deriv_order
        self.origin_error_delta_deriv = delta_deriv
        self.origin_error_deriv_full = deriv_full
        self.origin_error_deriv_dropped = deriv_dropped

        return delta_deriv, deriv_full, deriv_dropped

    def curvature_at_origin(self, fvals, coords, method='fornberg', fit_with_coul=True,
                            term='direct', dG0=False, degree=2):
        if term == 'direct':
            direct = True
            exchange = False
        elif term == 'exchange':
            direct = False
            exchange = True
        else:
            raise ValueError(f"Invalid term: {term}")

        if method == 'paraboloid_fit':
            f_curvature = model_function.PolynomialModel(orders=[degree])
            fit_multipliers = [10000.0]
            fit_method = MP2ScipyLeastSquares(f_curvature, fit_with_coul=fit_with_coul)
            fit_method.initial_guess = np.array([-1.0e-4])
            f_curvature.set_parameters(fit_method.initial_guess)
            closest_points = np.argsort(np.linalg.norm(coords, axis=1))[:7]
            coords_closest = coords[closest_points]
            fvals_closest = fvals[closest_points]
            fitted_params = fit_method.fit_model(coords_closest, fvals_closest,
                                                  fit_multipliers=fit_multipliers,
                                                  force_positive_params=False)
            c2_poly = -fitted_params[0]

        elif method == 'fornberg':
            closest_points = np.argsort(np.linalg.norm(coords, axis=1))[:4]
            if np.any(np.isclose(np.linalg.norm(coords[closest_points], axis=1), 0.0)):
                closest_points = closest_points[1:]
            coords_closest = coords[closest_points]
            A = coords_closest.T

            vector_multiples = np.arange(-self.curvature_n_nearest // 2,
                                          (self.curvature_n_nearest - 1) // 2 + 1)
            A_multiples = []
            for i in range(A.shape[1]):
                ai = A[:, i]
                ai_multiples = ai[:, None] * vector_multiples[None, :]
                A_multiples.append(ai_multiples.T)
            A_multiples = np.array(A_multiples).reshape(-1, 3)

            from scipy.spatial import KDTree
            tol = 1e-8
            if len(A_multiples) == 0:
                coords_curvature_samples = np.empty((0, 3))
            else:
                kdtree = KDTree(A_multiples)
                indices = kdtree.query_ball_point(A_multiples, r=tol)
                unique_mask = np.full(len(A_multiples), True)
                for i, inds in enumerate(indices):
                    for j in inds:
                        if j > i:
                            unique_mask[j] = False
                coords_curvature_samples = A_multiples[unique_mask]

            result_dict = self.mp2_structure_factor.build_structure_factor(
                qG_full=coords_curvature_samples,
                update_class=False, qG_cutoff=np.inf,
                direct=True, exchange=False, dG0=dG0)
            if direct:
                SqG_curvature_samples = (result_dict['SqG_full_dG0'] if dG0
                                         else result_dict['SqG_full_direct'])
            elif exchange:
                SqG_curvature_samples = result_dict['SqG_full_exchange']
            else:
                raise ValueError("direct or exchange must be True")

            if dG0:
                denominator = np.linalg.norm(coords_curvature_samples, axis=1) ** 4
                zero_idxs = np.isclose(denominator, 0.0)
                denominator[denominator < 1e-8] = np.inf
                SqG_curvature_samples = SqG_curvature_samples / denominator
                SqG_curvature_samples = SqG_curvature_samples[~zero_idxs]
                coords_curvature_samples = coords_curvature_samples[~zero_idxs]
                degree = 0
                print("Setting degree to 0 for DG0 curvature fit")

            curvatures = self.curvature_along_lattice(
                coords_curvature_samples, SqG_curvature_samples, A,
                n_nearest=self.curvature_n_nearest, output_order=degree)
            print(f"{degree}th derivative at origin: {curvatures}")
            c2_poly = np.mean(curvatures) / scipy.special.factorial(degree)
        else:
            raise ValueError(f"Invalid curvature method: {method}")

        return c2_poly

    @staticmethod
    def curvature_along_lattice(xyz, fvals, A, n_nearest=15, tol=1e-6, output_order=2):
        """Compute the n-th derivative of f(xyz) at the origin along each column of A."""
        from numdifftools.fornberg import fd_weights

        curvatures = np.zeros(A.shape[1])
        A = np.array(A)

        for i in range(A.shape[1]):
            ai = A[:, i]
            ai_unit = ai / np.linalg.norm(ai)
            proj = xyz @ ai_unit
            perp_dist = np.linalg.norm(xyz - np.outer(proj, ai_unit), axis=1)
            mask = perp_dist < tol
            idx = np.where(mask)[0]
            if len(idx) < n_nearest:
                print(f"NOTE: number of points ({len(idx)}) along lattice vector {i} is less than n_nearest ({n_nearest})")
            proj_sel = proj[idx]
            f_sel = fvals[idx]
            order = np.argsort(proj_sel)
            proj_sel = proj_sel[order]
            f_sel = f_sel[order]
            weights = fd_weights(proj_sel, x0=0, n=output_order)
            curvatures[i] = np.dot(weights, f_sel)

        return curvatures
