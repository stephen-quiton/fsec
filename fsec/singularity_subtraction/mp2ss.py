from dataclasses import dataclass, replace
from pyscf.pbc.tools import get_monkhorst_pack_size
import scipy.special
import time
from fsec.singularity_subtraction import model_function
from fsec.singularity_subtraction.function_fitting import ExxScipyMinimize, ExxScipyLeastSquares, MP2ScipyMinimize, MP2ScipyLeastSquares
from fsec.singularity_subtraction.structure_factor import MP2StructureFactor
from fsec.singularity_subtraction.structure_factor.helpers_sf import build_uKpts as _build_uKpts
from fsec.singularity_subtraction.grids import MP2SSGrids
from fsec.singularity_subtraction import SingularitySubtraction
from pyscf.pbc import df
from pyscf import lib
import numpy as np


def convert_t2_to_kikjq_format(t2,kGrid1,qGrid,cell,kGrid2=None):
    """
    Convert t2 from kikjka format to kikjq format.
    """
    from scipy.spatial import KDTree
    from fsec.singularity_subtraction.grids import minimum_image
    nkpts = kGrid1.shape[0]
    if kGrid2 is None:
        kGrid2 = kGrid1
    q_tree = scipy.spatial.KDTree(qGrid)
    # t2 = self.t2#.copy()
    # t2 = np.zeros_like(self.t2)
    
    # Precompute mapping from (ki, ka) -> qi (index in qGrid) only once per ki
    ki_ka_qi_map = []  # each element is a list of size nkpts for fixed ki: maps ka -> qi
    for ki in range(nkpts):
        qpt_list = kGrid2 - kGrid1[ki]
        qpt_list = minimum_image(cell, qpt_list)
        # qpt_list shape: (nkpts, 3)
        # Perform batch query
        _, qis = q_tree.query(qpt_list, distance_upper_bound=1e-8)
        if np.any(qis == len(qGrid)):
            raise TypeError("Cannot locate q in the kmesh.")
        ki_ka_qi_map.append(qis)
        
    # Now, for each (ki, kj), build q_indices as a mapping from qi -> ka
    for ki in range(nkpts):
        qis = ki_ka_qi_map[ki]
        for kj in range(nkpts):
            q_indices = [None] * nkpts
            for ka in range(nkpts):
                qi = qis[ka]
                q_indices[qi] = ka
            assert sorted(q_indices) == list(range(nkpts)), f"Not a permutation: {q_indices}"

            tmp = t2[ki, kj, :].copy()
            tmp = tmp[q_indices]
            t2[ki, kj, :] = tmp


@dataclass(frozen=True)
class MP2SSOptions:
    """User-facing optional controls for MP2SS."""
    auxfunc_direct: str = 'XNGauss'
    auxfunc_direct_q2: object = 'XNGaussStackedSingularity'
    auxfunc_direct_dG0: object = 'XNGauss'
    auxfunc_exchange: str = 'XNGaussStackedSingularity'
    fit_with_coul: bool = True
    truncated_qG_grid: bool = True
    qG_grid_type: str = 'truncated'
    initial_guess: object = None
    N_local: object = None
    qG_norm_cutoff: object = 4.0
    min_points: int = 6
    sq_inversion_symm: bool = True
    sq_ke_cutoff: object = None
    fit_method: object = 'scipy_least_squares'
    fit_with_coul_dG0: bool = False
    fit_with_coul_q2: bool = False
    line_sampling: bool = False
    t2_store_type: str = 'kikjka'
    correct_q2_q4_separately: bool = True


@dataclass(frozen=True)
class DirectFullCorrectionConfig:
    """Configuration bundle for full MP2 direct correction."""
    cell: object
    auxfunc_direct: str
    fit_class: object
    fit_method: object
    fit_with_coul: bool
    qG_norm_cutoff: object


@dataclass(frozen=True)
class DirectSecondOrderCorrectionConfig:
    """Configuration bundle for second-order MP2 direct correction."""
    cell: object
    auxfunc_direct_q2: str
    fit_class: object
    fit_method: object
    fit_with_coul_q2: bool
    qG_norm_cutoff: object


@dataclass(frozen=True)
class DirectFourthOrderCorrectionConfig:
    """Configuration bundle for fourth-order MP2 direct correction."""
    cell: object
    auxfunc_direct_dG0: str
    fit_class: object
    fit_method: object
    fit_with_coul_dG0: bool
    qG_norm_cutoff: object


@dataclass
class DirectCorrectionResult:
    """Structured outputs from direct-correction evaluation."""
    total_direct_correction: float
    direct_integral_term: float
    direct_quadrature_term: float
    direct_quadrature_term_q2: object = None
    direct_integral_term_q2: object = None
    direct_total_correction_q2: object = None
    direct_quadrature_term_dG0: object = None
    direct_integral_term_dG0: object = None
    direct_total_correction_dG0: object = None
    direct_total_correction_q2_dG0: object = None


@dataclass
class ExchangeCorrectionConfig:
    """Configuration bundle for MP2ExchangeCorrection."""
    auxfunc_exchange: str
    fit_class: object
    fit_with_coul: bool
    qG_norm_cutoff: object
    fit_multipliers: object = None


@dataclass
class ExchangeCorrectionResult:
    """Structured outputs from exchange-correction evaluation."""
    total_exchange_correction: float
    exchange_integral_term: float
    exchange_quadrature_term: float


@dataclass(frozen=True)
class StructureFactorSamplerConfig:
    """Configuration bundle for MP2StructureFactorSampler."""
    kmf: object
    kmp: object
    cell: object
    nks: object
    t2: object
    sq_ke_cutoff: object
    qG_norm_cutoff: object
    sq_inversion_symm: bool
    t2_store_type: str
    min_points: int
    N_local: object


@dataclass(frozen=True)
class StructureFactorSamplerDeps:
    """Stable collaborators for MP2StructureFactorSampler."""
    pass


@dataclass
class StructureFactorSamplerResult:
    """Structured outputs from structure-factor sampling."""
    mp2_structure_factor: object
    qG_full: object = None

def build_uKpts(kmf, kpts, mo_coeff_kpts, NsCell=None, rptGrid3D=None, nbands=None):
    return _build_uKpts(kmf, kpts, mo_coeff_kpts, NsCell=NsCell, rptGrid3D=rptGrid3D, nbands=nbands)

# ---------------------------------------------------------------------------
# Model spec registries for MP2SS
# Each dict maps a user-facing string name to the metadata needed to
# instantiate the correct ModelFunction subclass.
#
# Construction types:
#   'simple'  — XNGeneral subclass: cls(parameters=..., negative=True, deg=D)
#   'qmesh'   — QMesh variant: cls(qGrid, cell, deltaGs) + compute_sum_g_q_deltaG()
#   'exchange' — exchange variant: cls(parameters=..., q2s=..., dvol=...)
# ---------------------------------------------------------------------------

DIRECT_MODEL_SPECS = {
    'XNGaussStackedSingularity': {
        'cls_name': 'XNGaussStackedSingularityQMesh',
        'type': 'qmesh',
        'c0_scale': 1.0e-4,
        'fit_multipliers': [1e4, 1.0],
        'plot_cls_name': 'XNGaussStackedSingularity',
    },
    'XNExpAbs': {
        'cls_name': 'XNExpAbsStackedSingularityQMesh',
        'type': 'qmesh',
        'c0_scale': 1.0e-4,
        'fit_multipliers': [1e4, 1.0],
        'plot_cls_name': 'XNExpAbsStackedSingularity',
    },
    'XNExponential': {
        'cls_name': 'XNExponential',
        'type': 'simple',
        'initial_params': np.array([1e-4, 1.0, 1.0]),
        'fit_multipliers': [1e4, 1.0, 1.0],
    },
    'XNQuarticExponential': {
        'cls_name': 'XNQuarticExponential',
        'type': 'simple',
        'initial_params': np.array([1e-4, 1.0, 1.0, 1.0]),
        'fit_multipliers': [1e4, 1.0, 1.0, 1.0],
    },
    'XNGauss': {
        'cls_name': 'XNGauss',
        'type': 'simple',
        'initial_params': np.array([1.0, 1.0]),
        'fit_multipliers': None,
        'deg': 2,
    },
}

DG0_MODEL_SPECS = {
    'XNGauss': {
        'cls_name': 'XNGauss',
        'initial_params': np.array([1e-4, 1.0]),
        'fit_multipliers': [1e4, 1.0],
    },
    'XNGaussStackedSingularity': {
        'cls_name': 'XNGauss',
        'initial_params': np.array([1e-4, 1.0]),
        'fit_multipliers': [1e4, 1.0],
    },
    'XNExponential': {
        'cls_name': 'XNExponential',
        'initial_params': np.array([1e-4, 1.0, 1.0]),
        'fit_multipliers': [1e4, 1.0, 1.0],
    },
    'XNQuarticExponential': {
        'cls_name': 'XNQuarticExponential',
        'initial_params': np.array([1e-4, 1.0, 1.0, 1.0]),
        'fit_multipliers': [1e4, 1.0, 1.0, 1.0],
    },
    'XNExpAbs': {
        'cls_name': 'XNExpAbs',
        'initial_params': np.array([1e-4, 1.0]),
        'fit_multipliers': [1e4, 1.0],
    },
    'XNExpAbs2': {
        'cls_name': 'XNExpAbs2',
        'initial_params': np.array([1e-4, 1.0, 1.0]),
        'fit_multipliers': [1e4, 1.0, 1.0],
    },
}

Q2_MODEL_SPECS = {
    'XNExpAbs': {
        'cls_name': 'XNExpAbsStackedSingularityQMesh',
        'type': 'qmesh',
        'c0_scale': 1.0e-5,
        'fit_multipliers': [1e4, 1.0],
    },
    'XNGauss': {
        'cls_name': 'XNGauss',
        'type': 'simple',
        'initial_params': np.array([1.0, 1.0]),
        'fit_multipliers': None,
        'deg': 2,
    },
    'XNGaussStackedSingularity': {
        'cls_name': 'XNGaussStackedSingularityQMesh',
        'type': 'qmesh',
        'c0_scale': 1.0e-5,
        'fit_multipliers': [1e4, 1.0],
    },
}

EXCHANGE_MODEL_SPECS = {
    'XNExponentialExchange': {
        'cls_name': 'XNExponentialStackedSingularityExchange',
        'initial_params': np.array([1e-4, 1.0, 1.0]),
    },
    'XNQuarticExponentialExchange': {
        'cls_name': 'XNQuarticExponentialStackedSingularityExchange',
        'initial_params': np.array([1e-4, 1.0, 1.0, 1.0]),
    },
    'XNGaussStackedSingularityExchange': {
        'cls_name': 'XNGaussStackedSingularityExchange',
        'initial_params': np.array([1e-4, 1.0]),
    },
    'XNExpAbsStackedSingularityExchange': {
        'cls_name': 'XNExpAbsStackedSingularityExchange',
        'initial_params': np.array([1e-4, 1.0]),
    },
    'XNExpAbs2StackedSingularityExchange': {
        'cls_name': 'XNExpAbs2StackedSingularityExchange',
        'initial_params': np.array([1e-4, 1.0, 1.0]),
    },
    'XNExpAbsStackedSingularity': {
        'cls_name': 'XNExpAbsStackedSingularityExchange',
        'initial_params': np.array([1e-4, 1.0]),
    },
}
class MP2StructureFactorSampler:
    """Structure-factor builder with explicit configuration and inputs."""

    def __init__(self, config: StructureFactorSamplerConfig, deps: StructureFactorSamplerDeps):
        self.config = config
        self.deps = deps

    def set_structure_factor(self, *, direct=True, exchange=True, dG0=False,
                             line_sampling=False):
        config = self.config

        mp2_structure_factor = MP2StructureFactor(
            config.kmf, config.kmp, t2=config.t2, N_local=config.N_local,
            sq_ke_cutoff=config.sq_ke_cutoff, qG_cutoff=config.qG_norm_cutoff,
            sq_inversion_symm=config.sq_inversion_symm,
            t2_store_type=config.t2_store_type,
        )
        mp2_structure_factor.set_grids(min_fit_points=config.min_points)

        qGrid = mp2_structure_factor.grids.qGrid
        kpts = config.kmp.kpts
        cell = config.cell
        if config.t2_store_type == 'kikjka':
            convert_t2_to_kikjq_format(mp2_structure_factor.t2, kpts, qGrid, cell)

        qG_full = None
        if line_sampling:
            qG_full = mp2_structure_factor.grids.build_qG_line_sampling()

        result = StructureFactorSamplerResult(mp2_structure_factor=mp2_structure_factor, qG_full=qG_full)

        mp2_structure_factor.build_structure_factor(
            qG_full=qG_full, direct=direct, exchange=exchange, dG0=dG0,
        )

        if config.t2_store_type == 'kikjka':
            convert_t2_to_kikjq_format(mp2_structure_factor.t2, kpts, qGrid, cell)
        return result


class MP2DirectFourthOrderSS(SingularitySubtraction):
    results_title = "Direct Term (4th order)"

    def __init__(self, config: DirectFourthOrderCorrectionConfig):
        self.config = config

    def _create_dg0_model(self, name):
        spec = DG0_MODEL_SPECS[name]
        cls = model_function.ModelFunction.get_class(spec['cls_name'])
        initial_params = spec['initial_params'].copy()
        m = cls(parameters=initial_params, negative=True,
                deg=spec.get('deg', 4))
        return m, initial_params, spec.get('fit_multipliers')

    def optimize_parameters(self, *, SqG_full_dG0, qG_full, grids, nks):
        config = self.config

        Lvec_recip = config.cell.reciprocal_vectors()
        numKpt3D = np.prod(nks)
        omega_star = abs(np.linalg.det(Lvec_recip))
        plot_prefactor = numKpt3D / omega_star

        if config.fit_method is None or config.fit_method == 'Disabled':
            return None

        qGlocal_fit = qG_full
        qGlocal_grid_correction = grids.build_qG_grid(grids.qGrid, grids.GptGrid3D, faster_dim='G')

        jac = '2-point'
        force_positive_params = True
        fixed_params = None
        f_dG0, initial_params, fit_multipliers = self._create_dg0_model(config.auxfunc_direct_dG0)


        fit_method_dG0 = config.fit_class(f_dG0, fit_with_coul=config.fit_with_coul_dG0)
        fit_method_dG0.initial_guess = initial_params
        fit_method_dG0.coul_deg = 4 if config.fit_with_coul_dG0 else None
        if config.qG_norm_cutoff is not None:
            mask = np.linalg.norm(qGlocal_fit, axis=1) <= config.qG_norm_cutoff
            print(f"Fitting with {len(qGlocal_fit[mask])} points")
            print(f"qG_norm_cutoff: {config.qG_norm_cutoff}")
            SqG_vals = SqG_full_dG0[mask]
            qGlocal_fit_vals = qGlocal_fit[mask]

            fitted_params_dG0 = fit_method_dG0.fit_model(
                qGlocal_fit_vals, SqG_vals, fit_multipliers=fit_multipliers,
                fixed_params=fixed_params, force_positive_params=force_positive_params,
                jac=jac, x_scale='jac', max_nfev=1000 * f_dG0.num_params
            )
        else:
            fitted_params_dG0 = fit_method_dG0.fit_model(
                qGlocal_fit, SqG_full_dG0, fit_multipliers=fit_multipliers,
                fixed_params=fixed_params, force_positive_params=force_positive_params,
                jac=jac, x_scale='jac', max_nfev=1000 * f_dG0.num_params
            )

        f_dG0.set_parameters(fitted_params_dG0)
        print("Unnormalized dG0 c4_value: ", f_dG0.c0)
        print("Normalized dG0 c4_value: ", f_dG0.c0 * plot_prefactor)


        return f_dG0, qGlocal_grid_correction, numKpt3D, omega_star

    def compute_quadrature_term(self, *, f_dG0, qGlocal_grid_correction):
        prefac_dG0 = 4 * np.pi * (4 * np.pi)
        denominator = np.linalg.norm(qGlocal_grid_correction, axis=1) ** 2
        denominator[denominator < 1e-8] = np.inf
        return prefac_dG0 * np.sum(f_dG0.eval_model(qGlocal_grid_correction) / denominator ** 2)

    def compute_integral_term(self, *, f_dG0, numKpt3D, omega_star):
        prefactor_dG0 = 4 * np.pi * (4 * np.pi) * numKpt3D / omega_star
        return prefactor_dG0 * f_dG0.coulomb_integral(coul_deg=4)

    def compute_correction(self, *, SqG_full_dG0=None, qG_full=None, grids=None, nks=None):
        if SqG_full_dG0 is None:
            raise ValueError("SqG_full_dG0 must be provided explicitly")
        if qG_full is None:
            raise ValueError("qG_full must be provided explicitly")
        if grids is None:
            raise ValueError("grids must be provided explicitly")
        if nks is None:
            raise ValueError("nks must be provided explicitly")

        optimized = self.optimize_parameters(
            SqG_full_dG0=SqG_full_dG0,
            qG_full=qG_full,
            grids=grids,
            nks=nks,
        )
        if optimized is None:
            return None

        f_dG0, qGlocal_grid_correction, numKpt3D, omega_star = optimized
        quadrature_term_dG0 = self.compute_quadrature_term(
            f_dG0=f_dG0,
            qGlocal_grid_correction=qGlocal_grid_correction,
        )
        integral_term_dG0 = self.compute_integral_term(
            f_dG0=f_dG0,
            numKpt3D=numKpt3D,
            omega_star=omega_star,
        )
        total_direct_correction_dG0 = -quadrature_term_dG0 + integral_term_dG0

        self.quadrature_term = quadrature_term_dG0
        self.integral_term = integral_term_dG0
        self.correction = total_direct_correction_dG0

        print(f"dG0 quadrature term: {quadrature_term_dG0}")
        print(f"dG0 integral term: {integral_term_dG0}")
        print(f"dG0 total correction: {total_direct_correction_dG0}")
        return quadrature_term_dG0, integral_term_dG0, total_direct_correction_dG0


class MP2DirectSecondOrderSS(SingularitySubtraction):
    results_title = "Direct Term (2nd order)"

    def __init__(self, config: DirectSecondOrderCorrectionConfig):
        self.config = config

    def _create_q2_model(self, name, grids):
        if name not in Q2_MODEL_SPECS:
            name = 'XNGaussStackedSingularity'
        spec = Q2_MODEL_SPECS[name]
        cls = model_function.ModelFunction.get_class(spec['cls_name'])

        if spec['type'] == 'qmesh':
            m = cls(qGrid=grids.qGrid, cell=grids.cell,
                    deltaGs=grids.GptGrid3D, remove_deltaG_zero=True)
            m.compute_sum_g_q_deltaG()
            initial_params = np.array([spec['c0_scale'] / m.g0, 1.0])
        else:
            initial_params = spec['initial_params'].copy()
            m = cls(parameters=initial_params, negative=True,
                    deg=spec.get('deg', 2))

        return m, initial_params, spec.get('fit_multipliers')

    def optimize_parameters(self, *, SqG_full_q2_part, qG_full, grids, nks):
        config = self.config

        Lvec_recip = config.cell.reciprocal_vectors()
        numKpt3D = np.prod(nks)
        omega_star = abs(np.linalg.det(Lvec_recip))

        if config.fit_method is None or config.fit_method == 'Disabled':
            return None

        qGlocal_fit = qG_full
        qGlocal_grid_correction = grids.build_qG_grid(grids.qGrid, grids.GptGrid3D, faster_dim='G')

        f_q2, initial_params, fit_multipliers = self._create_q2_model(
            config.auxfunc_direct_q2, grids
        )

        fit_method = config.fit_class(f_q2, fit_with_coul=config.fit_with_coul_q2)
        fit_method.initial_guess = initial_params
        fixed_params = None
        if config.qG_norm_cutoff is not None:
            mask = np.linalg.norm(qGlocal_fit, axis=1) <= config.qG_norm_cutoff
        else:
            mask = np.ones_like(SqG_full_q2_part, dtype=bool)
        print(f"Fitting with {len(qGlocal_fit[mask])} points")
        print(f"qG_norm_cutoff: {config.qG_norm_cutoff}")
        fitted_params_q2 = fit_method.fit_model(
            qGlocal_fit[mask], SqG_full_q2_part[mask], fit_multipliers=fit_multipliers, fixed_params=fixed_params,
            max_nfev=1000 * f_q2.num_params, force_positive_params=True
        )
        f_q2.set_parameters(fitted_params_q2)

        return f_q2, qGlocal_grid_correction, numKpt3D, omega_star

    def compute_quadrature_term(self, *, f_q2, qGlocal_grid_correction):
        prefac = 4 * np.pi
        denominator = np.linalg.norm(qGlocal_grid_correction, axis=1) ** 2
        denominator[denominator < 1e-8] = np.inf
        return prefac * np.sum(f_q2.eval_model(qGlocal_grid_correction) / denominator)

    def compute_integral_term(self, *, f_q2, numKpt3D, omega_star):
        prefactor = 4 * np.pi * numKpt3D / omega_star
        return prefactor * f_q2.coulomb_integral()

    def compute_correction(self, *, SqG_full_q2_part=None, qG_full=None, grids=None, nks=None):
        if SqG_full_q2_part is None:
            raise ValueError("SqG_full_q2_part must be provided explicitly")
        if qG_full is None:
            raise ValueError("qG_full must be provided explicitly")
        if grids is None:
            raise ValueError("grids must be provided explicitly")
        if nks is None:
            raise ValueError("nks must be provided explicitly")

        optimized = self.optimize_parameters(
            SqG_full_q2_part=SqG_full_q2_part,
            qG_full=qG_full,
            grids=grids,
            nks=nks,
        )
        if optimized is None:
            return None

        f_q2, qGlocal_grid_correction, numKpt3D, omega_star = optimized
        quadrature_term_q2 = self.compute_quadrature_term(
            f_q2=f_q2,
            qGlocal_grid_correction=qGlocal_grid_correction,
        )
        integral_term_q2 = self.compute_integral_term(
            f_q2=f_q2,
            numKpt3D=numKpt3D,
            omega_star=omega_star,
        )
        total_direct_correction_q2 = -quadrature_term_q2 + integral_term_q2

        self.quadrature_term = quadrature_term_q2
        self.integral_term = integral_term_q2
        self.correction = total_direct_correction_q2

        print(f"q2 quadrature term: {quadrature_term_q2}")
        print(f"q2 integral term: {integral_term_q2}")
        print(f"q2 total correction: {total_direct_correction_q2}")
        return quadrature_term_q2, integral_term_q2, total_direct_correction_q2

class MP2DirectFullSS(SingularitySubtraction):
    """Direct-correction calculator with explicit configuration and dependencies."""

    results_title = "Direct Term"

    def __init__(self, config: DirectFullCorrectionConfig):
        self.config = config

    def _create_direct_model(self, name, grids):
        spec = DIRECT_MODEL_SPECS[name]
        cls = model_function.ModelFunction.get_class(spec['cls_name'])

        if spec['type'] == 'qmesh':
            m = cls(qGrid=grids.qGrid, cell=grids.cell,
                    deltaGs=grids.GptGrid3D)
            m.compute_sum_g_q_deltaG()
            initial_params = np.array([spec['c0_scale'] / m.g0, 1.0])
        else:
            initial_params = spec['initial_params'].copy()
            m = cls(parameters=initial_params, negative=True,
                    deg=spec.get('deg', 2))

        return m, initial_params, spec.get('fit_multipliers')

    def optimize_parameters(self, *, SqG_full_direct, qG_full, grids, nks):
        config = self.config

        Lvec_recip = config.cell.reciprocal_vectors()
        numKpt3D = np.prod(nks)
        omega_star = abs(np.linalg.det(Lvec_recip))

        if config.fit_method is None or config.fit_method == 'Disabled':
            return None

        qGlocal_fit = qG_full
        qGlocal_grid_correction = grids.build_qG_grid(grids.qGrid, grids.GptGrid3D, faster_dim='G')
        f_gauss, initial_params, fit_multipliers = self._create_direct_model(
            config.auxfunc_direct, grids
        )

        print("Using direct auxiliary function: ", f_gauss.__class__.__name__)
        f_gauss.set_parameters(initial_params)
        print("initial_params: ", initial_params)
        fit_method = config.fit_class(f_gauss, fit_with_coul=config.fit_with_coul)
        fit_method.initial_guess = initial_params

        fixed_params = None
        if config.qG_norm_cutoff is not None:
            mask = np.linalg.norm(qGlocal_fit, axis=1) <= config.qG_norm_cutoff
            print(f"Fitting with {len(qGlocal_fit[mask])} points")
            print(f"qG_norm_cutoff: {config.qG_norm_cutoff}")
        else:
            mask = np.ones_like(SqG_full_direct, dtype=bool)

        fitted_params = fit_method.fit_model(
            qGlocal_fit[mask], SqG_full_direct[mask],
            fit_multipliers=fit_multipliers,
            fixed_params=fixed_params,
            force_positive_params=True,
            max_nfev=1000 * f_gauss.num_params,
        )

        f_gauss.set_parameters(fitted_params)
        print(f"Direct auxiliary function parameters: {fitted_params}")
        return f_gauss, qGlocal_grid_correction, numKpt3D, omega_star

    def compute_quadrature_term(self, *, f_gauss, qGlocal_grid_correction):
        prefac = 4 * np.pi
        denominator = np.linalg.norm(qGlocal_grid_correction, axis=1) ** 2
        denominator[denominator < 1e-8] = np.inf
        return prefac * np.sum(f_gauss.eval_model(qGlocal_grid_correction) / denominator)

    def compute_integral_term(self, *, f_gauss, numKpt3D, omega_star):
        prefactor = 4 * np.pi * numKpt3D / omega_star
        return prefactor * f_gauss.coulomb_integral()

    def compute_correction(self, *, SqG_full_direct=None, qG_full=None, grids=None, nks=None):
        print("MP2SS: Computing direct correction...")
        ss_start = time.time()
        if SqG_full_direct is None:
            raise ValueError("SqG_full_direct must be provided explicitly")
        if qG_full is None:
            raise ValueError("qG_full must be provided explicitly")
        if grids is None:
            raise ValueError("grids must be provided explicitly")
        if nks is None:
            raise ValueError("nks must be provided explicitly")

        denominator = np.linalg.norm(qG_full, axis=1) ** 2
        denominator[denominator < 1e-8] = np.inf
        Emp2_d_ref = 4 * np.pi * np.sum(SqG_full_direct / denominator)
        print(f"Emp2_d_ref: {Emp2_d_ref}")

        optimized = self.optimize_parameters(
            SqG_full_direct=SqG_full_direct,
            qG_full=qG_full,
            grids=grids,
            nks=nks,
        )
        if optimized is None:
            return None

        f_gauss, qGlocal_grid_correction, numKpt3D, omega_star = optimized

        quadrature_term = self.compute_quadrature_term(
            f_gauss=f_gauss,
            qGlocal_grid_correction=qGlocal_grid_correction,
        )
        integral_term = self.compute_integral_term(
            f_gauss=f_gauss,
            numKpt3D=numKpt3D,
            omega_star=omega_star,
        )
        total_direct_correction = -quadrature_term + integral_term

        self.quadrature_term = quadrature_term
        self.integral_term = integral_term
        self.correction = total_direct_correction

        result = DirectCorrectionResult(
            total_direct_correction=total_direct_correction,
            direct_integral_term=integral_term,
            direct_quadrature_term=quadrature_term,
        )

        print(f"MP2SS: Direct correction computed in %.2f seconds" % (time.time() - ss_start))

        return result

    def compute_direct_correction(self, *, SqG_full_direct, qG_full, SqG_full_dG0, grids, nks):
        return self.compute_correction(
            SqG_full_direct=SqG_full_direct,
            qG_full=qG_full,
            grids=grids,
            nks=nks,
        )


MP2DirectSS = MP2DirectFullSS


class MP2ExchangeSS(SingularitySubtraction):
    """Exchange-correction calculator with explicit configuration and dependencies."""

    results_title = "Exchange Term"

    def __init__(self, config: ExchangeCorrectionConfig):
        self.config = config

    def _create_exchange_model(self, name, q2grid, dvol):
        spec = EXCHANGE_MODEL_SPECS[name]
        cls = model_function.ModelFunction.get_class(spec['cls_name'])
        initial_params = spec['initial_params'].copy()
        m = cls(parameters=initial_params, q2s=q2grid, dvol=dvol)
        return m, initial_params, spec.get('fit_multipliers')

    def optimize_parameters(self, *, SqG_full_exchange, qG_full, grids, nks):
        config = self.config

        Lvec_recip = grids.cell.reciprocal_vectors()
        numKpt3D = np.prod(nks)
        omega_star = abs(np.linalg.det(Lvec_recip))

        denominator = np.linalg.norm(qG_full, axis=1) ** 2
        denominator[denominator < 1e-8] = np.inf
        Emp2_x_ref = 4 * np.pi * np.sum(SqG_full_exchange / denominator)
        print(f"Emp2_x_ref: {Emp2_x_ref}")

        if config.fit_class is None:
            return None

        qGlocal_grid_correction = grids.build_qG_grid(grids.qGrid, grids.GptGrid3D, faster_dim='G')
        dvol = abs(np.linalg.det(Lvec_recip)) / numKpt3D
        q2grid = qGlocal_grid_correction
        dvol_x = dvol

        f_gauss, initial_params, fit_multipliers = self._create_exchange_model(
            config.auxfunc_exchange, q2grid, dvol_x
        )
        if config.fit_multipliers is not None:
            fit_multipliers = config.fit_multipliers

        f_gauss.set_parameters(initial_params)
        print("initial_params: ", initial_params)
        fit_method = config.fit_class(f_gauss, fit_with_coul=config.fit_with_coul)
        fit_method.initial_guess = initial_params

        fixed_params = None
        if config.qG_norm_cutoff is not None:
            mask = np.linalg.norm(qG_full, axis=1) <= config.qG_norm_cutoff
            print(f"Fitting with {len(qG_full[mask])} points")
            print(f"qG_norm_cutoff: {config.qG_norm_cutoff}")
            fitted_params = fit_method.fit_model(
                qG_full[mask], SqG_full_exchange[mask],
                fit_multipliers=fit_multipliers,
                fixed_params=fixed_params,
                force_positive_params=True,
                max_nfev=1000 * f_gauss.num_params,
            )
        else:
            fitted_params = fit_method.fit_model(
                qG_full, SqG_full_exchange,
                fit_multipliers=fit_multipliers,
                fixed_params=fixed_params,
                force_positive_params=True,
                max_nfev=1000 * f_gauss.num_params,
            )

        f_gauss.set_parameters(fitted_params)
        print(f"Exchange auxiliary function parameters: {fitted_params}")

        return f_gauss, qGlocal_grid_correction, numKpt3D, omega_star

    def compute_quadrature_term(self, *, f_gauss, qGlocal_grid_correction):
        prefac = 4 * np.pi
        denominator = np.linalg.norm(qGlocal_grid_correction, axis=1) ** 2
        denominator[denominator < 1e-8] = np.inf
        return prefac * np.sum(f_gauss.eval_model(qGlocal_grid_correction) / denominator)

    def compute_integral_term(self, *, f_gauss, numKpt3D, omega_star):
        prefactor = 4 * np.pi * numKpt3D / omega_star
        return prefactor * f_gauss.coulomb_integral()

    def compute_correction(self, *, SqG_full_exchange=None, qG_full=None, grids=None, nks=None):
        print("MP2SS: Computing exchange correction...")
        ss_start = time.time()
        if SqG_full_exchange is None:
            raise ValueError("SqG_full_exchange must be provided explicitly")
        if qG_full is None:
            raise ValueError("qG_full must be provided explicitly")
        if grids is None:
            raise ValueError("grids must be provided explicitly")
        if nks is None:
            raise ValueError("nks must be provided explicitly")

        optimized = self.optimize_parameters(
            SqG_full_exchange=SqG_full_exchange,
            qG_full=qG_full,
            grids=grids,
            nks=nks,
        )
        if optimized is None:
            return None

        f_gauss, qGlocal_grid_correction, numKpt3D, omega_star = optimized
        quadrature_term = self.compute_quadrature_term(
            f_gauss=f_gauss,
            qGlocal_grid_correction=qGlocal_grid_correction,
        )
        integral_term = self.compute_integral_term(
            f_gauss=f_gauss,
            numKpt3D=numKpt3D,
            omega_star=omega_star,
        )
        total_exchange_correction = -quadrature_term + integral_term

        self.quadrature_term = quadrature_term
        self.integral_term = integral_term
        self.correction = total_exchange_correction

        ss_end = time.time()
        print(f"MP2SS: Exchange correction computed in %.2f seconds" % (ss_end - ss_start))
        return ExchangeCorrectionResult(
            total_exchange_correction=total_exchange_correction,
            exchange_integral_term=integral_term,
            exchange_quadrature_term=quadrature_term,
        )

    def compute_exchange_correction(self, *, SqG_full_exchange=None, qG_full=None, grids=None, nks=None):
        return self.compute_correction(
            SqG_full_exchange=SqG_full_exchange,
            qG_full=qG_full,
            grids=grids,
            nks=nks,
        )


class MP2SS:
    """
    A class for performing singularity subtraction to correct the finite size error in
    MP2 correlation energy (both direct and exchange terms)

    Key attributes:
        kmf (object): PySCF Mean-field object. Should have converged SCF calculation.
        cell (object): The periodic cell object from the mean-field calculation.
        nocc (int): Number of occupied orbitals
        auxfunc_direct (model_function.ModelFunction): Auxiliary function for direct correction
        auxfunc_exchange (model_function.ModelFunction): Auxiliary function for exchange correction
        fit_method (callable): Fitting method for the auxiliary functions.
    """
    fit_method_dict = {
        'scipy_minimize': ExxScipyMinimize,
        'scipy_least_squares': ExxScipyLeastSquares
    }

    def __init__(self, kmf, kmp, auxfunc_direct=None, auxfunc_exchange=None, t2=None,
                 options=None, **kwargs):
        """
        Initialize the MP2SS class.
        Parameters:
            kmf (object): Kohn-Sham mean-field object containing the computational cell and related properties.
            t2 (array): MP2 amplitudes from KMP2.kernel(). Default is None (will be computed if not provided).
            auxfunc_direct (model_function.ModelFunction): Auxiliary model function used for direct correction fitting.
            auxfunc_exchange (model_function.ModelFunction): Auxiliary model function used for exchange correction fitting.
            options (MP2SSOptions or dict, optional): User-facing MP2SS options. When omitted, defaults are created
                by MP2SSOptions. Keyword arguments are still accepted for compatibility and override options.
        """
        if 'correct_q4_q2_separately' in kwargs:
            if 'correct_q2_q4_separately' in kwargs:
                raise TypeError(
                    "Use only one of correct_q4_q2_separately or "
                    "correct_q2_q4_separately"
                )
            kwargs['correct_q2_q4_separately'] = kwargs.pop('correct_q4_q2_separately')

        if auxfunc_exchange == 'XNGaussStackedSingularity':
            auxfunc_exchange = 'XNGaussStackedSingularityExchange'

        if options is None:
            options = MP2SSOptions(**kwargs)
        else:
            if isinstance(options, dict):
                options = MP2SSOptions(**options)
            elif not isinstance(options, MP2SSOptions):
                raise TypeError("options must be an MP2SSOptions instance, dict, or None")
            if kwargs:
                options = replace(options, **kwargs)

        if auxfunc_direct is None:
            auxfunc_direct = options.auxfunc_direct
        if auxfunc_exchange is None:
            auxfunc_exchange = options.auxfunc_exchange
        if auxfunc_exchange == 'XNGaussStackedSingularity':
            auxfunc_exchange = 'XNGaussStackedSingularityExchange'

        self.kmf = kmf
        self.kmp = kmp
        self.cell = kmf.cell
        self.nks = get_monkhorst_pack_size(self.cell, kmf.kpts)
        self.t2 = t2
        self.options = options
        self.auxfunc_direct = auxfunc_direct
        self.auxfunc_exchange = auxfunc_exchange
        
        if isinstance(self.kmp._scf.with_df, df.df.GDF):
            self.with_df_ints = True
        else:
            self.with_df_ints = False

        
        self.auxfunc_direct_dG0 = options.auxfunc_direct_dG0 or self.auxfunc_direct
        self.auxfunc_direct_q2 = options.auxfunc_direct_q2 or self.auxfunc_direct

        if self.auxfunc_direct not in DIRECT_MODEL_SPECS:
            raise ValueError(f"Invalid auxiliary function: {self.auxfunc_direct}. "
                             f"Must be one of: {sorted(DIRECT_MODEL_SPECS.keys())}")
        if self.auxfunc_exchange not in EXCHANGE_MODEL_SPECS:
            raise ValueError(f"Invalid auxiliary function: {self.auxfunc_exchange}. "
                             f"Must be one of: {sorted(EXCHANGE_MODEL_SPECS.keys())}")

        # Fit Settings (same as ExxSS)
        self.fit_with_coul = options.fit_with_coul
        truncated_qG_grid = options.truncated_qG_grid
        self.qG_grid_type = options.qG_grid_type
        if self.qG_grid_type == 'local' or self.qG_grid_type == 'full':
            assert truncated_qG_grid is False, "Requesting local or full qG grid, need truncated_qG_grid False"

        self.initial_guess = options.initial_guess
        self.N_local = options.N_local if options.N_local is not None else self.cell.mesh

        # Structure factor build settings
        self.qG_norm_cutoff = options.qG_norm_cutoff
        self.min_points = options.min_points
        self.sq_inversion_symm = options.sq_inversion_symm
        self.sq_ke_cutoff = options.sq_ke_cutoff
        if self.sq_ke_cutoff is not None:
            print("sq_ke_cutoff provided to MP2SS, overriding N_local")
            self.N_local = self.cell.cutoff_to_mesh(self.sq_ke_cutoff)
            print("Using N_local = ", self.N_local)
        self.fit_method = options.fit_method
        self.fit_class = None
        if self.fit_method == 'scipy_least_squares':
            self.fit_class = MP2ScipyLeastSquares
        elif self.fit_method == 'scipy_minimize':
            self.fit_class = MP2ScipyMinimize
        elif self.fit_method == None or self.fit_method == 'Disabled':
            pass # Disable fitting
        else:  
            raise ValueError(f"Invalid fit method: {self.fit_method}")
        self.fit_with_coul_dG0 = options.fit_with_coul_dG0
        self.fit_with_coul_q2 = options.fit_with_coul_q2
        self.line_sampling = options.line_sampling
        self.t2_store_type = options.t2_store_type # 'kikjka', 'kikj', or 'ki'
        
        
        self.correct_q2_q4_separately = options.correct_q2_q4_separately
        self.dG0 = self.correct_q2_q4_separately
        self.correct_q2_part = self.correct_q2_q4_separately
        
        self.mp2_structure_factor = None
        self.grids = None
        self.quadrature_term_direct = None
        self.quadrature_term_exchange = None
        self.integral_term_direct = None
        self.integral_term_exchange = None

        self.exchange_correction = MP2ExchangeSS(
            self._build_exchange_correction_config(),
        )
        self._refresh_direct_correction()



    def __str__(self):
        return f"MP2SS: {self.__class__.__name__} with direct auxfunc {self.auxfunc_direct.model} and exchange auxfunc {self.auxfunc_exchange.model}"


    def set_grids(self):
        self.grids = MP2SSGrids(
            cell=self.cell,
            kGrid1=self.kmf.kpts,
            N_local=self.N_local,
            qG_norm_cutoff=self.qG_norm_cutoff,
            min_points=self.min_points,
        )
        self.grids.build_grids()

    def _build_direct_full_correction_config(self):
        return DirectFullCorrectionConfig(
            cell=self.cell,
            auxfunc_direct=self.auxfunc_direct,
            fit_class=self.fit_class,
            fit_method=self.fit_method,
            fit_with_coul=self.fit_with_coul,
            qG_norm_cutoff=self.qG_norm_cutoff,
        )

    def _build_direct_second_order_correction_config(self):
        return DirectSecondOrderCorrectionConfig(
            cell=self.cell,
            auxfunc_direct_q2=self.auxfunc_direct_q2,
            fit_class=self.fit_class,
            fit_method=self.fit_method,
            fit_with_coul_q2=self.fit_with_coul_q2,
            qG_norm_cutoff=self.qG_norm_cutoff,
        )

    def _build_direct_fourth_order_correction_config(self):
        return DirectFourthOrderCorrectionConfig(
            cell=self.cell,
            auxfunc_direct_dG0=self.auxfunc_direct_dG0,
            fit_class=self.fit_class,
            fit_method=self.fit_method,
            fit_with_coul_dG0=self.fit_with_coul_dG0,
            qG_norm_cutoff=self.qG_norm_cutoff,
        )

    def _refresh_direct_correction(self):
        if self.correct_q2_q4_separately:
            self.direct_second_order_correction = MP2DirectSecondOrderSS(
                self._build_direct_second_order_correction_config(),
            )
            self.direct_fourth_order_correction = MP2DirectFourthOrderSS(
                self._build_direct_fourth_order_correction_config(),
            )
        else:
            self.direct_correction = MP2DirectSS(
                self._build_direct_full_correction_config(),
            )

    def _build_exchange_correction_config(self):
        return ExchangeCorrectionConfig(
            auxfunc_exchange=self.auxfunc_exchange,
            fit_class=self.fit_class,
            fit_with_coul=self.fit_with_coul,
            qG_norm_cutoff=self.qG_norm_cutoff,
        )

    def _refresh_exchange_correction(self):
        self.exchange_correction = MP2ExchangeSS(
            self._build_exchange_correction_config(),
        )


    def set_fit_method(self, method):
        print("MP2SS singularity subtraction initial_guess", self.initial_guess)
        # Create separate fit methods for direct and exchange
        self.fit_method_direct = MP2SS.fit_method_dict[method](self.auxfunc_direct, fit_with_coul=self.fit_with_coul,
                                                               is_contraction=self.auxfunc_direct.is_contraction,
                                                               initial_guess=self.initial_guess)
        self.fit_method_exchange = MP2SS.fit_method_dict[method](self.auxfunc_exchange,
                                                                fit_with_coul=self.fit_with_coul,
                                                                is_contraction=self.auxfunc_exchange.is_contraction,
                                                                initial_guess=self.initial_guess)
        if (
            hasattr(self, 'direct_correction')
            or hasattr(self, 'direct_second_order_correction')
            or hasattr(self, 'direct_fourth_order_correction')
        ):
            self._refresh_direct_correction()
        if hasattr(self, 'exchange_correction'):
            self._refresh_exchange_correction()
        return self.fit_method_direct  # Return one for compatibility
    

    def set_structure_factor(self, direct=True, exchange=True, dG0=False,
                             line_sampling=False):
        mp2_structure_factor = MP2StructureFactor(
            self.kmf, self.kmp, t2=self.t2, N_local=self.N_local,
            sq_ke_cutoff=self.sq_ke_cutoff, qG_cutoff=self.qG_norm_cutoff,
            sq_inversion_symm=self.sq_inversion_symm,
            t2_store_type=self.t2_store_type,
        )
        mp2_structure_factor.set_grids(min_fit_points=self.min_points)

        qGrid = mp2_structure_factor.grids.qGrid
        kpts = self.kmp.kpts
        if self.t2_store_type == 'kikjka':
            convert_t2_to_kikjq_format(mp2_structure_factor.t2, kpts, qGrid, self.cell)

        qG_full = None
        if line_sampling:
            qG_full = mp2_structure_factor.grids.build_qG_line_sampling()

        mp2_structure_factor.build_structure_factor(
            qG_full=qG_full, direct=direct, exchange=exchange, dG0=dG0,
        )

        if self.t2_store_type == 'kikjka':
            convert_t2_to_kikjq_format(mp2_structure_factor.t2, kpts, qGrid, self.cell)

        self.mp2_structure_factor = mp2_structure_factor
        return self.mp2_structure_factor
    

    def compute_direct_correction(self,SqG_full_direct=None,qG_full=None):
        if SqG_full_direct is None:
            SqG_full_direct = self.mp2_structure_factor.SqG_full_direct
        if qG_full is None:
            qG_full = self.mp2_structure_factor.qG_full
        SqG_full_dG0 = self.mp2_structure_factor.SqG_full_dG0 if self.dG0 else None

        self.direct_integral_term_q2 = None
        self.direct_quadrature_term_q2 = None
        self.direct_total_correction_q2 = None
        self.direct_integral_term_dG0 = None
        self.direct_quadrature_term_dG0 = None
        self.direct_total_correction_dG0 = None
        self.direct_total_correction_q2_dG0 = None

        if self.correct_q2_q4_separately:
            # Extract the deltaG=0 part as the fourth order O(q^4) portion. Correct q2 and q4 separately.

            if SqG_full_dG0 is None:
                raise ValueError("SqG_full_dG0 must be available when correct_q2_q4_separately=True")
            denominator = np.linalg.norm(qG_full, axis=1) ** 2
            denominator[denominator < 1e-8] = np.inf
            SqG_full_q2_part = SqG_full_direct - (4 * np.pi) * SqG_full_dG0 / denominator

            second_order_config = self._build_direct_second_order_correction_config()
            fourth_order_config = self._build_direct_fourth_order_correction_config()
            full_config = self._build_direct_full_correction_config()

            self.direct_second_order_correction = MP2DirectSecondOrderSS(second_order_config)
            q2_result = self.direct_second_order_correction.compute_correction(
                SqG_full_q2_part=SqG_full_q2_part,
                qG_full=qG_full,
                grids=self.grids,
                nks=self.nks,
            )
            self.direct_fourth_order_correction = MP2DirectFourthOrderSS(fourth_order_config)
            dG0_result = self.direct_fourth_order_correction.compute_correction(
                SqG_full_dG0=SqG_full_dG0,
                qG_full=qG_full,
                grids=self.grids,
                nks=self.nks,
            )
            if q2_result is None or dG0_result is None:
                return None

            quadrature_term_q2, integral_term_q2, total_direct_correction_q2 = q2_result
            quadrature_term_dG0, integral_term_dG0, total_direct_correction_dG0 = dG0_result
            self.direct_quadrature_term_q2 = quadrature_term_q2
            self.direct_integral_term_q2 = integral_term_q2
            self.direct_total_correction_q2 = total_direct_correction_q2
            self.direct_quadrature_term_dG0 = quadrature_term_dG0
            self.direct_integral_term_dG0 = integral_term_dG0
            self.direct_total_correction_dG0 = total_direct_correction_dG0

            self.direct_quadrature_term = quadrature_term_q2 + quadrature_term_dG0
            self.direct_integral_term = integral_term_q2 + integral_term_dG0
            self.direct_total_correction_q2_dG0 = (
                total_direct_correction_q2 + total_direct_correction_dG0
            )

            self.direct_correction = MP2DirectFullSS(full_config)
            self.direct_correction.results_title = "Direct Term (Full: 2nd + 4th)"
            self.direct_correction.quadrature_term = self.direct_quadrature_term
            self.direct_correction.integral_term = self.direct_integral_term
            self.direct_correction.correction = self.direct_total_correction_q2_dG0
            return self.direct_total_correction_q2_dG0
        else:

            self.direct_correction = MP2DirectFullSS(
                self._build_direct_full_correction_config(),
            )
            result = self.direct_correction.compute_direct_correction(
                SqG_full_direct=SqG_full_direct,
                qG_full=qG_full,
                SqG_full_dG0=SqG_full_dG0,
                grids=self.grids,
                nks=self.nks,
            )
            if result is None:
                return None

            self.direct_integral_term = result.direct_integral_term
            self.direct_quadrature_term = result.direct_quadrature_term
            return result.total_direct_correction

    def compute_exchange_correction(self, SqG_full_exchange=None, qG_full=None):
        if SqG_full_exchange is None:
            SqG_full_exchange = self.mp2_structure_factor.SqG_full_exchange
        if qG_full is None:
            qG_full = self.mp2_structure_factor.qG_full

        result = self.exchange_correction.compute_correction(
            SqG_full_exchange=SqG_full_exchange,
            qG_full=qG_full,
            grids=self.grids,
            nks=self.nks,
        )
        if result is None:
            return None

        self.exchange_integral_term = result.exchange_integral_term
        self.exchange_quadrature_term = result.exchange_quadrature_term
        return result.total_exchange_correction



    def compute_correction(self,direct=True,exchange=True):
        """
        Compute the direct MP2 correction using singularity subtraction.
        This implements the direct correction logic from the MATLAB mp2_ss_direct function.
        """
        print("Computing MP2SS correction...")
        ss_start = time.time()

        # Build grids if not already built
        if self.grids is None:
            self.set_grids()

        self.set_structure_factor(direct=direct, exchange=exchange, dG0=self.dG0,
                                  line_sampling=self.line_sampling)

        SqG_full_direct = self.mp2_structure_factor.SqG_full_direct if direct else None
        SqG_full_exchange = self.mp2_structure_factor.SqG_full_exchange if exchange else None
        qG_full = self.mp2_structure_factor.qG_full

        e_corr_ss = self.kmp.e_corr_ss
        e_corr_os = self.kmp.e_corr_os
        self.edi_uncorr = e_corr_os * 2.0
        self.exi_uncorr = e_corr_ss - e_corr_os
        self.emp2_uncorr = self.kmp.e_corr

        mp2ss_direct_correction = self.compute_direct_correction(
            SqG_full_direct=SqG_full_direct,
            qG_full=qG_full,
        )
        mp2ss_exchange_correction = self.compute_exchange_correction(
            SqG_full_exchange=SqG_full_exchange,
            qG_full=qG_full,
        )

        total_correction = mp2ss_direct_correction + mp2ss_exchange_correction
        self.mp2ss_total_correction = total_correction
        self.mp2ss_direct_correction = mp2ss_direct_correction
        self.mp2ss_exchange_correction = mp2ss_exchange_correction
        self.emp2ss_direct = self.edi_uncorr + mp2ss_direct_correction
        self.emp2ss_exchange = self.exi_uncorr + mp2ss_exchange_correction
        self.emp2ss = self.emp2ss_direct + self.emp2ss_exchange

        self.print_results()

        total_correction = lib.tag_array(
            total_correction,
            mp2ss_direct_correction=mp2ss_direct_correction,
            mp2ss_exchange_correction=mp2ss_exchange_correction,
        )
        print('Total time for MP2SS: %.2f seconds' % (time.time() - ss_start))
        return total_correction
    
    def compute_quadrature_term(self):
        """
        Compute the total quadrature term (sum of direct and exchange).
        """
        if self.quadrature_term_direct is None or self.quadrature_term_exchange is None:
            self.compute_direct_correction()
            self.compute_exchange_correction()

        total_quadrature = self.quadrature_term_direct + self.quadrature_term_exchange
        print(f"Total quadrature term: {total_quadrature}")
        return total_quadrature

    def compute_integral_term(self):
        """
        Compute the total integral term (sum of direct and exchange).
        """
        if self.integral_term_direct is None or self.integral_term_exchange is None:
            self.compute_direct_correction()
            self.compute_exchange_correction()

        total_integral = self.integral_term_direct + self.integral_term_exchange
        print(f"Total integral term: {total_integral}")
        return total_integral

    def print_results(self):
        print(f"=== Results for {self.__class__.__name__} ===")
        print(f"Using with_df_ints: {self.with_df_ints}")
        print(f"Using auxiliary function: {self.auxfunc_direct} for direct correction")
        print(f"Using auxiliary function: {self.auxfunc_exchange} for exchange correction")

        print("Uncorrected MP2 Energies:")
        print(f" MP2 uncorrected direct energy (hartree)   = {self.edi_uncorr}")
        print(f" MP2 uncorrected exchange energy (hartree) = {self.exi_uncorr}")
        print(f" MP2 uncorrected total energy (hartree)    = {self.emp2_uncorr}")
        print()

        if self.correct_q2_q4_separately:
            self.direct_second_order_correction.print_results()
            self.direct_fourth_order_correction.print_results()
            self.direct_correction.print_results()
        else:
            self.direct_correction.print_results()
        self.exchange_correction.print_results()

        print("Final Energies:")
        print(f" MP2SS direct final energy (hartree)        = {self.emp2ss_direct}")
        print(f" MP2SS exchange final energy (hartree)      = {self.emp2ss_exchange}")
        print(f" MP2SS total final energy (hartree)         = {self.emp2ss}")
