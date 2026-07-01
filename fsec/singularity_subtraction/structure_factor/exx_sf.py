import numpy as np
import scipy

from pyscf.lib import logger
from pyscf.pbc.tools import get_monkhorst_pack_size

from fsec.singularity_subtraction.grids import ExxSSGrids, minimum_image

from fsec.singularity_subtraction.structure_factor import StructureFactor
from fsec.singularity_subtraction.structure_factor.helpers_sf import build_uKpts, TimingProfile


class ExxStructureFactor(StructureFactor):
    def __init__(self, kmf, N_local=None, sq_ke_cutoff=None, qG_cutoff=None, relative_shift=0.0, **kwargs):
        """
        Initialize the exchange structure factor.
        """
        self.kmf = kmf
        self.dm_kpts1 = kwargs.get("dm_kpts1", kmf.make_rdm1())
        self.dm_kpts2 = kwargs.get("dm_kpts2", kmf.make_rdm1())
        self.mo_coeff_kpts1 = kwargs.get("mo_coeff_kpts1", kmf.mo_coeff_kpts)
        self.mo_coeff_kpts2 = kwargs.get("mo_coeff_kpts2", kmf.mo_coeff_kpts)
        self.kGrid1 = minimum_image(kmf.cell, kwargs.get("kGrid1", kmf.kpts))
        self.kGrid2 = kwargs.get("kGrid2", None)
        self.relative_shift = relative_shift
        self.min_points = kwargs.get("min_points", 6)

        self.debug_options = kwargs.get("debug_options", {})
        super().__init__(self.kmf.cell, N_local, sq_ke_cutoff, qG_cutoff, **kwargs)

    def set_grids(self, min_fit_points=6):
        self.min_points = min_fit_points
        self.grids = ExxSSGrids(
            self.kmf.cell,
            self.kGrid1,
            N_local=self.N_local,
            qG_norm_cutoff=self.qG_cutoff,
            min_points=self.min_points,
        )
        self.grids.build_grids()
        if self.qG_cutoff is None:
            self.grids.build_truncated_qG_grid()

    def build_structure_factor(self, verbose=None):
        kmf = self.kmf
        log = logger.new_logger(kmf, verbose)
        profile = TimingProfile()
        total_t0 = profile.start()

        phase_t0 = profile.start()
        self.set_grids(min_fit_points=self.min_points)
        profile.stop("grid construction", phase_t0)
        qG_full = self.grids.qG_grid_truncated
        kGrid1 = self.grids.kGrid1
        kGrid2 = self.grids.kGrid2
        rptGrid3D = self.grids.RptGrid3D_coarse

        nocc = kmf.cell.tot_electrons() // 2
        phase_t0 = profile.start()
        uKpts1 = build_uKpts(kmf, kGrid1, self.mo_coeff_kpts1, rptGrid3D=rptGrid3D, nbands=nocc)
        uKpts2 = build_uKpts(kmf, kGrid2, self.mo_coeff_kpts2, rptGrid3D=rptGrid3D, nbands=nocc)
        profile.stop("uKpts construction", phase_t0)

        phase_t0 = profile.start()
        NsCell = np.array(self.N_local)
        nks = get_monkhorst_pack_size(kmf.cell, kmf.kpts)
        nkpts = np.prod(nks)

        Lvec_real = kmf.cell.lattice_vectors()
        L_delta = Lvec_real / NsCell[:, None]
        dvol = np.abs(np.linalg.det(L_delta))

        nqG = qG_full.shape[0]
        nG = np.prod(NsCell)
        print("ExxStructureFactor nG: ", nG)
        print("ExxStructureFactor nqG: ", nqG)
        SqG_full = np.zeros(nqG, dtype=np.float64)
        print("SqG MEM USAGE (KB) IS: {:.3f}".format(SqG_full.nbytes / 1024))
        profile.stop("structure-factor setup", phase_t0)

        num_equiv_qG = 0

        loop_t0 = profile.start()
        for qG in range(qG_full.shape[0]):
            qGpt = qG_full[qG, :]

            region_t0 = profile.start()
            if self.sq_inversion_symm and qG > 1:
                qg_tree = scipy.spatial.KDTree(qG_full[:qG, :])
                _, equiv_qG_index = qg_tree.query(-qGpt, distance_upper_bound=1e-8)
                if equiv_qG_index != len(qG_full[:qG, :]):
                    num_equiv_qG += 1
                    SqG_full[qG] = SqG_full[equiv_qG_index]
                    profile.stop("inversion-symmetry reuse", region_t0)
                    continue
            profile.stop("inversion-symmetry lookup", region_t0)

            for k in range(nkpts):
                region_t0 = profile.start()
                temp_SqG_k = 0
                kpt1 = kGrid1[k, :]
                kpt2 = kpt1 + qGpt

                kpt2_BZ = minimum_image(kmf.cell, kpt2)
                idx_kpt2 = np.where(np.sum((kGrid2 - kpt2_BZ[None, :]) ** 2, axis=1) < 1e-8)[0]
                if len(idx_kpt2) != 1:
                    raise TypeError("Cannot locate (k+q) in the kmesh.")
                idx_kpt2 = idx_kpt2[0]
                kGdiff = kpt2 - kpt2_BZ
                profile.stop("k-point lookup", region_t0)

                region_t0 = profile.start()
                exp_term = np.exp(-1j * (rptGrid3D @ kGdiff))
                conj_u1 = np.conj(uKpts1[k, :, :])
                u2 = uKpts2[idx_kpt2] * exp_term[None, :]
                profile.stop("pair-density phase/orbital setup", region_t0)

                region_t0 = profile.start()
                rho12 = conj_u1 @ u2.T
                profile.stop("pair-density matrix multiply", region_t0)

                region_t0 = profile.start()
                rho12 = np.abs(rho12) ** 2
                temp_SqG_k = np.sum(rho12) * dvol**2

                SqG_full[qG] += temp_SqG_k / nkpts
                profile.stop("pair-density square/reduction", region_t0)

        profile.stop("main qG loop (inclusive)", loop_t0)
        log.note("Number of equivalent qG points: %d", num_equiv_qG)

        phase_t0 = profile.start()
        self.SqG_full = SqG_full
        self.SqG_truncated = SqG_full
        profile.stop("class result update", phase_t0)
        print("ExxStructureFactor: nocc = ", self.SqG_full[0])

        self.last_build_timings = profile.summary(total_t0)
        profile.log_summary(log, self.last_build_timings)
        return SqG_full
