from collections import defaultdict

import numpy as np
from pyscf.lib import logger
from pyscf.pbc.tools import get_monkhorst_pack_size

from fsec.singularity_subtraction.grids import minimum_image


class TimingProfile:
    """Accumulate PySCF CPU and wall-clock timings for repeated regions."""

    def __init__(self):
        self._times = defaultdict(lambda: [0.0, 0.0])

    @staticmethod
    def start():
        return logger.process_clock(), logger.perf_counter()

    def stop(self, label, start):
        cpu = logger.process_clock() - start[0]
        wall = logger.perf_counter() - start[1]
        self._times[label][0] += cpu
        self._times[label][1] += wall
        return cpu, wall

    def summary(self, total_start):
        total_cpu = logger.process_clock() - total_start[0]
        total_wall = logger.perf_counter() - total_start[1]
        summary = {
            label: {"cpu": values[0], "wall": values[1]}
            for label, values in self._times.items()
        }
        summary["total"] = {"cpu": total_cpu, "wall": total_wall}
        return summary

    @staticmethod
    def log_summary(log, summary):
        total = summary["total"]
        log.note(
            "build_structure_factor CPU %.2f sec, wall %.2f sec",
            total["cpu"], total["wall"],
        )
        for label, values in sorted(
                ((key, value) for key, value in summary.items() if key != "total"),
                key=lambda item: item[1]["wall"], reverse=True):
            cpu_fraction = 100.0 * values["cpu"] / total["cpu"] if total["cpu"] else 0.0
            wall_fraction = 100.0 * values["wall"] / total["wall"] if total["wall"] else 0.0
            log.note(
                "  %-36s CPU %9.2f sec (%5.1f%%), wall %9.2f sec (%5.1f%%)",
                label, values["cpu"], cpu_fraction,
                values["wall"], wall_fraction,
            )


def build_uKpts(kmf, kpts, mo_coeff_kpts, NsCell=None, rptGrid3D=None, nbands=None):
    # Setup constants
    NsCell = np.array(kmf.cell.mesh) if NsCell is None else NsCell
    nbands = kmf.cell.tot_electrons() // 2 if nbands is None else nbands
    nks = get_monkhorst_pack_size(kmf.cell, kpts)
    Nk = np.prod(nks)

    # Setup real space grid points
    if rptGrid3D is None:
        Lvec_real = kmf.cell.lattice_vectors()
        L_delta = Lvec_real / NsCell[:, None]
        xv, yv, zv = np.meshgrid(
            np.arange(NsCell[0]),
            np.arange(NsCell[1]),
            np.arange(NsCell[2]),
            indexing='ij',
        )
        mesh_idx = np.hstack([xv.reshape(-1, 1), yv.reshape(-1, 1), zv.reshape(-1, 1)])
        rptGrid3D = mesh_idx @ L_delta

    assert rptGrid3D.shape[1] == 3, "build_uKpts: rptGrid3D should be a 3D array"
    nG = rptGrid3D.shape[0]

    # Evaluate the atomic orbitals at the real space grid points
    kGrid = minimum_image(kmf.cell, kpts)
    aoval = kmf.cell.pbc_eval_gto("GTOval_sph", coords=rptGrid3D, kpts=kpts)

    # Compute uKpts
    exp_part = np.exp(-1j * (rptGrid3D @ kGrid.T)).T
    utmp = aoval @ np.array(mo_coeff_kpts)[:, :, :nbands]
    utmp = utmp.transpose(0, 2, 1)
    uKpts = exp_part[:, None, :] * utmp
    return uKpts
