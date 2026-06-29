import numpy as np
from pyscf.pbc.tools import get_monkhorst_pack_size

from fsec.singularity_subtraction.grids import minimum_image


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
