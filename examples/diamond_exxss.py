#!/usr/bin/env python
# Copyright 2026 Stephen Quiton. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""
Exact-exchange singularity subtraction for diamond.

This example runs PBE0 and HF calculations on a 2x2x2 k-point mesh, then
applies the contracted-Gaussian and quartic-exponential ExxSS models.

Method reference:
    S. J. Quiton, J. D. F. Pottecher, X. Xing, M. Head-Gordon, and L. Lin,
    J. Chem. Theory Comput. 21, 8863–8875 (2025).
    https://doi.org/10.1021/acs.jctc.5c01066
"""

from pyscf.pbc import df, dft, gto, scf
from pyscf.lib import logger

from fsec.singularity_subtraction import ExxSS, ExxSSQuarticExponential


def build_diamond_cell():
    cell = gto.Cell()
    cell.unit = "Bohr"
    cell.atom = """
        C 0.0 0.0 0.0
        C 1.68516327271508 1.68516327271508 1.68516327271508
    """
    cell.a = """
        0.0 3.370326545430162 3.370326545430162
        3.370326545430162 0.0 3.370326545430162
        3.370326545430162 3.370326545430162 0.0
    """
    cell.basis = "gth-szv"
    cell.pseudo = "gth-pbe"
    cell.ke_cutoff = 100
    cell.precision = 1e-8
    cell.max_memory = 4000
    cell.verbose = 4
    cell.build()
    return cell


cell = build_diamond_cell()
kpts = cell.make_kpts(
    [2, 2, 2], wrap_around=True, with_gamma_point=True
)

# PBE0 with Gaussian density fitting.
kmf_pbe0 = dft.KRKS(cell, kpts)
kmf_pbe0.verbose = logger.DEBUG
kmf_pbe0.xc = "PBE0"
kmf_pbe0.exxdiv = "ewald"
kmf_pbe0.with_df = df.GDF(cell, kpts).build()
kmf_pbe0.kernel()

# The default ExxSS model is a contracted Gaussian.
exxss_pbe0 = ExxSS(
    kmf_pbe0,
    fit_method="scipy_least_squares",
    fit_with_coul=True,
)
exxss_pbe0.compute_correction()
exxss_pbe0.print_results()
print("Corrected exact exchange (hartree) =", exxss_pbe0.Ek_ss)

# The quartic-exponential model is an alternative representation of the
# exchange structure factor near the origin.
exxss_pbe0_quartic = ExxSSQuarticExponential(
    kmf_pbe0,
    fit_method="scipy_least_squares",
    fit_with_coul=True,
    qG_norm_cutoff_sigma=2.0,
)
exxss_pbe0_quartic.compute_correction()
exxss_pbe0_quartic.print_results()
print("Corrected exact exchange (hartree) =", exxss_pbe0_quartic.Ek_ss)

# ExxSS can also be applied to exact exchange from a pure HF calculation.
kmf_hf = scf.KRHF(cell, kpts)
kmf_hf.exxdiv = "ewald"
kmf_hf.with_df = df.GDF(cell, kpts).build()
kmf_hf.kernel()

exxss_hf = ExxSS(
    kmf_hf,
    fit_method="scipy_least_squares",
    fit_with_coul=True,
)
exxss_hf.compute_correction()
exxss_hf.print_results()
print("Corrected exact exchange (hartree) =", exxss_hf.Ek_ss)

exxss_hf_quartic = ExxSSQuarticExponential(
    kmf_hf,
    fit_method="scipy_least_squares",
    fit_with_coul=True,
    qG_norm_cutoff_sigma=2.0,
)
exxss_hf_quartic.compute_correction()
exxss_hf_quartic.print_results()
print("Corrected exact exchange (hartree) =", exxss_hf_quartic.Ek_ss)
