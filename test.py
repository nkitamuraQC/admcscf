from pyscfad import gto, scf, fci, ao2mo
from ad.orbital_grad import CASSCF
import pytest
from jax import numpy as jnp
import numpy as np
from pyscf.mcscf import CASSCF as plainCASSCF
from pyscf import gto as plaingto

def fci_energy(mol, nroots=1):
    mf = scf.RHF(mol)
    mf.kernel()
    e, fcivec = fci.solve_fci(mf, nroots=nroots)
    return e

def test_casscf():
    # molecular structure
    mol = gto.Mole()
    mol.verbose = 0
    mol.output = None#"out_h2o"
    mol.atom = [
        ["H",  (0.000000,  0.000000,  0.000000)],
        ["H",  (0.000000,  0.000000,  1.200000)],
    ]
    mol.basis = "sto-3g"
    mol.build()


    e_fci = fci_energy(mol)
    casscf = CASSCF(mol, 2, 2)
    casscf.run_HF()
    e = casscf.opt_mo()
    assert(abs(e_fci - e) < 1e-6)
    assert(abs(-1.056740746305258 - e) < 1e-6)
    return

def test_casscf2():
    # molecular structure
    mol = gto.Mole()
    mol.verbose = 0
    mol.output = None#"out_h2o"
    mol.atom = [
        ["H",  (0.000000,  0.000000,  0.000000)],
        ["H",  (0.000000,  0.000000,  1.200000)],
        ["H",  (0.000000,  0.000000,  2.400000)],
        ["H",  (0.000000,  0.000000,  3.600000)],
    ]
    mol.basis = "sto-3g"
    mol.build()#

    e_fci = fci_energy(mol)
    casscf = CASSCF(mol, 2, 2)
    casscf.run_HF()
    e = casscf.opt_mo()
    print(e)
    assert(abs(-1.1354082498557863 - casscf.e_cas) < 1e-6)
    return

if __name__ == "__main__":
    test_casscf()
    test_casscf2()
