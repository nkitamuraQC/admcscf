import jax
from pyscfad import gto, scf, fci, ao2mo
from orbital_grad import CASSCF

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
        ["C",  (0.000000,  0.000000,  0.000000)]
        ["O",  (0.000000,  0.000000,  1.200000)]
        ["H",  (0.000000,  0.947700, -0.634900)]
        ["H",  (0.000000, -0.947700, -0.634900)]
    ]
    mol.basis = "sto-3g"
    mol.build()


    e_fci = fci_energy(mol)
    casscf = CASSCF(mol)
    e = casscf.opt_mo()
    assert(abs(e_fci - e) < 1e-6)

    casscf.opt_ci(mol)
    return
        
    
