import jax
from pyscfad import gto, scf, fci, ao2mo

# from pyscf.mcscf.mc1step import expmat
from pyscfad.fci.fci_slow import absorb_h1e, contract_2e, fci_ovlp, kernel
from pyscfad import pytree
import jax.numpy as jnp
import numpy as np
from math import factorial
from pyscfad.tools import rotate_mo1
from scipy.optimize import minimize
from jax import value_and_grad

# molecular structure
mol = gto.Mole()
mol.verbose = 0
mol.output = None#"out_h2o"
mol.atom = [
    ['H', ( 6., 0.    ,0.   )],
    ['H', ( 5., 0.    ,0.   )],
    ['H', ( 4., 0.    ,0.   )],
    ['H', ( 3., 0.    ,0.   )],
    ['H', ( 2., 0.    ,0.   )],
    ['H', ( 1., 0.    ,0.   )],
]
mol.basis = {'H': 'sto-3g',
             'O': '6-31g',}
mol.build()

mf = scf.RHF(mol)
mf.kernel()


def fci_energy(mol, nroots=1):
    mf = scf.RHF(mol)
    mf.kernel()
    e, fcivec = fci.solve_fci(mf, nroots=nroots)
    return e


class CASSCF_Energy:
    def __init__(self, mf, x=None, ncas=None, nelecas=None):
        self.x = x
        self.mol = mf.mol
        self.nmo = mf.mo_coeff.shape[1]
        if self.x is None:
            nao = self.mol.nao
            assert nao == self.nmo
            size = nao * (nao - 1) // 2
            self.x = np.zeros((size,))
        self.mo = rotate_mo1(mf.mo_coeff, self.x)
        mo = self.mo
        self.mf = mf
        self.mol = mf.mol
        self.norb = mf.mo_coeff.shape[0]
        norb = self.norb
        self.nelec = mf.mol.nelectron
        hcore = mf.get_hcore()
        self.int1e = jnp.einsum("ia,jb,ij->ab", mo, mo, hcore)
        self.int2e = ao2mo.incore.full(mf._eri, mo)
        self.r0 = jnp.zeros((norb, norb))
        
        self.ncas = ncas
        self.nelecas = nelecas
        self.ncore = self.mf.mol.nelectron // 2 - nelecas // 2

    def expmat(self, r0, nterms=10):
        exp_r = jnp.identity(r0.shape[0])
        save = jnp.identity(r0.shape[0])
        exp_r = save
        for i in range(1, nterms):
            save = jnp.dot(save, r0)
            exp_r += save / factorial(i)
        return exp_r

    def casscf_energy2(self, r0):
        u = self.expmat(-r0)
        # mo = jnp.dot(self.mo, u)
        # u = r0
        int1e = jnp.einsum("ia,jb,ij->ab", u, u, self.int1e)
        int2e = jnp.einsum("ia,jb,kc,ld,ijkl->abcd", u, u, u, u, self.int2e)
        e, ci = kernel(int1e, int2e, self.norb, self.nelec)
        # hci = contract_2e(int2e, ci, self.norb, self.nelec)
        # e = fci_ovlp(self.mol, self.mol, ci, hci, self.norb, self.norb, self.nelec, self.nelec, mo, mo)
        # e = jnp.dot(ci, hci)
        self.e_tot = e + self.mol.energy_nuc()
        return e

    def casscf_energy3(self):
        e, ci = kernel(self.int1e, self.int2e, self.norb, self.nelec)
        self.e_tot = e #+ self.mol.energy_nuc()
        return e

    def get_cas(self):
        if mo_coeff is None: mo_coeff = self.mf.mo_coeff
        if ncas is None: ncas = self.ncas
        if ncore is None: ncore = casci.ncore
        mo_core = mo_coeff[:,:ncore]
        mo_cas = mo_coeff[:,ncore:ncore+ncas]

        hcore = self.mf.get_hcore()
        energy_core = self.mf.mol.energy_nuc()
        if mo_core.size == 0:
            corevhf = 0
        else:
            core_dm = numpy.dot(mo_core, mo_core.conj().T) * 2
            corevhf = casci.get_veff(casci.mol, core_dm)
            energy_core += numpy.einsum('ij,ji', core_dm, hcore).real
            energy_core += numpy.einsum('ij,ji', core_dm, corevhf).real * .5
        h1eff = reduce(numpy.dot, (mo_cas.conj().T, hcore+corevhf, mo_cas))
        h2eff = self.int2e[ncore:ncore+ncas, ncore:ncore+ncas, ncore:ncore+ncas, ncore:ncore+ncas]
        return h1eff, h2eff, energy_core


# jac = jax.jacrev(fci_energy)(mol)
# print(f'Nuclaer gradient:\n{jac.coords}')
# print(f'Gradient wrt basis exponents:\n{jac.exp}')
# print(f'Gradient wrt basis contraction coefficients:\n{jac.ctr_coeff}')

# jac = jax.jacrev(casscf_energy)(inp)
# print(f'kappa gradient:\n{jac.r0}')


#ce = CASSCF_Energy(mf)
#nmo = mf.mo_coeff.shape[0]
#jac_h = jax.jacobian(ce.casscf_energy2)
#x_value = np.array([[0.0, -0.01], [0.01, 0.0]])
#print(jac_h(x_value))


def func(x0, mf):
    def energy(x0, mf):
        myci = CASSCF_Energy(mf, x=jnp.asarray(x0))
        e = myci.casscf_energy3()
        return myci.e_tot

    def grad(x0, mf):
        f, g = value_and_grad(energy)(x0, mf)
        return f, g

    f, g = grad(x0, mf)
    return (jnp.array(f), jnp.array(g))


nao = mol.nao
size = nao * (nao - 1) // 2
x0 = np.zeros((size,))
options = {"gtol": 1e-6}
res = minimize(func, x0, args=(mf,), jac=True, method="BFGS", options=options)
e = func(res.x, mf)[0]
print(e)
