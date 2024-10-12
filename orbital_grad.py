import jax
from pyscfad import gto, scf, fci, ao2mo
from pyscfad.fci.fci_slow import absorb_h1e, contract_2e, fci_ovlp, kernel
from pyscfad import pytree
import jax.numpy as jnp
import numpy as np
from math import factorial
from pyscfad.tools import rotate_mo1
from scipy.optimize import minimize
from jax import value_and_grad
from functools import reduce




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
        int1e = jnp.einsum("ia,jb,ij->ab", u, u, self.int1e)
        int2e = jnp.einsum("ia,jb,kc,ld,ijkl->abcd", u, u, u, u, self.int2e)
        e, ci = kernel(int1e, int2e, self.norb, self.nelec)
        self.e_tot = e + self.mol.energy_nuc()
        return e

    def casscf_energy3(self, nroots=1):
        h1eff, h2eff, energy_core = self.get_cas()
        e, ci = kernel(h1eff, h2eff, self.norb, self.nelec, nroots=nroots)
        if nroots == 1:
            self.e_tot = e + energy_core
            return e + energy_core
        if nroots == 2:
            return e[1] - e[0]

    def get_cas(self):
        if mo_coeff is None: mo_coeff = self.mf.mo_coeff
        if ncas is None: ncas = self.ncas
        if ncore is None: ncore = self.ncore
        mo_core = mo_coeff[:,:ncore]
        mo_cas = mo_coeff[:,ncore:ncore+ncas]

        hcore = self.mf.get_hcore()
        energy_core = self.mf.mol.energy_nuc()
        if mo_core.size == 0:
            corevhf = 0
        else:
            core_dm = np.dot(mo_core, mo_core.conj().T) * 2
            corevhf = self.mf.get_veff(self.mol, core_dm)
            energy_core += np.einsum('ij,ji', core_dm, hcore).real
            energy_core += np.einsum('ij,ji', core_dm, corevhf).real * .5
        h1eff = reduce(np.dot, (mo_cas.conj().T, hcore+corevhf, mo_cas))
        h2eff = self.int2e[ncore:ncore+ncas, ncore:ncore+ncas, ncore:ncore+ncas, ncore:ncore+ncas]
        return h1eff, h2eff, energy_core
    

class CASSCF:
    def __init__(self, mol, ncas, nelecas):
        self.mol = mol
        self.mf = scf.RHF(mol)
        self.mf.kernel()
        self.ncas = ncas
        self.nelecas = nelecas
        

    def energy(self, x0, mf):
        myci = CASSCF_Energy(mf, x=jnp.asarray(x0), ncas=self.ncas, nelecas=self.nelecas)
        e = myci.casscf_energy3()
        return myci.e_tot
    
    def optimize_mo_fun(self, x0):
        f, g = value_and_grad(self.energy)(x0, self.mf)
        return (jnp.array(f), jnp.array(g))
    
    def opt_mo(self):
        nao = self.mf.mol.nao
        size = nao * (nao - 1) // 2
        x0 = np.zeros((size,))
        options = {"gtol": 1e-6}
        res = minimize(self.optimize_mo_fun, x0, args=(self.mf, ), jac=True, method="BFGS", options=options)
        e = self.optimize_mo_fun(res.x, self.mf)[0]
        self.x = res.x 
        return e
    
    def energy_ci(self, mol):
        nao = mol.nao
        size = nao * (nao - 1) // 2
        x0 = np.zeros((size,))
        mf = scf.RHF(mol)
        mf.kernel()
        myci = CASSCF_Energy(mf, x=jnp.asarray(x0), ncas=self.ncas, nelecas=self.nelecas)
        e = myci.casscf_energy3(nroots=2)
        return e

    def optimize_ci_fun(self, mol):
        f, g = value_and_grad(self.energy_ci)(mol)
        return (jnp.array(f), jnp.array(g))
    
    def opt_ci(self, mol):
        res = minimize(self.optimize_ci_fun, mol, jac=True, method="BFGS", options=options)
        e = self.optimize_ci_fun(res.mol, self.mf)[0]
        self.mol_opt = res.mol
        return e
    

