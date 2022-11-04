import os
import subprocess
from elastic2D_tools import *
from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
from pymatgen.io.vasp.outputs import Vasprun

# example of operations()

def example_operations():
    # my own script to 'prepare VASP files' (pvf), which generates the INCAR, KPOINTS and POTCAR from a POSCAR file
    subprocess.run('~/bin/pvf.py -t gr -d 2 -k 0.06 -e 500 -s 0.05 -vdw D3 -lreal f -isif 2 -ediff 1E-6 -ftol 0.01', shell=True)
    # run VASP
    subprocess.run('mpirun -np 48 vasp_std', shell=True)
    # could optionally create and submit a cluster job script


def setup_elastic2D(structure_file, operations):
    """
    Sets up 2D elastic calculations using VASP.
    
    Args:
        structure_file (str): 'POSCAR' or 'CONTCAR' file corresponding to equilibrium structure
        operations (func)   : includes creation of VASP input files, running of VASP, or 
                              submission of job to queue
    """
    struc = Structure.from_file(structure_file)
    lattice = struc.lattice.matrix
    ds, def_lats = gen_strained_lats(lattice, f=0.02, N=7)

    os.mkdir('elastic')
    os.chdir('elastic')

    eps_dirs = ['eps_11', 'eps_22', 'eps_12']

    for i, eps_dir in enumerate(eps_dirs):
        os.mkdir(eps_dir)
        os.chdir(eps_dir)
        for j, lat in enumerate(def_lats[i]):
            os.mkdir('deformed_lattice_{}'.format(j + 1))
            os.chdir('deformed_lattice_{}'.format(j + 1))
            struc.lattice = Lattice(lat)
            struc.to(filename='POSCAR')

            operations()
            
            os.chdir('..')

        os.chdir('..')

    os.chdir('..')



def post_elastic2D(structure_file):

    struc = Structure.from_file(structure_file)
    lattice = struc.lattice.matrix
    ds, def_lats = gen_strained_lats(lattice, f=0.02, N=7)

    os.chdir('elastic')

    # collect stresses
    eps_dirs = ['eps_11', 'eps_22', 'eps_12']
    stresses = []
    for i, eps_dir in enumerate(eps_dirs):
        os.chdir(eps_dir)
        stress_eps = []
        for j, lat in enumerate(def_lats[i]):
            os.chdir('deformed_lattice_{}'.format(j + 1))
            stress = - np.array(Vasprun('vasprun.xml').ionic_steps[-1]['stress'])  # in kbar
            stress_eps.append(stress)
            os.chdir('..')
        stresses.append(stress_eps)
        os.chdir('..')
    os.chdir('..')
 
    # post-process elastic constants
    Cs_voigt, Cs_mandel, Cs_err = process_2D_elastic_constants(ds, stresses, c=lattice[2, 2] * 1e-2, symmetrize=True)
    stable, eigvals = check_elastic_stability(Cs_mandel)
    bm = planar_bulk_modulus(Cs_voigt)
 
    with open('elastic.dat', 'w') as f:
        f.write('{:^20} : {:^8}\n'.format('Stability', 'True' if stable else 'False'))
        f.write('{:^20} : {:^12.3f} {:^12.3f} {:^12.3f}\n\n'.format('Eigvals [N/m]', eigvals[0], eigvals[1], eigvals[2]))
        f.write('{:^20} : {:^12.3f}\n\n'.format('Bulk Modulus [N/m]', bm))
        f.write('Elastic Constants (Voigt) [N/m]\n')
        f.write('{:^12.3f} {:^12.3f} {:^12.3f}\n'.format(Cs_voigt[0, 0], Cs_voigt[0, 1], Cs_voigt[0, 2]))
        f.write('{:^12.3f} {:^12.3f} {:^12.3f}\n'.format(Cs_voigt[1, 0], Cs_voigt[1, 1], Cs_voigt[1, 2]))
        f.write('{:^12.3f} {:^12.3f} {:^12.3f}\n\n'.format(Cs_voigt[2, 0], Cs_voigt[2, 1], Cs_voigt[2, 2]))
        f.write('Elastic Constants (Mandel) [N/m]\n')
        f.write('{:^12.3f} {:^12.3f} {:^12.3f}\n'.format(Cs_mandel[0, 0], Cs_mandel[0, 1], Cs_mandel[0, 2]))
        f.write('{:^12.3f} {:^12.3f} {:^12.3f}\n'.format(Cs_mandel[1, 0], Cs_mandel[1, 1], Cs_mandel[1, 2]))
        f.write('{:^12.3f} {:^12.3f} {:^12.3f}\n\n'.format(Cs_mandel[2, 0], Cs_mandel[2, 1], Cs_mandel[2, 2]))
        f.write('Elastic Constants RMS Fitting Error [N/m]\n')
        f.write('{:^12.5g} {:^12.5g} {:^12.5g}\n'.format(Cs_err[0, 0], Cs_err[0, 1], Cs_err[0, 2]))
        f.write('{:^12.5g} {:^12.5g} {:^12.5g}\n'.format(Cs_err[1, 0], Cs_err[1, 1], Cs_err[1, 2]))
        f.write('{:^12.5g} {:^12.5g} {:^12.5g}\n\n'.format(Cs_err[2, 0], Cs_err[2, 1], Cs_err[2, 2]))
       

