from sys import argv
import os
import subprocess
from vasp_elastic2D import *
from pymatgen.io.vasp.inputs import Incar

task = 'post'  # pre or post
mpi = 48
ecut = 520     # default plane wave cutoff is 520 eV
kspace = 0.03  # default kspace is 0.03 × 2π/Å
sigma = 0.05
ediff = '1E-8'
ftol = '0.002'
ibrion = 2
vdw_scheme = 'optB88'
frac = 0.01
N = 7
sym = False
q = 'q1'

for i in range(len(argv)):
    if argv[i] == '-t':
        task = argv[i + 1]
    if argv[i] == '-q':
        q = argv[i + 1]
    if argv[i] == '-e':
        ecut = argv[i + 1]
    if argv[i] == '-k':
        kspace = float(argv[i + 1])
    if argv[i] == '-s':
        sigma = argv[i + 1]
    if argv[i] == '-ediff':
        ediff = argv[i + 1]
    if argv[i] == '-ftol':
        ftol = argv[i + 1]
    if argv[i] == '-ibrion':
        ibrion = argv[i + 1]
    if argv[i] == '-vdw':
        vdw_scheme = argv[i + 1]
    if argv[i] == '-frac':
        frac = float(argv[i + 1])
    if argv[i] == '-N':
        N = int(argv[i + 1])
    if argv[i] == '-sym':
        sym = (argv[i + 1] == 'T')


def ops():
    # copy custom_incar.dat file if present
    if 'custom_incar.dat' in os.listdir('../../..'):
        subprocess.run('cp ../../../custom_incar.dat .', shell=True)

    # read INCAR settings from GGA-SCF for ENCUT and SIGMA
    incar = Incar.from_file('../../../INCAR').as_dict()
    ecut, sigma, ispin = incar['ENCUT'], incar['SIGMA'], incar['ISPIN']

    subprocess.run('cp POSCAR orig_POSCAR', shell=True)
    subprocess.run('~/bin/pvf.py -t gr -d 2 -k {ksp} -e {ec} -s {sig} -ispin {sp} -vdw {vdw} -lreal f -isif 2 -isym 0 -ibrion {ib} -ediff {ed} -ftol {ft} -algo All'
                   .format(ksp=kspace, ec=ecut, sig=sigma, sp=ispin, vdw=vdw_scheme, ib=ibrion, ed=ediff, ft=ftol), shell=True)
    # subprocess.run(\'mpirun -np {mp} vasp_std\', shell=True)

    if q == 'dev':
        subprocess.run('~/bin/pbs.py dev -N {} -n 2 -v std'.format('-'.join(os.getcwd().split('/')[-5:])), shell=True)
    elif q == 'q1':
        subprocess.run('~/bin/pbs.py q1 -N {} -v std'.format('-'.join(os.getcwd().split('/')[-5:])), shell=True)
    else:
        print('Queue {} not recognized'.format(q))
        exit()

    subprocess.run('find {} -name submit.pbs -exec qsub {{}} \;'.format(os.getcwd()), shell=True)


# ic stands for ion-clamped
def ops_ic():
    subprocess.run('~/bin/pvf.py -t er -d 2 -k {ksp} -e {ec} -s {sig} -vdw {vdw} -lreal f -ediff {ed}'
                   .format(ksp=kspace, ec=ecut, sig=sigma, vdw=vdw_scheme, ed=ediff), shell=True)
    subprocess.run('~/bin/pbs.py dev -N {} -n 2 -v std'.format('-'.join(os.getcwd().split('/')[-5:])), shell=True)
    subprocess.run('find {} -name submit.pbs -exec qsub {{}} \;'.format(os.getcwd()), shell=True)


def read_time():
    """
    Reads OUTCAR file and returns time taken in seconds.
    """
    time = None
    with open('OUTCAR') as f:
        for line in f.readlines():
            if 'Elapsed time (sec)' in line:
                time = float(line.split()[3])
    return time


def finished():
    done = read_time() is not None
    return done


def reset():
    os.chdir('elastic')
    for eps_dir in ['eps_11', 'eps_22', 'eps_12']:
        os.chdir(eps_dir)
        for sub_dir in os.listdir():
            os.chdir(sub_dir)
            if not finished():
                print('{}/{} unfinished and will be rerun.'.format(eps_dir, sub_dir))
                subprocess.run('mv CONTCAR POSCAR', shell=True)
                subprocess.run('find {} -name submit.pbs -exec qsub {{}} \;'.format(os.getcwd()), shell=True)
            os.chdir('..')
        os.chdir('..')
    os.chdir('..')


if task == 'pre':
    setup_elastic2D('CONTCAR', ops, frac, N)
elif task == 'pre-ic':
    setup_elastic2D('CONTCAR', ops_ic, frac, N)
elif task == 'post':
    post_elastic2D('CONTCAR', symmetrize=sym)
elif task == 'plot':
    post_elastic2D('CONTCAR', plot=True, symmetrize=sym)
elif task == 'reset':
    reset()
