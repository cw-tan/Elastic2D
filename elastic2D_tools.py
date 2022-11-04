import numpy as np


def gen_strained_lats(lattice, f=0.02, N=7):
    """
    Generates strained lattices for later post-processing into the second-order
    elastic constants by fitting stress-strain data points to the generalized
    Hooke's Law.

    Args:
        lattice (Numpy array) : lattice corresponding to a 2D material
                                (which includes a vvacuum layer)
        f (float)             : fraction of the lattice deformation
        N (int)               : number of stress-strain data points

    Returns:
        tuple of (Numpy array, list):
        1. Numpy array of strain values
        2. list of deformed lattices for 2D elastic constant calculations
    """
    ds = np.linspace(-f, f, N)

    eps_11_deformed_lattices = []
    # apply eps_11 (for C1111, C2211, C1211)
    for d in ds:
        strain = np.zeros((3, 3))
        strain[0, 0] = d
        deformed_lattice = ((np.eye(3) + strain) @ lattice.T).T
        eps_11_deformed_lattices.append(deformed_lattice)

    eps_22_deformed_lattices = []
    # apply eps_22 (for C1122, C2222, C1222)
    for d in ds:
        strain = np.zeros((3, 3))
        strain[1, 1] = d
        deformed_lattice = ((np.eye(3) + strain) @ lattice.T).T
        eps_22_deformed_lattices.append(deformed_lattice)

    eps_12_deformed_lattices = []
    # apply eps_12 (for C1112, C2212, C1212)
    for d in ds:
        strain = np.zeros((3, 3))
        strain[0, 1] = d / 2
        strain[1, 0] = d / 2
        deformed_lattice = ((np.eye(3) + strain) @ lattice.T).T
        eps_12_deformed_lattices.append(deformed_lattice)

    return ds, [eps_11_deformed_lattices, eps_22_deformed_lattices, eps_12_deformed_lattices]


def process_2D_elastic_constants(ds, stresses, c=None, symmetrize=True):
    """
    Fits the stress-strain data points to a quadratic polynomial and extracts
    the linear coefficient as the elastic constants. Note that the units of the
    returned elastic constants is the units of the stresses times the units of
    the c variable (if given). The distinction between the Voigt and Mandel notation
    of the returned elastic constants can be found in
    Marcin Maździarz 2019 2D Mater.6 048001
    (https://iopscience.iop.org/article/10.1088/2053-1583/ab2ef3).

    Args:
        ds (Numpy array) : strain array output by gen_strained_lats()
        stresses (list)  : list of stresses in the same format as the list of
                           deformed lattices from gen_strained_lats()
        c (float)        : height of the lattice (c-axis length)
        symmetrize (bool): whether the elastic constants matrix is symmetrized
    Returns:
        tuple of Numpy arrays: (Cs_voigt, Cs_mandel, Cs_res)
        Cs_voigt and Cs_mandel are the elastic constants in Voigt and Mandel (tensor)
        notation while Cs_res shows the stress-strain fitting error.
    """
    stresses_eps11, stresses_eps22, stresses_eps12 = [], [], []

    # symmetrize stresses (in case)
    for stress in stresses[0]:
        stresses_eps11.append((stress + stress.T) / 2)
    for stress in stresses[1]:
        stresses_eps22.append((stress + stress.T) / 2)
    for stress in stresses[2]:
        stresses_eps12.append((stress + stress.T) / 2)

    # for C1111, C2211, C1211
    C1111_fit = np.polyfit(ds, np.array([stress[0, 0] for stress in stresses_eps11]), 2, full=True)
    C1111 = C1111_fit[0][-2]
    C1111_res = C1111_fit[1][0]
    C2211_fit = np.polyfit(ds, np.array([stress[1, 1] for stress in stresses_eps11]), 2, full=True)
    C2211 = C2211_fit[0][-2]
    C2211_res = C2211_fit[1][0]
    C1211_fit = np.polyfit(ds, np.array([stress[0, 1] for stress in stresses_eps11]), 2, full=True)
    C1211 = C1211_fit[0][-2]
    C1211_res = C1211_fit[1][0]

    # for C1122, C2222, C1222
    C1122_fit = np.polyfit(ds, np.array([stress[0, 0] for stress in stresses_eps22]), 2, full=True)
    C1122 = C1122_fit[0][-2]
    C1122_res = C1122_fit[1][0]
    C2222_fit = np.polyfit(ds, np.array([stress[1, 1] for stress in stresses_eps22]), 2, full=True)
    C2222 = C2222_fit[0][-2]
    C2222_res = C2222_fit[1][0]
    C1222_fit = np.polyfit(ds, np.array([stress[0, 1] for stress in stresses_eps22]), 2, full=True)
    C1222 = C1222_fit[0][-2]
    C1222_res = C1222_fit[1][0]

    # for C1112, C2212, C1212
    C1112_fit = np.polyfit(ds, np.array([stress[0, 0] for stress in stresses_eps12]), 2, full=True)
    C1112 = C1112_fit[0][-2]
    C1112_res = C1112_fit[1][0]
    C2212_fit = np.polyfit(ds, np.array([stress[1, 1] for stress in stresses_eps12]), 2, full=True)
    C2212 = C2212_fit[0][-2]
    C2212_res = C2212_fit[1][0]
    C1212_fit = np.polyfit(ds, np.array([stress[0, 1] for stress in stresses_eps12]), 2, full=True)
    C1212 = C1212_fit[0][-2]
    C1212_res = C1212_fit[1][0]

    # possible future TODO: additional symmetrization based on lattice type
    if symmetrize:
        C1122_sym = (C1122 + C2211) / 2
        C1122 = C2211 = C1122_sym
        C1112_sym = (C1112 + C1211) / 2
        C1112 = C1211 = C1112_sym
        C2212_sym = (C2212 + C1222) / 2
        C2212 = C1222 = C2212_sym

    Cs_mandel = np.array([[C1111, C1122, np.sqrt(2) * C1112],
                          [C2211, C2222, np.sqrt(2) * C2212],
                          [np.sqrt(2) * C1211, np.sqrt(2) * C1222, 2 * C1212]])

    Cs_voigt = np.array([[C1111, C1122, C1112],
                         [C2211, C2222, C2212],
                         [C1211, C1222, C1212]])

    Cs_res = np.array([[C1111_res, C1122_res, C1112_res],
                       [C2211_res, C2222_res, C2212_res],
                       [C1211_res, C1222_res, C1212_res]])
    if c is not None:
        return Cs_voigt * c, Cs_mandel * c, np.sqrt(Cs_res) * c
    else:
        return Cs_voigt, Cs_mandel, np.sqrt(Cs_res)


def check_elastic_stability(Cs):
    """
    Assesses the elastic stability using the criterion that the eigenvalues
    of the elastic constants in Mandel notation must be positive.
    Ref: Marcin Maździarz 2019 2D Mater.6 048001
    (https://iopscience.iop.org/article/10.1088/2053-1583/ab2ef3).

    Args:
        Cs (Numpy array): 2D elastic constants in the second-rank tensor
                          or Mandel notation
    Returns:
        Tuple of (boo, Numpy array): (Stable?, eigenvalues of Cs)
    """
    evals = np.linalg.eigvals(Cs)
    return np.all(evals > 0), evals


def planar_bulk_modulus(Cs, extensibility=False):
    """
    Computes the planar bulk modulus or areal extensibility as defined and derived
    in https://doi.org/10.1016/j.carbon.2019.10.041.

    Args:
        Cs (Numpy array)    : 2D elastic constants in Voigt notation
        extensibility (bool): Whether to return the areal extensibility instead
    Returns:
        float: planar bulk modulus (or optionally areal extensibility)
    """
    Ss = np.linalg.inv(Cs)  # compliance matrix
    k = Ss[0, 0] + Ss[1, 1] + 2 * Ss[0, 1]
    if extensibility:
        return k
    else:
        return 1 / k
