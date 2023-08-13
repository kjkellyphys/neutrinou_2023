from scipy.linalg import expm
import numpy as np

def UMat(t12, t13, t23, d):
    c12, s12 = np.cos(t12), np.sin(t12)
    c13, s13 = np.cos(t13), np.sin(t13)
    c23, s23 = np.cos(t23), np.sin(t23)
    en = np.cos(-d) + 1.j*np.sin(-d)
    ep = np.cos(d) + 1.j*np.sin(d)
    return [[c12*c13, s12*c13, s13*en],
            [-s12*c23 - c12*s23*s13*ep, c12*c23 - s12*s23*s13*ep, s23*c13],
            [s12*s23 - c12*c23*s13*ep, -c12*s23 - s12*c23*s13*ep, c23*c13]]

def KMat(z, L, dm21, dm31):
    return [[0.0, 0.0, 0.0], [0.0, 2*1.267*dm21*L/z, 0.0], [0.0, 0.0, 2*1.267*dm31*L/z]]

def VMass(L, dens, Mode, t12, t13, t23, d):
    VFl = [[3.845e-04*0.5*L*dens*Mode, 0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.]]
    UM = UMat(t12, t13, t23, d)
    UDagM = np.conj(np.transpose(UM))
    UStarM = np.conj(UM)
    UTranM = np.transpose(UM)

    if Mode == 1:
        return np.matmul(np.matmul(UDagM, VFl), UM)
    else:
        return np.matmul(np.matmul(UTranM, VFl), UStarM)

def ExpHFn(z,L,dens,Mode,t12, t13, t23, d, dm21, dm31):
    KMa = KMat(z,L,dm21,dm31)
    VMa = VMass(L,dens,Mode,t12, t13, t23, d)
    return expm(-1.j*(KMa+VMa))

def GetProb3x3(evec, NuNuBar, oscillation_parameters, L, rho):
    sinsq_t12 = oscillation_parameters['sinsq_t12']
    sinsq_t13 = oscillation_parameters['sinsq_t13']
    sinsq_t23 = oscillation_parameters['sinsq_t23']
    delta = oscillation_parameters['delta']
    dm21 = oscillation_parameters['dm_21']
    dm31 = oscillation_parameters['dm_31']

    t12, t13, t23 = np.arcsin(np.sqrt([sinsq_t12, sinsq_t13, sinsq_t23]))

    UM = UMat(t12, t13, t23, delta)
    UDagM = np.conj(np.transpose(UM))
    UStarM = np.conj(UM)
    UTranM = np.transpose(UM)

    PVec = [[] for ke in range(len(evec))]
    for ke in range(len(PVec)):
        EH = ExpHFn(evec[ke],L,rho,NuNuBar,t12, t13, t23, delta,dm21,dm31)
        if NuNuBar == 1:
            PVec[ke] = np.abs(np.matmul(np.matmul(UM, EH), UDagM))**2
        else:
            PVec[ke] = np.abs(np.matmul(np.matmul(UStarM, EH), UTranM))**2

    return PVec
