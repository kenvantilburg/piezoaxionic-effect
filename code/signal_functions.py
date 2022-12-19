import numpy as np
import scipy as sp
from my_units import *
from tqdm import tqdm
from scipy.interpolate import interp1d
from scipy.optimize import fmin, minimize, fsolve, brentq

######## TANGENT EXPANSION #######################
def fn_tan_expansion(x,k_min,k_max):
    """Series expansion of tan(x) between the poles labeled by k_min and k_max. The first pole is at k_min = 1."""
    return np.sum([8 * x / ((2*k-1)**2 * np.pi**2 - 4 * x**2 ) for k in range(k_min,k_max+1)],axis=0)

######## RELATIVISTIC ENHANCEMENT FACTORS #######################
def fn_gamma_rel(j,Z,alpha=AlphaEM):
    """Relativistic gamma factor as a function of total angular momentum j, nuclear charge Z, and fine-structure constant alpha."""
    return np.sqrt((j+1/2)**2 - Z**2 * alpha**2)

def fn_R_rel_factor_12(Z,A_over_Z=2,alpha=AlphaEM):
    """Relativistic enhancement factor R_{1/2} as a function of nuclear charge Z, ratio of A/Z ('A_over_Z'), and fine-structure constant alpha."""
    gamma = fn_gamma_rel(1/2,Z,alpha)
    A = A_over_Z * Z
    R0 = A**(1/3) * (1.2 * FemtoMeter)
    a0 = BohrRadius
    frac_1 = 3 * gamma * (2 * gamma - 1) / (2 * gamma + 1)
    frac_2 = 4 / (sp.special.gamma(2 * gamma + 1))**2
    frac_3 = (a0 / (2 * Z * R0))**(2-2*gamma)
    return frac_1 * frac_2 * frac_3

def fn_R_rel_factor_32(Z,A_over_Z=2,alpha=AlphaEM):
    """Relativistic enhancement factor R_{3/2} as a function of nuclear charge Z, ratio of A/Z ('A_over_Z'), and fine-structure constant alpha."""
    gamma_12 = fn_gamma_rel(1/2,Z,alpha)
    gamma_32 = fn_gamma_rel(3/2,Z,alpha)
    A = A_over_Z * Z
    R0 = A**(1/3) * (1.2 * FemtoMeter)
    a0 = BohrRadius
    frac_1 = 6 * ((gamma_12+1)*(gamma_32+2)+Z**2*alpha**2) / (sp.special.gamma(2*gamma_12+1) * sp.special.gamma(2*gamma_32+1))
    frac_2 = gamma_12 + gamma_32 - 2
    frac_3 = (a0 / (2 * Z * R0))**(3-gamma_12-gamma_32)
    return frac_1 * frac_2 * frac_3


######## AXION SIGNAL #######################
def fn_V_axion(omega,xi_11,zeta_11,l,v,h_11,c_11,N_series,P_nuc):
    """Axion-induced voltage for the thickness expander mode as a function of:
    -- angular frequency omega
    -- piezoaxionic tensor component xi_11
    -- electroaxionic tensor component zeta_11
    -- crystal thickness l
    -- longitudinal sound speed v
    -- piezoelectric tensor component h_11
    -- elastic stiffness tensor component c_11
    -- the number of crystals used in series N_series
    -- the spin polarization fraction P_nuc.
    """
    return np.real(- P_nuc * N_series * ((h_11 * xi_11 / c_11) * (2 * v / omega) * np.tan(omega*l/(2*v)) + zeta_11 * l))

def fn_xi_over_V(omega,l,v,h_11,c_11,N_series,P_nuc):
    """Ratio of piezoaxionic tensor component xi_11 and axion-induced voltage as a function of:
    -- angular frequency omega
    -- crystal thickness l
    -- longitudinal sound speed v
    -- piezoelectric tensor component h_11
    -- elastic stiffness tensor component c_11
    -- the number of crystals used in series N_series
    -- the spin polarization fraction P_nuc.
    """
    return N_series**-1 * P_nuc**-1 * ((h_11/c_11) * (2*v)/omega * np.tan( omega*l / (2*v) ) )**-1

def fn_zeta_over_V(l,N_series,P_nuc):
    """Ratio of electroaxionic tensor component zeta_11 and axion-induced voltage as a function of:
    -- crystal thickness l
    -- the number of crystals used in series N_series
    -- the spin polarization fraction P_nuc.
    """
    return N_series**-1 * P_nuc**-1 * l**-1

######## FREQUENCIES ############################
def fn_omega_n(l,v,n):
    """Mechanical (angular) resonance frequency at harmonic n (fundamental harmonic is n=1) for the thickness expander mode, as a function of crystal thickness l and longitudinal sound speed v. Returns half the fundamental frequency if n = 0."""
    if n > 0:
        return (2*n -1)*np.pi*np.real(v)/l
    else:
        return 0.5*np.pi*np.real(v)/l

def fn_omega_res(l,v,h_11,c_11,beta_11):
    """Natural (angular) resonance frequency near fundamental harmonic (n=1) of the thickness expander mode, as a function of:
    -- crystal thickness l
    -- longitudinal sound speed v
    -- piezoelectric tensor component h_11
    -- elastic stiffness tensor component c_11
    -- impermittivity tensor component beta_11.
    """
    k2 = fn_k2_TE(h_11,c_11,beta_11) #EM coupling factor
    return np.real((np.pi * v / l) * np.sqrt(1-8*k2 / np.pi**2))

######## IMPEDANCES ##############################
#crystal impedances
def fn_k2_TE(h_11,c_11,beta_11):
    """EM coupling factor k^2 for thickness expander mode, as function of:
    -- piezoelectric tensor component h_11
    -- elastic stiffness tensor component c_11
    -- impermittivity tensor component beta_11.
    """
    return h_11**2 / (c_11 * beta_11)

def fn_C_crystal_TE_c(l,a,b,beta_11):
    """Clamped capacitance for thickness expander mode, as a function of:
    -- crystal thickness l = l_1
    -- transverse aspect ratios a = l_2 / l_1 and b = l_3 / l_1
    -- impermittivity tensor component beta_11.
    """
    return a * b * l / beta_11

def fn_Z_crystal_TE_c(omega,l,a,b,beta_11):
    """Clamped impedance for thickness expander mode, as a function of:
    -- angular frequency omega
    -- crystal thickness l = l_1
    -- transverse aspect ratios a = l_2 / l_1 and b = l_3 / l_1
    -- impermittivity tensor component beta_11.
    """
    C_c = fn_C_crystal_TE_c(l,a,b,beta_11)
    return 1/(1j * omega * C_c)

def fn_Z_crystal_TE_LA(omega,l,v,a,b,beta_11,k2):
    """Total impedance of crystal setup, for thickness expander mode, as a function of:
    -- angular frequency omega
    -- crystal thickness l = l_1
    -- longitudinal sound speed v
    -- transverse aspect ratios a = l_2 / l_1 and b = l_3 / l_1
    -- impermittivity tensor component beta_11
    -- EM coupling factor k^2.
    """
    Z_c = fn_Z_crystal_TE_c(omega,l,a,b,beta_11)
    tan_piece = k2 * (2 * v/(omega*l)) * np.tan(omega * l / (2 * v))
    return Z_c * (1-tan_piece)

# readout impedances
def fn_Z_L_1(omega,L_1):
    """Readout inductor impedance as a function of angular frequency omega and inductance L_1."""
    return 1j * omega * L_1

def fn_Z_C_1(omega,C_1):
    """Readout capacitor impedance as a function of angular frequency omega and capacitance C_1."""
    return 1/(1j * omega * C_1)

def fn_Z_squid(omega, L_squid, R_squid, L_i, k_i, L_2):
    """Dynamic squid impedance as a function of angular frequency omega, inductance L_squid, resistance R_squid and SQUID coupling factor k_i."""
    alpha_e_squared = k_i**2/(1+np.real(L_2/L_i))
    R_dyn_red = R_squid
    L_dyn_red = (1-alpha_e_squared) * L_squid / 0.1
    return ((1j * omega * L_dyn_red)**-1 + (R_dyn_red)**-1)**-1

def fn_Z_squid_t(omega, L_squid, R_squid, L_i, k_i, L_2):
    """Squid back-impedance on transformer as a function of angular frequency omega, inductance L_squid and resistance R_squid, input inductor L_i, and SQUID coupling factor k_i."""
    return omega**2 * k_i**2 * np.real(L_i) * np.real(L_squid) / fn_Z_squid(omega, L_squid, R_squid, L_i, k_i, L_2)

def fn_Z_squid_p(omega, L_squid, R_squid, L_i, k_i, L_1, L_2, k_f):
    """Squid back-impedance on primary circuit, as a function of:
    -- angular frequency omega
    -- dynamical inductance L_squid and resistance R_squid
    -- input inductor L_i
    -- SQUID coupling factor k_i
    -- readout inductor L_1
    -- transformer inductor L_2
    -- transformer coupling factor k_f
    """
    return omega**2 * k_f**2 * np.real(L_1) * np.real(L_2) / (1j * omega * L_2 + 1j * omega * L_i + fn_Z_squid_t(omega, L_squid, R_squid, L_i, k_i, L_2))

# total circuit impedance
def fn_Z_total(omega,l,v,a,b,beta_11,k2,L_squid, R_squid, L_i, k_i, C_1, L_1, L_2, k_f, N_series,N_parallel):
    """Total impedance of primary circuit: crystal equivalent circuit of N_series * N_parallel crystals, input inductor, input circuit capacitor, squid back-impedance; as a function of:
    -- angular frequency omega
    -- crystal thickness l = l_1
    -- longitudinal sound speed v
    -- transverse aspect ratios a = l_2 / l_1 and b = l_3 / l_1
    -- impermittivity tensor component beta_11
    -- EM coupling factor k^2
    -- dynamical inductance L_squid and resistance R_squid
    -- input inductor L_i
    -- SQUID coupling factor k_i
    -- readout capacitor C_1
    -- readout inductor L_1
    -- transformer inductor L_2
    -- transformer coupling factor k_f
    """
    Z_crystal = fn_Z_crystal_TE_LA(omega,l,v,a,b,beta_11,k2)
    Z_L_1 = fn_Z_L_1(omega,L_1)
    Z_C_1 = fn_Z_C_1(omega,C_1)
    Z_squid_p = fn_Z_squid_p(omega, L_squid, R_squid, L_i, k_i, L_1, L_2, k_f)
    return N_series/N_parallel * Z_crystal + Z_C_1 + Z_L_1 + Z_squid_p

######## AXION CURRENTS ######################################
def fn_I_axion_p(omega,xi_11,zeta_11,h_11,c_11,P_nuc,l,v,a,b,beta_11,k2,L_squid, R_squid, L_i, k_i, C_1, L_1, L_2, k_f, N_series,N_parallel):
    """Total axion-induced current through primary circuit, as a function of:
    -- angular frequency omega
    -- piezoaxionic tensor component xi_11
    -- electroaxionic tensor component zeta_11
    -- piezoelectric tensor component h_11
    -- elastic stiffness tensor component c_11
    -- the spin polarization fraction P_nuc
    -- crystal thickness l = l_1
    -- longitudinal sound speed v
    -- transverse aspect ratios a = l_2 / l_1 and b = l_3 / l_1
    -- impermittivity tensor component beta_11
    -- EM coupling factor k^2
    -- dynamical inductance L_squid and resistance R_squid
    -- input inductor L_i
    -- SQUID coupling factor k_i
    -- readout capacitor C_1
    -- readout inductor L_1
    -- transformer inductor L_2
    -- transformer coupling factor k_f
    -- number of crystals in series (N_series) and parallel (N_parallel).
    """
    V_axion = fn_V_axion(omega,xi_11,zeta_11,l,v,h_11,c_11,N_series,P_nuc)
    Z_total = fn_Z_total(omega,l,v,a,b,beta_11,k2,L_squid, R_squid, L_i, k_i, C_1, L_1, L_2, k_f,N_series,N_parallel)
    I_axion_p = V_axion / Z_total
    return I_axion_p

def fn_flux_axion_squid(omega,xi_11,zeta_11,h_11,c_11,P_nuc,l,v,a,b,beta_11,k2,L_squid, R_squid, L_i, k_i, C_1, L_1, L_2, k_f, N_series,N_parallel):
    """Total axion-induced flux through SQUID, as a function of:
    -- angular frequency omega
    -- piezoaxionic tensor component xi_11
    -- electroaxionic tensor component zeta_11
    -- piezoelectric tensor component h_11
    -- elastic stiffness tensor component c_11
    -- the spin polarization fraction P_nuc
    -- crystal thickness l = l_1
    -- longitudinal sound speed v
    -- transverse aspect ratios a = l_2 / l_1 and b = l_3 / l_1
    -- impermittivity tensor component beta_11
    -- EM coupling factor k^2
    -- dynamical inductance L_squid and resistance R_squid
    -- input inductor L_i
    -- SQUID coupling factor k_i
    -- readout capacitor C_1
    -- readout inductor L_1
    -- transformer inductor L_2
    -- transformer coupling factor k_f
    -- number of crystals in series (N_series) and parallel (N_parallel).
    """
    I_axion_p = fn_I_axion_p(omega,xi_11,zeta_11,h_11,c_11,P_nuc,l,v,a,b,beta_11,k2,L_squid, R_squid, L_i, k_i, C_1, L_1, L_2, k_f, N_series,N_parallel)
    flux = I_axion_p * (k_f * k_i * np.sqrt(np.real(L_1) * L_2 * L_i * L_squid) / (L_i + L_2) )
    return flux

######## CRYSTAL QUALITY FACTOR ##############################
def fn_Q_factor(omega,l,v,a,b,beta_11,k2):
    """Crystal quality factor as a function of:
    -- angular frequency omega
    -- crystal thickness l = l_1
    -- longitudinal sound speed v
    -- transverse aspect ratios a = l_2 / l_1 and b = l_3 / l_1
    -- impermittivity tensor component beta_11
    -- EM coupling factor k^2.
    """
    C_c = np.real(fn_C_crystal_TE_c(l,a,b,beta_11))
    Re_Z_p = np.real(fn_Z_crystal_TE_LA(omega,l,v,a,b,beta_11,k2))
    k_fac = np.real(k2) * ( 1 - (3 * np.real(v) / (omega * l)) * np.sin(omega * l / np.real(v)) ) / ( 1 + np.cos(omega * l / np.real(v)) )
    return 1/(omega * C_c * Re_Z_p) * (1 + k_fac )


######## NOISE SPECTRAL DENSITIES ##############################
def fn_S_V_p_crystal_TE(T,omega,l,v,a,b,beta_11,k2,N_series,N_parallel):
    """Mechanical voltage noise as referred to primary circuit, as a function of:
    -- temperature T
    -- angular frequency omega
    -- crystal thickness l = l_1
    -- longitudinal sound speed v
    -- transverse aspect ratios a = l_2 / l_1 and b = l_3 / l_1
    -- impermittivity tensor component beta_11
    -- EM coupling factor k^2
    -- number of crystals in series (N_series) and parallel (N_parallel).
    """
    return 4 * T * N_series / N_parallel * np.real(fn_Z_crystal_TE_LA(omega,l,v,a,b,beta_11,k2))

def fn_S_V_p_L_1(T,omega,L_1):
    """Inductor voltage noise as referred to primary circuit as a function of temperature T, angular frequency omega, and readout inductance L_1."""
    return 4 * T * np.real(fn_Z_L_1(omega,L_1))

def fn_S_flux_squid(omega,S_flux_squid):
    """Squid flux noise as a function of angular frequency omega, and baseline noise spectral density S_flux_squid."""
    return S_flux_squid

def fn_S_V_p_squid_flux(T, omega,l,v,a,b,beta_11,k2,L_squid, R_squid, L_i, k_i, C_1, L_1, L_2, k_f,S_flux_squid,N_series,N_parallel):
    """Squid imprecision voltage noise as referred to primary circuit, as a function of:
    -- temperature T
    -- angular frequency omega
    -- crystal thickness l = l_1
    -- longitudinal sound speed v
    -- transverse aspect ratios a = l_2 / l_1 and b = l_3 / l_1
    -- impermittivity tensor component beta_11
    -- EM coupling factor k^2
    -- dynamical inductance L_squid and resistance R_squid
    -- input inductor L_i
    -- SQUID coupling factor k_i
    -- readout capacitor C_1
    -- readout inductor L_1
    -- transformer inductor L_2
    -- transformer coupling factor k_f
    -- SQUID flux noise S_flux_squid
    -- number of crystals in series (N_series) and parallel (N_parallel).
    """
    S_flux = fn_S_flux_squid(omega,S_flux_squid);
    Z_total = fn_Z_total(omega,l,v,a,b,beta_11,k2,L_squid, R_squid, L_i, k_i, C_1, L_1, L_2, k_f,N_series,N_parallel)
    return np.abs(Z_total)**2 * S_flux *  (np.real(L_i) + L_2)**2/(k_f**2 * k_i**2 * np.real(L_1) * L_2 * L_i * L_squid)

def fn_S_V_p_squid_BA(T, omega,l,v,a,b,beta_11,k2,L_squid, R_squid, L_i, k_i, C_1, L_1, L_2, k_f, eta, S_flux_squid,N_series,N_parallel):
    """Squid back-action noise as referred to primary circuit (assuming quantum limit), as a function of:
    -- temperature T
    -- angular frequency omega
    -- crystal thickness l = l_1
    -- longitudinal sound speed v
    -- transverse aspect ratios a = l_2 / l_1 and b = l_3 / l_1
    -- impermittivity tensor component beta_11
    -- EM coupling factor k^2
    -- dynamical inductance L_squid and resistance R_squid
    -- input inductor L_i
    -- SQUID coupling factor k_i
    -- readout capacitor C_1
    -- readout inductor L_1
    -- transformer inductor L_2
    -- transformer coupling factor k_f
    -- SQUID flux noise S_flux_squid
    -- number of crystals in series (N_series) and parallel (N_parallel).
    """
    S_V_P_squid_flux = fn_S_V_p_squid_flux(T, omega,l,v,a,b,beta_11,k2,L_squid, R_squid, 
                                           L_i, k_i, C_1, L_1, L_2, k_f,S_flux_squid,N_series,N_parallel)
    Z_total = fn_Z_total(omega,l,v,a,b,beta_11,k2,L_squid, R_squid, L_i, k_i, C_1, L_1, L_2, k_f,N_series,N_parallel)
    return omega**2 * np.abs(Z_total)**2 * eta**2 / S_V_P_squid_flux

def fn_S_V_p_squid_BI(T, omega, L_squid, R_squid, L_i, k_i, L_1, L_2, k_f):
    """Squid back-impedance voltage noise as referred to primary circuit, as a function of:
    -- temperature T
    -- angular frequency omega
    -- dynamical inductance L_squid and resistance R_squid
    -- input inductor L_i
    -- SQUID coupling factor k_i
    -- readout capacitor C_1
    -- readout inductor L_1
    -- transformer inductor L_2
    -- transformer coupling factor k_f
    """
    return 4 * T * np.real(fn_Z_squid_p(omega, L_squid, R_squid, L_i, k_i, L_1, L_2, k_f))

def fn_S_V_p_magnetization(omega, T_2, mu_N, n_N, l, a, b):
    """Voltage noise power originating from transverse magnetization fluctuations as referred to primary circuit, as a function of:
    -- angular frequency omega
    -- transverse spin coherence time T_2
    -- nuclear magnetic moment mu_N
    -- nuclear spin number density n_N
    -- crystal thickness l = l_1
    -- transverse aspect ratios a = l_2 / l_1 and b = l_3 / l_1.
    """
    A = l**2 * a * b
    Vol = l * A
    return (1/8) * mu_N**2 * n_N * Vol**(-1) * T_2**(-1) * A**2 * omega**0

def fn_S_V_p_total(T, omega,l,v,a,b,beta_11,k2,L_squid, R_squid, L_i, k_i, C_1, L_1, L_2, k_f, eta, S_flux_squid, T_2, mu_N, n_N, N_series,N_parallel):
    """Total voltage noise as referred to primary circuit, as a function of:
    -- temperature T
    -- angular frequency omega
    -- crystal thickness l = l_1
    -- longitudinal sound speed v
    -- transverse aspect ratios a = l_2 / l_1 and b = l_3 / l_1
    -- impermittivity tensor component beta_11
    -- EM coupling factor k^2
    -- dynamical inductance L_squid and resistance R_squid
    -- input inductor L_i
    -- SQUID coupling factor k_i
    -- readout capacitor C_1
    -- readout inductor L_1
    -- transformer inductor L_2
    -- transformer coupling factor k_f
    -- SQUID flux noise S_flux_squid
    -- transverse spin coherence time T_2
    -- nuclear magnetic moment mu_N
    -- nuclear spin number density n_N
    -- number of crystals in series (N_series) and parallel (N_parallel).
    """
    S_V_p_crystals = fn_S_V_p_crystal_TE(T,omega,l,v,a,b,beta_11,k2,N_series,N_parallel)
    S_V_p_L_1 = fn_S_V_p_L_1(T,omega,L_1)
    S_V_p_squid_flux = fn_S_V_p_squid_flux(T, omega,l,v,a,b,beta_11,k2,L_squid, R_squid, L_i, k_i, C_1, L_1, L_2, k_f,S_flux_squid,N_series,N_parallel)
    S_V_p_squid_BA = fn_S_V_p_squid_BA(T, omega,l,v,a,b,beta_11,k2,L_squid, R_squid, L_i, k_i, C_1, L_1, L_2, k_f, eta, S_flux_squid,N_series,N_parallel);
    S_V_p_squid_BI = fn_S_V_p_squid_BI(T, omega, L_squid, R_squid, L_i, k_i, L_1, L_2, k_f);
    S_V_p_magnetization = fn_S_V_p_magnetization(omega, T_2, mu_N, n_N, l, a, b);
    return S_V_p_crystals + S_V_p_L_1 + S_V_p_squid_flux + S_V_p_squid_BA + S_V_p_squid_BI + S_V_p_magnetization

def fn_S_theta(T, omega,l,v,a,b,h_11,c_11,beta_11,k2,L_squid, R_squid, L_i, k_i, C_1, L_1, L_2, k_f, eta, S_flux_squid, T_2, mu_N, n_N, xi_11, zeta_11, P_nuc, N_series,N_parallel):
    """Total theta noise as a function of:
    -- temperature T
    -- angular frequency omega
    -- crystal thickness l = l_1
    -- longitudinal sound speed v
    -- transverse aspect ratios a = l_2 / l_1 and b = l_3 / l_1
    -- impermittivity tensor component beta_11
    -- EM coupling factor k^2
    -- dynamical inductance L_squid and resistance R_squid
    -- input inductor L_i
    -- SQUID coupling factor k_i
    -- readout capacitor C_1
    -- readout inductor L_1
    -- transformer inductor L_2
    -- transformer coupling factor k_f
    -- SQUID flux noise S_flux_squid
    -- transverse spin coherence time T_2
    -- nuclear magnetic moment mu_N
    -- nuclear spin number density n_N
    -- piezoaxionic tensor xi_11
    -- electroaxionic tensor zeta_11
    -- nuclear spin polarization P_nuc
    -- number of crystals in series (N_series) and parallel (N_parallel).
    """
    V_over_theta = fn_V_axion(omega,xi_11/ThetaAxion,zeta_11/ThetaAxion,l,v,h_11,c_11,N_series,P_nuc)
    S_V_p = fn_S_V_p_total(T, omega,l,v,a,b,beta_11,k2,L_squid, R_squid, L_i, k_i, C_1, L_1, L_2, k_f, eta, S_flux_squid, T_2, mu_N, n_N, N_series,N_parallel)
    return S_V_p / V_over_theta**2

######## OPTIMIZATION ##############################

def res_freq(T, l,v,a,b,h_11,c_11,beta_11,k2,L_squid, R_squid, L_i, k_i, C_1, L_1, L_2, k_f, S_flux_squid, T_2, mu_N, n_N, xi_11, zeta_11, P_nuc, N_series,N_parallel, branch):
    """find the fractional resonant frequency for a given branch, as a function of:
    -- temperature T
    -- crystal thickness l = l_1
    -- longitudinal sound speed v
    -- transverse aspect ratios a = l_2 / l_1 and b = l_3 / l_1
    -- impermittivity tensor component beta_11
    -- EM coupling factor k^2
    -- dynamical inductance L_squid and resistance R_squid
    -- input inductor L_i
    -- SQUID coupling factor k_i
    -- readout capacitor C_1
    -- readout inductor L_1
    -- transformer inductor L_2
    -- transformer coupling factor k_f
    -- SQUID flux noise S_flux_squid
    -- transverse spin coherence time T_2
    -- nuclear magnetic moment mu_N
    -- nuclear spin number density n_N
    -- piezoaxionic tensor xi_11
    -- electroaxionic tensor zeta_11
    -- nuclear spin polarization P_nuc
    -- number of crystals in series (N_series) and parallel (N_parallel)
    -- branch (below or above resonance) labelled 0 or 1.
    """
    omega_0 = np.abs(fn_omega_n(l,v,1));
    omega_res = fn_omega_res(l,v,h_11,c_11,beta_11)
    def fun_Z(omega_frac):
        return np.imag(fn_Z_total(omega_frac * omega_0,l,v,a,b,beta_11,k2,L_squid, R_squid, L_i, k_i, C_1, L_1, L_2, k_f, N_series,N_parallel))
    if branch==0:
        try:
            omega_frac1 = brentq(f=fun_Z,a=(4*omega_res-3*omega_0)/omega_0,b=(1-1e-10),xtol=1e-20)
            omega_frac2 = np.nan
        except ValueError:
            omega_frac1 = np.nan
            omega_frac2 = np.nan
    elif branch==1:
        try:
            omega_frac1 = np.nan
            omega_frac2 = brentq(f=fun_Z,a=1+1e-10,b=(3*omega_0-2*omega_res)/omega_0,xtol=1e-20)
        except ValueError:
            omega_frac1 = np.nan
            omega_frac2 = np.nan
    else:
        print("branch must be 0 or 1")
    return np.array([omega_frac1,omega_frac2])

        
def BA_crit(T, omega,l,v,a,b,beta_11,k2,L_squid, R_squid, L_i, k_i, C_1, L_1, L_2, k_f,eta,S_flux_squid, N_series,N_parallel):
    """criteria for optimised back action from arXiv:1803.01627:
    -- temperature T
    -- angular frequency omega
    -- crystal thickness l = l_1
    -- longitudinal sound speed v
    -- transverse aspect ratios a = l_2 / l_1 and b = l_3 / l_1
    -- impermittivity tensor component beta_11
    -- EM coupling factor k^2
    -- dynamical inductance L_squid and resistance R_squid
    -- input inductor L_i
    -- SQUID coupling factor k_i
    -- readout capacitor C_1
    -- readout inductor L_1
    -- transformer inductor L_2
    -- transformer coupling factor k_f
    -- SQUID flux noise S_flux_squid
    -- transverse spin coherence time T_2
    -- nuclear magnetic moment mu_N
    -- nuclear spin number density n_N
    -- piezoaxionic tensor xi_11
    -- electroaxionic tensor zeta_11
    -- nuclear spin polarization P_nuc
    -- number of crystals in series (N_series) and parallel (N_parallel)
    -- branch (below or above resonance) labelled 0 or 1.
    """    
    return  np.abs(2 * T * np.real(fn_Z_total(omega,l,v,a,b,beta_11,k2,L_squid, R_squid, L_i, k_i, C_1, L_1, L_2, k_f,N_series,N_parallel))/fn_S_V_p_squid_BA(T, omega,l,v,a,b,beta_11,k2,L_squid, R_squid, L_i, k_i, C_1, L_1, L_2, k_f,eta, S_flux_squid,N_series,N_parallel)-1)


def L2_find(T, l,v,a,b,h_11,c_11,beta_11,k2,L_squid, R_squid, L_i, k_i, C_1, L_1, k_f,eta, S_flux_squid, T_2, mu_N, n_N, xi_11, zeta_11, P_nuc, N_series,N_parallel,branch, x0):
    """Find the value of L2 that minimises the BA crit, as a function of:
    -- temperature T
    -- angular frequency omega
    -- crystal thickness l = l_1
    -- longitudinal sound speed v
    -- transverse aspect ratios a = l_2 / l_1 and b = l_3 / l_1
    -- impermittivity tensor component beta_11
    -- EM coupling factor k^2
    -- dynamical inductance L_squid and resistance R_squid
    -- input inductor L_i
    -- SQUID coupling factor k_i
    -- readout capacitor C_1
    -- readout inductor L_1
    -- transformer inductor L_2
    -- transformer coupling factor k_f
    -- SQUID flux noise S_flux_squid
    -- transverse spin coherence time T_2
    -- nuclear magnetic moment mu_N
    -- nuclear spin number density n_N
    -- piezoaxionic tensor xi_11
    -- electroaxionic tensor zeta_11
    -- nuclear spin polarization P_nuc
    -- number of crystals in series (N_series) and parallel (N_parallel)
    -- branch (below or above resonance) labelled 0 or 1
    -- Starting guess for L_2.
    """   
    omega_0 = np.abs(fn_omega_n(l,v,1))
    def omega_frac(L_2_new, branch):
        return res_freq(T, l,v,a,b,h_11,c_11,beta_11,k2,L_squid, R_squid, L_i, k_i, C_1, L_1, L_2_new, k_f, S_flux_squid, T_2, mu_N, n_N, xi_11, zeta_11, P_nuc, N_series,N_parallel, branch)[branch]
    def crit(L_2_new):
        return BA_crit(T, omega_frac(L_2_new, branch)*omega_0,l,v,a,b,beta_11,k2,L_squid, R_squid, L_i, k_i, C_1, L_1, L_2_new, k_f,eta,S_flux_squid, N_series,N_parallel)
    minimum = sp.optimize.minimize(crit, x0, method='Nelder-Mead', bounds=((np.real(L_i), 10000*np.real(L_1)),)).x[0]
    return [minimum, omega_frac(minimum, branch), crit(minimum)]

######## SENSITIVITY ##############################

def fn_theta_sens(t_shot, Q_a, T, omega,l,v,a,b,h_11,c_11,beta_11,k2,L_squid, R_squid, L_i, k_i, C_1, L_1, L_2, k_f, eta, S_flux_squid, T_2, mu_N, n_N, xi_11, zeta_11, P_nuc, N_series, N_parallel):
    """theta_axion sensitivity as a function of:
    -- shot time t_shot
    -- axion signal quality factor Q_a
    -- temperature T
    -- angular frequency omega
    -- crystal thickness l = l_1
    -- longitudinal sound speed v
    -- transverse aspect ratios a = l_2 / l_1 and b = l_3 / l_1
    -- impermittivity tensor component beta_11
    -- EM coupling factor k^2
    -- dynamical inductance L_squid and resistance R_squid
    -- input inductor L_i
    -- SQUID coupling factor k_i
    -- readout capacitor C_1
    -- readout inductor L_1
    -- transformer inductor L_2
    -- transformer coupling factor k_f
    -- SQUID flux noise S_flux_squid
    -- transverse spin coherence time T_2
    -- nuclear magnetic moment mu_N
    -- nuclear spin number density n_N
    -- piezoaxionic tensor xi_11
    -- electroaxionic tensor zeta_11
    -- nuclear spin polarization P_nuc
    -- number of crystals in series (N_series) and parallel (N_parallel).
    """
    S = fn_S_theta(T, omega,l,v,a,b,h_11,c_11,beta_11,k2,L_squid, R_squid, L_i, k_i, C_1, L_1, L_2, k_f, eta, S_flux_squid, T_2, mu_N, n_N, xi_11, zeta_11, P_nuc, N_series,N_parallel)
    return S**(1/2) * t_shot**(-1/4) * (Q_a * 2*np.pi / omega)**(-1/4)
