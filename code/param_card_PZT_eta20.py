from my_units import *
from signal_functions import *

##############################
##### CRYSTAL PARAMETERS #####
##############################

R_S = 6; #relativistic enhancement factor divided by effective quantum numbers
Z_S = 88; # charge of Schiff atom
A_S = 207; # atomic number of Schiff atom
M_S = Z_S**2 * R_S / BohrRadius**4 # atomic matrix element

rho = 7500 * Kg * Meter**-3 # mass density
N_c = 4 # number of atoms in unit cell
N_S = 1 # number of Schiff atoms
V_c = 150*10**-30 * Meter**3 # volume of unit cell

n_N = N_S/V_c # number density of Schiff spins
S_schiff_over_theta = 0.16 * ElectronCharge * fm**3; # Schiff moment proportionality constant with theta
S_schiff = ThetaAxion * S_schiff_over_theta # Schiff moment from QCD axion DM

c_LA = 1e-3 #mechanical loss angle
c_11 = 1e11 * Newton / Meter**2 * (1+1j*c_LA) # mechanical stiffness constant

v = np.sqrt(c_11 / rho) # longitudinal sound speed

eps_LA = 1e-3 #electrical loss angle
eps_11 = (1-1j*eps_LA) * 235 # dielectric constant
beta_11 = eps_11**-1 # impermittivity constant

zeta_11_over_S = (1/235) * 4 * np.pi * ElectronCharge * N_S / V_c * (ElectronCharge * BohrRadius**2 / AlphaEM ) * M_S
zeta_11 = zeta_11_over_S * S_schiff # electroaxionic tensor for nuclear coupling
zeta_11_G_aee = np.sqrt(RhoDMG/2) * (AlphaEM * N_c / V_c) * (ElectronCharge * BohrRadius**2 / AlphaEM) # electroaxionic tensor for electron coupling


h_LA = 0e-6
e_11 = 9.5 * Coulomb / Meter**2
h_11 = (1-1j*h_LA) * np.real(beta_11 * e_11)
h_tilde = e_11 / ((N_c * ElectronCharge * BohrRadius)/V_c)

xi_11_over_S = h_tilde * 4 * np.pi * ElectronCharge * N_S / V_c * M_S
xi_11 =  xi_11_over_S * S_schiff # piezoaxionic tensor for nuclear coupling
xi_11_G_aee = np.sqrt(RhoDMG/2) * (AlphaEM * N_c / V_c) # piezoaxionic tensor for electron coupling

a = 10; b = 10; # transverse aspect ratios a = l_2/l_1, b = l_3/l_1
T = 10**-3 * Kelvin # thermodynamic temperature
P_nuc = 1 # nuclear spin polarization fraction

mu_N = MuNuclear # nuclear magnetic moment
T_2 = 10**(-3) * Second # transverse spin coherence time

k2 = fn_k2_TE(h_11,c_11,beta_11); #EM coupling factor

##############################
##### READOUT PARAMETERS #####
##############################

L_squid = 1e-10 * Henry # SQUID dynamical inductance
R_squid = 10 * Ohm # SQUID dynamical resistance

k_i = 0.75 # SQUID input coupling factor
L_i = 1e-6 * Henry # input inductance
k_f = 0.75 # transformer coupling factor

eps_L_1 = 1e-6; # loss angle of readout inductor

Phi0 = np.pi/ElectronCharge
S_flux_squid = (2.5*10**-7 * Phi0 / np.sqrt(Hz))**2

eta = 1