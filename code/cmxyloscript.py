dir_home = '../'
dir_data = dir_home+'data/'
dir_fig = dir_home+'figs/'
import sys
sys.path.insert(0, dir_home)


from preamble import *
from my_units import *
from signal_functions import *
from param_card_eta20 import *
import multiprocessing

l = 4 * CentiMeter; N_series = 1; N_parallel = 1; 

omega_0 = np.abs(fn_omega_n(l,v,1)); 
omega_res = fn_omega_res(l,v,h_11,c_11,beta_11);

C_c = fn_C_crystal_TE_c(l,a,b,beta_11)
C_1_fid = np.real(C_c)
L_1_fid = 1/(omega_0**2 * np.real(C_c))

spacing = 3 * (1- omega_res/omega_0);

N_series = 1; N_parallel = 1; 
vec_l = 4*CentiMeter / (1+spacing)**np.arange(0.,np.log(2)/spacing,1.); 
t_int = 10 * Year
N_shots = int(1e5)
t_shot = t_int / N_shots / len(vec_l)
Q_a = (235 / np.sqrt(2) * KiloMeter / Second)**(-2);
print(vec_l / CentiMeter)

vec_omega_plot = np.logspace(np.log10((1-spacing)*np.abs(fn_omega_n(vec_l[0],v,1))),np.log10((1+spacing)*np.abs(fn_omega_n(vec_l[-1],v,1))),400)

def xylo_scan(l):
    vec_theta_sens_plot = np.zeros(vec_omega_plot.shape)
    omega_0 = np.abs(fn_omega_n(l,v,1)); 
    omega_res = fn_omega_res(l,v,h_11,c_11,beta_11);
    C_c = N_series**-1 * fn_C_crystal_TE_c(l,a,b,beta_11)
    C_1_fid = np.real(C_c)
    L_1_fid = 1/(omega_0**2 * np.real(C_c))
    L_2 = np.real(L_i)
    vec_L_1 = [L_1_fid * np.logspace(-2,2,int(1e4)), L_1_fid * np.logspace(0.0,4,int(1e4))]
    vec_C_1 = C_1_fid * np.asarray([0.1,30])
    vec_L_2 = np.ones((len(vec_L_1[0]),len(vec_C_1))) * np.nan;
    arr_omega_opt = np.ones((len(vec_L_1[0]),len(vec_C_1))) * np.nan; 
    for j,C_1 in enumerate(vec_C_1):
        for i,L_1 in enumerate(tqdm(vec_L_1[j])):
            L_1_n =  L_1 * (1 - 1j * eps_L_1)
            try:
                optimise = L2_find(T, l,v,a,b,h_11,c_11,beta_11,k2,L_squid, R_squid, L_i, k_i, C_1, L_1_n, k_f, eta, S_flux_squid, T_2, mu_N, n_N, xi_11, zeta_11, P_nuc, N_series,N_parallel,j, L_2)
                if optimise[1] > 0:
                    arr_omega_opt[i,j] = optimise[1]
                    L_2 = optimise[0]
                    vec_L_2[i,j] = L_2
                else:
                    pass
            except ValueError:
                pass
    fn_L_1_down = interp1d(omega_0 * arr_omega_opt[np.isfinite(arr_omega_opt[:,0]),0],vec_L_1[0][np.isfinite(arr_omega_opt[:,0])],bounds_error=False,fill_value=np.nan)
    fn_L_1_up = interp1d(omega_0 * arr_omega_opt[np.isfinite(arr_omega_opt[:,1]),1],vec_L_1[1][np.isfinite(arr_omega_opt[:,1])],bounds_error=False,fill_value=np.nan)    
    fn_L_2_down = sp.interpolate.interp1d(omega_0 * arr_omega_opt[np.isfinite(arr_omega_opt[:,0]),0],vec_L_2[:,0][np.isfinite(arr_omega_opt[:,0])],bounds_error=False,fill_value=np.nan)
    fn_L_2_up = sp.interpolate.interp1d(omega_0 * arr_omega_opt[np.isfinite(arr_omega_opt[:,1]),1],vec_L_2[:,1][np.isfinite(arr_omega_opt[:,1])],bounds_error=False,fill_value=np.nan)
    vec_omega_branch_down = omega_0 * np.linspace(np.nanmin(arr_omega_opt[:,0]),np.nanmax(arr_omega_opt[:,0]),int(N_shots/2))
    vec_omega_branch_up = omega_0 * np.linspace(np.nanmin(arr_omega_opt[:,1]),np.nanmax(arr_omega_opt[:,1]),int(N_shots/2))
    vec_L_1_branch_down = fn_L_1_down(vec_omega_branch_down)
    vec_L_1_branch_up = fn_L_1_up(vec_omega_branch_up)
    vec_L_1_branch = np.concatenate([vec_L_1_branch_down,vec_L_1_branch_up])
    vec_L_2_branch_down = fn_L_2_down(vec_omega_branch_down)
    vec_L_2_branch_up = fn_L_2_up(vec_omega_branch_up)
    vec_L_2_branch = np.concatenate([vec_L_2_branch_down,vec_L_2_branch_up])
    for i,L_1 in enumerate(tqdm(vec_L_1_branch)):
        L_1_n = L_1 * (1 - 1j * eps_L_1)
        L_2 = vec_L_2_branch[i]
        if i < N_shots//2:
            C_1 = vec_C_1[0]
        else:
            C_1 = vec_C_1[1]
        vec_theta_sens_shot = fn_theta_sens(t_shot,Q_a,T,vec_omega_plot,l,v,a,b,h_11,c_11,beta_11,k2,L_squid, R_squid,L_i, k_i,C_1,L_1_n,L_2,k_f,eta, S_flux_squid,
                                            T_2,mu_N,n_N,xi_11,zeta_11,P_nuc,N_series,N_parallel)
        if vec_theta_sens_shot[0].all() > 0:
            vec_theta_sens_plot += vec_theta_sens_shot**(-4)
        else:
            print("L_2=",L_2, "C_1=",C_1, "L_1_n=",L_1,end='')
    return vec_theta_sens_plot

ncores = len(vec_l)
pool = multiprocessing.Pool(ncores)

result = pool.map(xylo_scan, vec_l)
resultlist = list(result)
vec_theta_sens_plot = sum(resultlist)**(-1/4)

np.save(dir_data+'cmvec_omega_plot.npy', vec_omega_plot)
np.save(dir_data+'cmvec_theta_sens_plot.npy', vec_theta_sens_plot)

