from .atm_tools import (Calcul_T_ant_1_el, Calcul_T_ant_cumulative)
import matplotlib.pyplot as plt
import numpy as np
import pycraf
from pycraf import conversions as cnv
from astropy import units as u
from scipy.interpolate import UnivariateSpline
from scipy.integrate import simpson
from scipy.integrate import trapezoid, cumulative_trapezoid

pi = np.pi

def generate_Pwater_MC(P_water_ref, N_MC, rel_sigma, rng=None):
    """
    Génère un tableau (N_MC, ...) de P_water autour de P_water_ref
    en supposant une loi normale centrée avec écart-type = rel_sigma * P_water_ref.
    """
    rng = np.random.default_rng() if rng is None else rng

    P_water_ref = np.asarray(P_water_ref)
    sigma = rel_sigma * np.where(P_water_ref==0, 1.0, P_water_ref)

    # Tirages Gaussiens
    draws = rng.normal(loc=P_water_ref, scale=sigma, size=(N_MC,) + P_water_ref.shape)


    return draws  # <--- maintenant c'est un seul array de shape (N_MC, *P_water_ref.shape)

def generate_Pwater_MC_lognormal(P_water_ref, N_MC, rel_sigma, rng=None):
    """
    Génère un tableau (N_MC, ...) de P_water > 0 autour de P_water_ref
    en supposant une loi log-normale avec CV = rel_sigma (peut être un scalaire ou un array).
    
    Parameters
    ----------
    P_water_ref : array-like
        Profil de P_water de référence (>0), shape (n_alt,) ou similaire.
    N_MC : int
        Nombre de tirages Monte-Carlo.
    rel_sigma : float or array-like
        Incertitude relative (CV). Peut être un scalaire ou un array de même shape que P_water_ref.
    rng : np.random.Generator, optional
        Générateur pseudo-aléatoire NumPy.
    
    Returns
    -------
    draws : ndarray
        Tableau de shape (N_MC, *P_water_ref.shape) avec les tirages Monte-Carlo log-normaux.
    """
    rng = np.random.default_rng() if rng is None else rng

    P_water_ref = np.asarray(P_water_ref, dtype=float)
    rel_sigma = np.asarray(rel_sigma, dtype=float)

    # Coefficient de variation (évite divisions par 0)
    cv = np.maximum(rel_sigma, 1e-12)

    # Paramètres log-normaux
    sigma2_ln = np.log(1.0 + cv**2)
    sigma_ln = np.sqrt(sigma2_ln)
    mu_ln = np.log(np.where(P_water_ref > 0, P_water_ref, 1e-20)) - 0.5 * sigma2_ln

    # Tirages log-normaux
    draws = rng.lognormal(mean=mu_ln, sigma=sigma_ln,
                          size=(N_MC,) + P_water_ref.shape)

    return draws

def monte_carlo_t_ant(f,theta_b, N, elev, MC_law, N_MC, WVMR, s_WVMR, Temperature, Pressure, z) :
    "WVMR and s_WVMR in g/kg, Pressure in hPa, Temperature in K, z in m, elev in degree, f in Hz, theta_b in rad"

    SNR_rand  = WVMR/s_WVMR
    spline = UnivariateSpline(z[1:], SNR_rand[1:], s=1000)

    snr_smooth_part = spline(z[1:])

    # Réinsérer la première valeur brute
    snr_smooth = np.insert(snr_smooth_part, 0, SNR_rand[0])

    rel_sigma_WVMR = 1/snr_smooth

    WVMR_MC = MC_law(WVMR, N_MC, rel_sigma_WVMR, rng=None)

    
    R_d_air = 287 #J/(kg*K)
    R_water = 462 #J/(kg*K)


    rho_water_MC = Pressure*100*WVMR_MC/(1000*R_d_air*Temperature)*1/(1+R_water/R_d_air*WVMR_MC/1000) #en kg/m3
    P_water_MC = rho_water_MC*R_water*Temperature/100 #en hPa

    Tant_MC = np.zeros((N_MC, len(f)))  # tableau de sortie

    for i in range(N_MC):
        P_i = P_water_MC[i, :]  # réalisation i de P_water
        """print(P_i)"""
        Tant_MC[i] = Calcul_T_ant_1_el(f, theta_b, z,Temperature, Pressure, P_i, elev, N)
        
    

    rho_water = Pressure*100*WVMR/(1000*R_d_air*Temperature)*1/(1+R_water/R_d_air*WVMR/1000) #en kg/m3
    

    P_water = rho_water*R_water*Temperature/100 #en hPa

    T_ant = Calcul_T_ant_1_el(f, theta_b, z, Temperature, Pressure, P_water, elev, N)

    #plt.figure(figsize=(6,5))    
    #plt.hist(Tant_MC[:,0], bins=50, density=True, alpha=0.7)
    #plt.axvline(np.mean(Tant_MC[:, 0]), color='green', linestyle='--', label = 'mean')
    #plt.axvline(T_ant[0], color='red', linestyle='--', label = 'ref')
    #plt.xlabel(r'$T_{\mathrm(ant)}$ K)')
    #plt.ylabel('Density')
    #plt.legend()
    #plt.show()
    #from pathlib import Path
    
    #out_dir = Path("/Users/vl284796/Documents/Images_for_Latex")

    #plt.savefig(out_dir / "t_ant_90_2_mc_hist.pdf", bbox_inches="tight")

    # Écart-type initial
    std_Tant = np.std(Tant_MC[:, 0], ddof=1)
    return std_Tant, T_ant


import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Any
from scipy.optimize import least_squares




"""
def calcul_snr (A, N0, aR, aw, WVMR, z, elev) :
    #z en m, WVMR en g/kg
    z_rel = z - z[0]
    H = 8000.0

    comp_IR_Iw = compute_IR_IW(z,WVMR,H,z[0]-0.001,assume_sorted=True)

    IR, Iw = comp_IR_Iw["IR"], comp_IR_Iw["Iw"]

    snr_associe = snr_from_params_2(A, N0, aR, aw, z_rel, IR, Iw, elev)
    
    return snr_associe
"""

import numpy as np
from scipy.integrate import cumulative_trapezoid

def fit_snr_model(z, SNR, P, T, WVMR, z0=None):
    """
    Fit des paramètres A et c du modèle :
    
        SNR(z) = (A / (z - z0)) * sqrt(WVMR(z) * P(z) / T(z)) * exp(-c * ∫[z0->z] P/T dz')
    
    Paramètres
    ----------
    z, SNR, P, T, WVMR : np.ndarray 1D de même taille
    z0 : float ou None
        Altitude de référence pour l'intégrale.
        Si None, on prend z0 = z[0] et on enlève le premier point (où z=z0).
    
    Retour
    ------
    A, c : floats
        Paramètres ajustés du modèle.
    SNR_model : np.ndarray
        SNR théorique reconstruit sur tout le profil (même taille que SNR).
    mask_fit : np.ndarray bool
        Masque des points effectivement utilisés pour le fit.
    """

    z = np.asarray(z)
    SNR = np.asarray(SNR)
    P = np.asarray(P)
    T = np.asarray(T)
    WVMR = np.asarray(WVMR)

    # 1) Choix de z0
    if z0 is None:
        z0 = z[0]

    # 2) Calcul des quantités intermédiaires (sur tout le profil)
    # Phi(z) = 1/(z - z0) * sqrt(WVMR * P / T)
    with np.errstate(divide='ignore', invalid='ignore'):
        Phi = 1.0 / (z - z0) * np.sqrt(WVMR * P / T)

    # I(z) = ∫[z0->z] P/T dz  (intégrale cumulative par trapèzes)
    integrand = P / T
    I = cumulative_trapezoid(integrand, z, initial=0.0)

    # 3) Construction du masque de points "propres" pour le fit
    #    - z > z0 pour éviter la division par 0
    #    - SNR > 0 et Phi > 0 (log bien défini)
    #    - valeurs finies (pas de NaN/inf)
    mask_fit = (
        (z > z0) &
        (SNR > 0) &
        (Phi > 0) &
        np.isfinite(SNR) &
        np.isfinite(Phi) &
        np.isfinite(I)
    )

    if np.sum(mask_fit) < 2:
        raise ValueError("Pas assez de points valides pour effectuer le fit.")

    z_fit = z[mask_fit]
    SNR_fit = SNR[mask_fit]
    Phi_fit = Phi[mask_fit]
    I_fit = I[mask_fit]

    # 4) On passe dans le log :
    # Y = ln(SNR / Phi) = ln(A) - c * I
    Y = np.log(SNR_fit / Phi_fit)

    # 5) Régression linéaire Y = a + b * I  =>  a = ln(A), b = -c
    b, a = np.polyfit(I_fit, Y, 1)
    c_est = -b
    A_est = np.exp(a)

    # 6) Reconstruction de la SNR théorique sur tout le profil
    SNR_model = A_est * Phi * np.exp(-c_est * I)

    return A_est, c_est, SNR_model, mask_fit



def calcul_snr (A, c, WVMR, z, P, T, elev) :
    #z en m, WVMR en g/kg
    
    thet = 90 - elev
    thet_rad = thet * pi/180
    m = 1/(np.cos(thet_rad) + 0.50572*(96.07995-thet)**(-1.6364)) 
    
    snr_associe = A*np.sqrt(WVMR * P / T) / (m*(z - (z[0]-0.01))) * np.exp(-c*m * cumulative_trapezoid(P / T, z, initial=0.0))
    

    return snr_associe

def remove_nans(arr, *others):
    """
    Supprime les éléments NaN de `arr` ainsi que les éléments correspondants
    dans les autres listes ou tableaux de même longueur.

    Paramètres :
        arr : array-like principal
        *others : autres listes/arrays associées (de même longueur)

    Retour :
        arr_clean, autres_listes_clean
    """
    arr = np.asarray(arr)
    others = [np.asarray(o) for o in others]
    assert all(len(o) == len(arr) for o in others), "Toutes les listes doivent avoir la même taille"

    # masque des valeurs valides (non-NaN)
    mask_keep = ~np.isnan(arr)

    # filtrer arr et les autres listes
    arr_clean = arr[mask_keep]
    others_clean = [o[mask_keep] for o in others]

    return (arr_clean, *others_clean)

def Monte_Carlo_T_ant_mod(f,theta_b, N, elev, MC_law, N_MC, WVMR, SNR, Temperature, Pressure, z) :
    "WVMR and s_WVMR in g/kg, Pressure in hPa, Temperature in K, z in m, elev in degree, f in Hz, theta_b in rad"

    

    WVMR_MC = MC_law(WVMR, N_MC, 1/SNR, rng=None)
    
    """print(WVMR_MC)"""

    
    R_d_air = 287 #J/(kg*K)
    R_water = 462 #J/(kg*K)


    rho_water_MC = Pressure*100*WVMR_MC/(1000*R_d_air*Temperature)*1/(1+R_water/R_d_air*WVMR_MC/1000) #en kg/m3
    P_water_MC = rho_water_MC*R_water*Temperature/100  #en hPa

    Tant_MC = np.zeros((N_MC, len(f)))  # tableau de sortie

    for i in range(N_MC):
        P_i = P_water_MC[i, :]  # réalisation i de P_water
        """print(P_i)"""
        Tant_MC[i] = Calcul_T_ant_1_el(f, theta_b, z,Temperature, Pressure, P_i, elev, N)
        
    

    rho_water = Pressure*100*WVMR/(1000*R_d_air*Temperature)*1/(1+R_water/R_d_air*WVMR/1000) #en kg/m3
    

    P_water = rho_water*R_water*Temperature/100  #en hPa

    T_ant = Calcul_T_ant_1_el(f, theta_b, z, Temperature, Pressure, P_water, elev, N)
    """print(T_ant)
    print(Tant_MC)"""

    """plt.figure(figsize=(6,5))    
    plt.hist(Tant_MC[:,0], bins=50, density=True, alpha=0.7)
    plt.axvline(np.mean(Tant_MC[:, 0]), color='green', linestyle='--', label = 'mean')
    plt.axvline(T_ant[0], color='red', linestyle='--', label = 'ref')
    plt.xlabel('T_ant (K)')
    plt.show()"""

    # Écart-type initial
    std_Tant = np.zeros(len(f))
    for i in range (len(f)):
        
        std_Tant[i] = np.std(Tant_MC[:, i], ddof=1)
    return std_Tant, T_ant

#A= 7350529829.075809
#N0 = 1.0244874615346347e-06 
#aR = 0.0003564186913625303
#aw = 9.110808513015025e-09


import numpy as np

def local_bin_width(z):
    """
    z : array (N,), niveaux d'altitude (supposés triés croissant)
    retourne dz_local : array (N,), épaisseur verticale représentée par chaque point
    """
    z = np.asarray(z)
    N = len(z)
    dz = np.empty(N)

    if N == 1:
        dz[0] = 0.0  # cas dégénéré

    elif N == 2:
        dz[0] = z[1] - z[0]
        dz[1] = z[1] - z[0]

    else:
        # bords
        dz[0]  = z[1]   - z[0]
        dz[-1] = z[-1]  - z[-2]
        # milieu
        dz[1:-1] = 0.5 * (z[2:] - z[:-2])
    # Remplace toute valeur nulle par 1 (évite divisions/ratios invalides en aval)
    dz = np.where(dz == 0, 1.0, dz)

    return dz  # en mètres



#valeur de predict SNR sur le fichier de patrick 90°, 50cm mirror
A = 106533.049
c = 1.470303e-05

def scale_snr_for_variable_bins(z, snr30, elev = 90, dz_ref=30.0):
    """
    z : altitudes
    snr30 : SNR calibré pour dz_ref (30 m typiquement), même shape que z
    dz_ref : résolution verticale de référence (m)
    """
    
    thet = 90 - elev
    thet_rad = thet * pi/180
    m = 1/(np.cos(thet_rad) + 0.50572*(96.07995-thet)**(-1.6364))
    dz_local = local_bin_width(z)
    factor = np.sqrt(dz_local*m / dz_ref)
    return snr30 * factor, dz_local


def predict_SNR_T (frequency, theta_b,z,WVMR,elev,T,P,N_MC) :
    N=500
    
    #snr_attendu_ref = calcul_snr (A, N0, aR, aw, WVMR, z, elev) #ancien modèle
    snr_attendu_ref = calcul_snr(A, c, WVMR, z, P, T, elev) #modèle ajusté avec patrick
    
    snr_attendu_ref_t, z_t, WVMR_t, temp_t, P_data_t = remove_nans(snr_attendu_ref, z, WVMR, T, P)
    
    snr_attendu_t = scale_snr_for_variable_bins(z_t, snr_attendu_ref_t, elev, dz_ref=30.0) [0]
    
    simu = Monte_Carlo_T_ant_mod(frequency, theta_b, N, elev, generate_Pwater_MC_lognormal, N_MC, WVMR_t, snr_attendu_t, temp_t, P_data_t, z_t) #pour un miroir de 50cm
    
    return simu



def hybrid_lin_geom(zmin, zbreak, zmax, N_lin=80, N_geom=70, gamma=1.0):
    z1 = np.linspace(zmin, zbreak, N_lin, endpoint=False)
    # partie géométrique au-dessus du breakpoint
    t = np.linspace(0, 1, N_geom)
    z2 = zbreak * (zmax / zbreak)**(t**gamma)
    return np.concatenate([z1, z2])

def Monte_Carlo_T_ant_profile(f, theta_b, N, elev, MC_law, N_MC, WVMR, SNR, Temperature, Pressure, z):
    """
    Calcule :
    1. Le profil de sigma_T (écart-type dû aux fluctuations d'eau)
    2. Le profil de T_ant moyen (avec eau)
    3. Le profil de T_ant Dry (sans eau)
    
    Retourne 3 arrays de taille len(z).
    """
    
    # --- 1. Génération des perturbations (Eau) ---
    # WVMR_MC est une matrice (N_MC, len(z))
    WVMR_MC = MC_law(WVMR, N_MC, 1/SNR, rng=None)
    
    R_d_air = 287 # J/(kg*K)
    R_water = 462 # J/(kg*K)

    # Calcul de la Pression partielle d'eau pour tous les tirages
    rho_water_MC = Pressure*100*WVMR_MC/(1000*R_d_air*Temperature)*1/(1+R_water/R_d_air*WVMR_MC/1000) 
    P_water_MC = rho_water_MC*R_water*Temperature/100  # en hPa

    # --- 2. Calcul du Profil "Sans Eau" (Dry) ---
    # On le fait UNE SEULE FOIS ici pour gagner du temps.
    # On met une pression de vapeur d'eau quasi-nulle (1e-15 pour éviter des logs de zero potentiels)
    P_water_dry = np.full_like(Pressure, 1e-15)
    
    # On suppose que Calcul_T_ant_cumulative renvoie un array (len(z), 1) pour une seule fréq
    res_dry = Calcul_T_ant_cumulative(f, theta_b, z, Temperature, Pressure, P_water_dry, elev, N)
    
    # Gestion des dimensions si f est un scalaire
    if res_dry.ndim > 1 and res_dry.shape[1] == 1:
        T_ant_dry_profile = res_dry[:, 0]
    else:
        T_ant_dry_profile = res_dry
        
        
    #Calcul du profil avec de l'eau

    # --- 3. Boucle Monte Carlo (Atmosphère Humide) ---
    
    Tant_MC_profiles = np.zeros((N_MC, len(z)))

    for i in range(N_MC):
        P_i = P_water_MC[i, :]  # Profil d'eau perturbé
        
        # Calcul cumulatif avec eau
        res = Calcul_T_ant_cumulative(f, theta_b, z, Temperature, Pressure, P_i, elev, N)
        
        # Stockage (flatten si nécessaire)
        Tant_MC_profiles[i, :] = res[:, 0] if res.ndim > 1 else res

    # --- 4. Statistiques ---
    
    # Ecart-type (Sigma_T)
    std_Tant_profile = np.std(Tant_MC_profiles, axis=0, ddof=1)
    
    # Moyenne (T_ant Total)
    mean_Tant_profile = np.mean(Tant_MC_profiles, axis=0)
    
    # Si vous préférez retourner le profil de référence (profil "moyen" sans bruit) au lieu de la moyenne des tirages :
    # Recalculer P_water_ref à partir du WVMR moyen et appeler Calcul_T_ant_cumulative une fois.
    # Ici, je retourne la moyenne des simulations MC.

    return std_Tant_profile, mean_Tant_profile, T_ant_dry_profile



#On essaye de prendre en compte l'incertitude sur la température

def monte_carlo_t_ant_2(f,theta_b, N, elev, MC_law_wvmr, MC_law_temperature, N_MC, WVMR, s_WVMR, Temperature, s_Temperature, Pressure, z) :
    "WVMR and s_WVMR in g/kg, Pressure in hPa, Temperature in K, z in m, elev in degree, f in Hz, theta_b in rad"


    rel_sigma_WVMR = s_WVMR / WVMR

    WVMR_MC = MC_law_wvmr(WVMR, N_MC, rel_sigma_WVMR, rng=None)
    
    rel_sigma_temperature = s_Temperature / Temperature
    
    Temperature_MC = MC_law_temperature(Temperature, N_MC, rel_sigma_temperature, rng=None)

    
    R_d_air = 287 #J/(kg*K)
    R_water = 461.5 #J/(kg*K)


    rho_water_MC = Pressure*100*WVMR_MC/(1000*R_d_air*Temperature_MC)*1/(1+R_water/R_d_air*WVMR_MC/1000) #en kg/m3
    P_water_MC = rho_water_MC*R_water*Temperature_MC/100 #en hPa

    Tant_MC = np.zeros((N_MC, len(f)))  # tableau de sortie

    for i in range(N_MC):
        P_i = P_water_MC[i, :]  # réalisation i de P_water
        T_i = Temperature_MC[i,:]
        """print(P_i)"""
        Tant_MC[i] = Calcul_T_ant_1_el(f, theta_b, z, T_i, Pressure, P_i, elev, N)
        
    

    rho_water = Pressure*100*WVMR/(1000*R_d_air*Temperature)*1/(1+R_water/R_d_air*WVMR/1000) #en kg/m3
    

    P_water = rho_water*R_water*Temperature/100 #en hPa

    T_ant = Calcul_T_ant_1_el(f, theta_b, z, Temperature, Pressure, P_water, elev, N)

    plt.figure(figsize=(6,5))    
    plt.hist(Tant_MC[:,0], bins=50, density=True, alpha=0.7)
    plt.axvline(np.mean(Tant_MC[:, 0]), color='green', linestyle='--', label = 'mean')
    plt.axvline(T_ant[0], color='red', linestyle='--', label = 'ref')
    #plt.xlabel(r'$T_{\mathrm(ant)}$ K)')
    plt.ylabel('Density')
    plt.legend()
    plt.show()
    #from pathlib import Path
    
    #out_dir = Path("/Users/vl284796/Documents/Images_for_Latex")

    #plt.savefig(out_dir / "t_ant_90_2_mc_hist.pdf", bbox_inches="tight")

    # Écart-type initial
    std_Tant = np.std(Tant_MC[:, 0], ddof=1)
    return std_Tant, T_ant

def Monte_Carlo_T_ant_mod_2(f,theta_b, N, elev, MC_law_wvmr, MC_law_temperature, N_MC, WVMR, SNR_wvmr, Temperature, SNR_temperature, Pressure, z) :
    "WVMR and s_WVMR in g/kg, Pressure in hPa, Temperature in K, z in m, elev in degree, f in Hz, theta_b in rad"

    

    WVMR_MC = MC_law_wvmr(WVMR, N_MC, 1/SNR_wvmr, rng=None)
    
    Temperature_MC = MC_law_temperature(Temperature, N_MC, 1/SNR_temperature, rng=None)
    
    """print(WVMR_MC)"""

    
    R_d_air = 287 #J/(kg*K)
    R_water = 461.5 #J/(kg*K)


    rho_water_MC = Pressure*100*WVMR_MC/(1000*R_d_air*Temperature_MC)*1/(1+R_water/R_d_air*WVMR_MC/1000) #en kg/m3
    P_water_MC = rho_water_MC*R_water*Temperature_MC/100  #en hPa

    Tant_MC = np.zeros((N_MC, len(f)))  # tableau de sortie

    for i in range(N_MC):
        P_i = P_water_MC[i, :]  # réalisation i de P_water
        T_i = Temperature_MC[i,:]
        """print(P_i)"""
        Tant_MC[i] = Calcul_T_ant_1_el(f, theta_b, z, T_i, Pressure, P_i, elev, N)
        
    

    rho_water = Pressure*100*WVMR/(1000*R_d_air*Temperature)*1/(1+R_water/R_d_air*WVMR/1000) #en kg/m3
    

    P_water = rho_water*R_water*Temperature/100  #en hPa

    T_ant = Calcul_T_ant_1_el(f, theta_b, z, Temperature, Pressure, P_water, elev, N)
    """print(T_ant)
    print(Tant_MC)"""

    """plt.figure(figsize=(6,5))    
    plt.hist(Tant_MC[:,0], bins=50, density=True, alpha=0.7)
    plt.axvline(np.mean(Tant_MC[:, 0]), color='green', linestyle='--', label = 'mean')
    plt.axvline(T_ant[0], color='red', linestyle='--', label = 'ref')
    plt.xlabel('T_ant (K)')
    plt.show()"""

    # Écart-type initial
    std_Tant = np.std(Tant_MC[:, 0], ddof=1)
    return std_Tant, T_ant

def predict_SNR_T_2 (frequency, theta_b,z,WVMR,elev,T,SNR_temperature, P,N_MC) :
    N=500
    
    #snr_attendu_ref = calcul_snr (A, N0, aR, aw, WVMR, z, elev) #ancien modèle
    snr_attendu_ref = calcul_snr(A, c, WVMR, z, P, T, elev) #modèle ajusté avec patrick
    
    snr_attendu_ref_t, z_t, WVMR_t, temp_t, P_data_t, SNR_temperature_t = remove_nans(snr_attendu_ref, z, WVMR, T, P, SNR_temperature)
    
    snr_attendu_t = scale_snr_for_variable_bins(z_t, snr_attendu_ref_t, elev, dz_ref=30.0) [0]
    
    #pour un miroir de 50cm
    simu = Monte_Carlo_T_ant_mod_2(frequency,theta_b, N, elev, generate_Pwater_MC_lognormal, generate_Pwater_MC, N_MC, WVMR_t, snr_attendu_t, temp_t, SNR_temperature_t, P_data_t, z_t)
    return simu