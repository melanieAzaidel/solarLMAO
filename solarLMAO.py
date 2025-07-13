#Melanie A. Zaidel
#This file contains functions and variables that various Jupyter Notebooks in this repository depend on.
#Mixing parameters and constants are defined here.

import numpy as np
import astropy.constants as const
from astropy import units as u
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib
#---------------------------------------------------------------------------------------------------------------------------------------

#Plotting parameters
cm = plt.get_cmap('jet')
plt.style.use('style_prof2.mplstyle')
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

default_size = 18
plt.rcParams.update({
    'font.size': default_size,  # This sets the default font size for everything
    'font.weight': 'bold',
    'axes.titlesize': default_size,
    'axes.labelsize': default_size,
    'xtick.labelsize': default_size,
    'ytick.labelsize': default_size,
    'legend.fontsize': default_size,
    'figure.titlesize': default_size,
    'axes.labelweight': 'bold',
    'text.usetex': True,
})

figsize = 5

plt.rcParams.update({
    'axes.linewidth': 1.5,       # Set frame (spine) thickness
    'xtick.major.width': 1.5,    # Set x-axis major tick thickness
    'ytick.major.width': 1.5,    # Set y-axis major tick thickness
    'xtick.minor.width': 1,  # Set x-axis minor tick thickness
    'ytick.minor.width': 1,  # Set y-axis minor tick thickness
    'xtick.major.size': 8,     # Set x-axis major tick length
    'ytick.major.size': 8,     # Set y-axis major tick length
    'xtick.minor.size': 3,     # Set x-axis minor tick length
    'ytick.minor.size': 3,     # Set y-axis minor tick length
})


matplotlib.rcParams['figure.figsize'] = [figsize,figsize]#[10, 10] # for square canvas
matplotlib.rcParams['figure.subplot.left'] = 0
matplotlib.rcParams['figure.subplot.bottom'] = 0
matplotlib.rcParams['figure.subplot.right'] = 1
matplotlib.rcParams['figure.subplot.top'] = 1

dpi = 350
#---------------------------------------------------------------------------------------------------------------------------------------

#Set the mixing parameters
delta2_m_12 = 7.53e-5 * u.eV**2 #source: https://inspirehep.net/literature/1464091
tan2_theta_12 = 0.436			#source: https://inspirehep.net/literature/1464091
sin2_theta_13 = 0.022 			#source: https://inspirehep.net/literature/2838825

#Convert mixing angles
theta_12 = (np.arctan(np.sqrt(tan2_theta_12))*u.rad).to('degree')
theta_13 = (np.arcsin(np.sqrt(sin2_theta_13))*u.rad).to('degree')
#---------------------------------------------------------------------------------------------------------------------------------------

#Set the 8B neutrino flux normalization
phi_tot_8B = 5.25e6 * u.cm**-2 * u.s**-1 #source: https://arxiv.org/pdf/1109.0763
#---------------------------------------------------------------------------------------------------------------------------------------

#Define useful constants
G_F = (1.16639e-5 * u.GeV**-2 * (const.hbar * const.c)**3).to('GeV m^3') #source: https://arxiv.org/pdf/astro-ph/9502003
m_electron = const.m_e.to(u.MeV, u.mass_energy())
m_e = m_electron.value
#---------------------------------------------------------------------------------------------------------------------------------------

#Define the observed spectrum of scattered electrons in Super-K
#source: Table IX of https://arxiv.org/pdf/2312.12907
all_SK_bin_left = np.array([3.49,3.99,4.49,4.99,5.49,5.99,6.49,6.99,7.49,7.99,8.49,8.99,9.49,9.99,10.49,10.99,11.49,11.99,12.49,12.99,13.49,14.49,15.49])
all_SK_bin_center = np.array([3.75,4.25,4.75,5.25,5.75,6.25,6.75,7.25,7.75,8.25,8.75,9.25,9.75,10.25,10.75,11.25,11.75,12.25,12.75,13.25,14,15,17.5])
all_SK_widths = np.array([0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,1,1,4])

day_SK_observed = np.array([100.2, 77.5, 76.9, 67.2, 62.7, 54.8, 48.0, 41.4, 34.0, 27.8, 23.0, 18.3, 13.6, 11.7, 7.99, 6.20, 4.87, 2.96, 1.81, 1.34, 1.55, 0.59, 0.16])/(365 * all_SK_widths) #day
day_SK_stat_err = np.array([13.15,5.9,3.55,2.1,1.6,1.45,1.3,1.2,1.05,0.9,0.8,0.7,0.6,0.5,0.42,0.35,0.3,0.24,0.19,0.16,0.17,0.11,0.07])/(365 * all_SK_widths) #day

#Cut the last three bins from 13.49--19.49 MeV to remove contributions from the hep flux:
SK_bin_center = all_SK_bin_center[0:-3]
SK_bin_left = all_SK_bin_left[0:-3]
SK_widths = all_SK_widths[0:-3]

SK_observed = day_SK_observed[0:-3]
SK_err = day_SK_stat_err[0:-3] 
#---------------------------------------------------------------------------------------------------------------------------------------

#Obtain the reference solar model variables of interest
solarModelPath = 'data/Individual Variable Profiles/'
SSM_r = np.array(pd.read_csv(solarModelPath+'T.csv',names=['r','q'],skiprows=1,index_col=False)['r'])
SSM_T = np.array(pd.read_csv(solarModelPath+'T.csv',names=['r','q'],skiprows=1,index_col=False)['q'])
SSM_rho = np.array(pd.read_csv(solarModelPath+'rho.csv',names=['r','q'],skiprows=1,index_col=False)['q'])
#---------------------------------------------------------------------------------------------------------------------------------------

#Get the number density of electrons profile and turn it into a scipy interpolated function
#This profile is sourced from the same model above

def getElectronNumberDensity(path):
	"""
	Returns a scipy function containing the interpolated number density of electrons as a function of solar radius.

		Parameters:
			path (string): The location + name of the .csv file containing electron number density profile relative to the home directory of this repository.

		Returns:
			n_r (scipy.interpolate._interpolate.interp1d): Scipy function which takes fractions of the solar radius and returns the electron number density at those fractions.
			Valid domain: 0.0015985 <= r/R_sun <= 1.0005108
			Contains no Astropy units

	"""
	n_e_profile = pd.read_csv(path,names=['r','n_e'],skiprows=1)
	r,n_e = np.array(n_e_profile['r']), 10**(np.array(n_e_profile['n_e'])) * const.N_A.value
	return sp.interpolate.interp1d(r,n_e,fill_value=(n_e[0],n_e[-1]),bounds_error=True)

#n_r = getElectronNumberDensity("data/n_e.csv")
n_r = getElectronNumberDensity("data/Individual Variable Profiles/log10(n_eN_A).csv")
#---------------------------------------------------------------------------------------------------------------------------------------

#Get the 8B neutrino production zone as calculated using the above solar model
def get8BProductionZone(path):
	"""
	Returns the unnormalized 8B neutrino production zone as calculated using the reference solar model and the nuclear reaction rate expression.
	See "Tutorial-Production.ipynb" for how this is calculated and the source of the file.

		Parameters:
			path (string): The location + name of the .csv file containing the nuclear reaction rate prediction of the 8B production zone relative 
						   to the home directory of this repository.

		Returns:
			df_8B_zone (pandas.core.frame.DataFrame): DataFrame containing the 8B production zone in units of cm^-1 s^-1.
			Columns are labeled ['Radius','Rate'].
	"""
	return pd.read_csv(path,names=['Radius','Rate'],skiprows=1,index_col=False)
	
#---------------------------------------------------------------------------------------------------------------------------------------

#Functions related to neutrino mixing

def zeta(n_e,E_nu):
	"""
	Returns zeta as as function of electron number density and neutrino energy.

		Parameters:
			n_e (astropy.units.quantity.Quantity): Number density of electrons with astropy units
				Example: n_e = 1e20 * u.cm**-3

			E_nu (astropy.units.quantity.Quantity): Neutrino energy with astropy units
				Example: E_nu = 1 * u.MeV

		Returns:
			zeta (numpy.float64): Unitless zeta for use in calculating the matter angle.

	"""
	z = 2*np.sqrt(2)*G_F*E_nu*n_e/ delta2_m_12
	return z.decompose()

def matterAngle(n_e,E_nu):
	"""
	Returns the matter angle as a function of electron number density and neutrino energy

		Parameters:
				n_e (astropy.units.quantity.Quantity): Number density of electrons with astropy units
					Example: n_e = 1e20 * u.cm**-3

				E_nu (astropy.units.quantity.Quantity): Neutrino energy with astropy units
					Example: E_nu = 1 * u.MeV

		Returns:
			theta (numpy.float64): Matter angle in implicit units of radians.
	"""
	lamb = np.sin(2*theta_12)**2
	phi = zeta(n_e,E_nu) - np.cos(2*theta_12)
	fraction = -phi/np.sqrt(lamb + phi**2)
	trig = np.arccos(fraction)
	return 0.5*trig

def generate_production_zone(beta):
	"""
	Returns a normalized neutrino production zone profile as a function of beta.

		Parameters:
			beta (float): Temperature scaling parameter

		Returns:
			production_zone (scipy.interpolate._interpolate.interp1d): Scipy function describing the normalized neutrino production 
																	   zone for a given beta as a function of fractions of the solar radius

	"""
	power_law = (SSM_T/1e9)**beta * SSM_r**2 #scale temperature by 1e9 to prevent overflow in exponent
	norm = sp.integrate.simpson(power_law, SSM_r)
	normalized_power_law = power_law/norm

	return sp.interpolate.interp1d(SSM_r,normalized_power_law,bounds_error=False,fill_value=(0,0))

def source_term(E_nu, zone):
	"""
	Returns the source term in the survival probability as a function of neutrino energy for a given neutrino production zone.

		Parameters:
			E_nu (astropy.units.quantity.Quantity): Neutrino energy with astropy units
					Example: E_nu = 1 * u.MeV 

			zone (scipy.interpolate._interpolate.interp1d): Scipy function describing the normalized neutrino production 
																	   zone for a given beta as a function of fractions of the solar radius
		
		Returns:
			source (float): Unitless source term for use in calculating the survival probability
	"""
	radius_grid = np.linspace(0.00161,0.3,500) #Smallest radius where electron number density is described, out to 30% of the solar radius.
	integrand = lambda ell: np.cos(2*matterAngle(n_r(ell)*u.cm**-3,E_nu)).value * zone(ell) #ell = r/R_sun
	return sp.integrate.simpson(integrand(radius_grid),radius_grid)

def p_ee(E_nu, zone):
	"""
	Returns the 3-flavor electron neutrino survival probability as a function of neutrino energy for a given neutrino production zone.

		Parameters:
			E_nu (astropy.units.quantity.Quantity): Neutrino energy with astropy units
					Example: E_nu = 1 * u.MeV 

			zone (scipy.interpolate._interpolate.interp1d): Scipy function describing the normalized neutrino production 
															zone for a given beta as a function of fractions of the solar radius

		Returns:
			prob (float): Unitless electron neutrino survival probability
			
	"""
	source = source_term(E_nu,zone)
	detector = np.cos(2*theta_12)
	two_flavor = 0.5 * (1 + source*detector)
	return np.cos(theta_13)**4 * two_flavor + np.sin(theta_13)**4
#---------------------------------------------------------------------------------------------------------------------------------------

#Functions and variables relevant for calculating the spectrum of scattered electrons in Super-K

num_points = 500 #Grid size for evaluating differential cross sections and survival probability.
				 #Do not change without re-generating differential cross sections.
				 #See 'Tutorial-Detection.ipnyb' for more information.

unit_factor = 86400 * 7.521e33 / 22.5 #1 day = 86400 seconds
									  #7.521e33 electrons in Super-K
									  #22.5 kton of water in Super-K

def getBoron8Spectra(path):
	"""
	Returns the energy spectrum and cutoff energy of neutrinos produced by the decay of 8B.

		Parameters:
				path (string): The location + name of the .csv file containing the 8B neutrino energy spectrum relative 
							   to the home directory of this repository.
							   Source: Table IV in https://arxiv.org/pdf/nucl-ex/0406019

		Returns:
				f_8B (scipy.interpolate._interpolate.interp1d): Scipy function describing the 8B neutrino energy spectrum
																in implicit units of MeV^-1, as a function of neutrino
																energy in implicit units of MeV.

				E_max_8B (float): Cutoff energy of the 8B neutrino spectrum in implicit units of MeV.


	"""
	df_boron8 = pd.read_csv(path,names=['Energy','dNdE'])
	na_boron8_energy = np.array(df_boron8['Energy'])
	na_boron8_flux = np.array(df_boron8['dNdE'])
	return sp.interpolate.interp1d(na_boron8_energy,na_boron8_flux,fill_value=(0,0),bounds_error = False),na_boron8_energy[-1]

f_8B,E_max_8B = getBoron8Spectra('data/f_8Boron.csv')

def getDifferentialCrossSections(path):
	"""
	Returns the differential cross sections for electron/muon/tau neutrino-electron scattering as functions of true electron recoil energy and neutrino energy
	See "Tutorial-Detection.ipynb" for details on differential cross sections and how to generate them.

		Parameters:
			path (string): The location (only) of the .csv file containing the differential cross sections relative 
						   to the home directory of this repository.
		
		Returns:
			differential_cross_sections_e (numpy.ndarray): 2D numpy array containing dSigmadT for electron neutrino-electron scattering.

			differential_cross_sections_mu_tau (numpy.ndarray): 2D numpy array containing dSigmadT for muon/tau neutrino-electron scattering.
	"""
	df_e = pd.read_csv(path+'differential_cross_sections_e.csv', index_col=0)
	df_mu_tau = pd.read_csv(path+'differential_cross_sections_mu_tau.csv', index_col=0)
	return df_e.to_numpy(), df_mu_tau.to_numpy()
    
#Intialize differential cross sections
differential_cross_sections_e, differential_cross_sections_mu_tau = getDifferentialCrossSections('data/')

#Initialize theory neutrino and electron energies
recoil_energies = np.linspace(0.1,25,num_points) #implicit units of MeV
neutrino_energies = np.linspace(0.5,30,num_points) #implicit units of MeV

def resolution(T):
	"""
	Returns the energy resolution in Super-K as a function of true electron kinetic energy.
	Source of formula: Equation 9 in https://arxiv.org/pdf/2312.12907
	Super-K quotes resolution in terms of total reconstructed (true) electron energy, I
	equivalently parameterize it as kinetic + rest.

		Parameters:
			T (float): True electron recoil energy in implicit units of MeV

		Returns:
			sigma (float): Energy resolution in Super-K
	"""
	return -0.05525 + 0.3162*np.sqrt(T + m_e) + 0.04572*(T + m_e)

def resolvingGaussian(T_measured,T_true):
	"""
	Returns a Gaussian window for use in applying the effects of energy resolution to the prompt
	8B neutrino spectrum.

		Parameters:
			T_measured (float): Measured electron recoil energy in implicit units of MeV 

			T_true (numpy.ndarray): 1D array of true kinetic energies in implicit units of MeV

		Returns:
			G (numpy.ndarray): Gaussian smearing window
	"""
	sigma = resolution(T_true)
	return 1/(sigma*np.sqrt(2*np.pi)) * np.exp(- (T_true - T_measured)**2 / (2*(sigma)**2))

def smear_numeric_sampled(true_spectrum,true_energies):
	"""
	Returns a spectrum of scattered electrons accounting for effects due to energy resolution
	given a true spectrum.

		Parameters:
			true_spectrum (numpy.ndarray): 1D array containing the "continuous" true spectrum.

			true_energies (numpy.ndarray): 1D array containing the true energies along which the
										   true spectrum is defined.

		Returns:
			measured_rates (numpy.ndarray): 1D array containing the calculated measured spectrum
											after taking into account energy resolution effects.
	"""
	measured_energies = np.array(true_energies,copy=True)
	measured_rates = []
	for T_measured in measured_energies:
		G = resolvingGaussian(T_measured,true_energies)
		integrand = true_spectrum * G
		sol = sp.integrate.simpson(integrand,true_energies)
		measured_rates.append(sol)

	return np.array(measured_rates)

def discretize(recoil_energies, measured_rates):
	"""
	Returns the discretized scattered electron spectrum after energy resolution
	effects, binned according to Super-K's data presentation.

		Parameters:
			recoil_energies (numpy.ndarray): 1D array containing measured electron kinetic 
											 energies in imlicit units of MeV.

			measured_rates (numpy.ndarray): 1D array containing the calculated measured spectrum
											after taking into account energy resolution effects.

		Returns:
			discrete_rates (numpy.ndarray): 1D array containing the discretized electron spectrum
	"""
	rate_interp = sp.interpolate.interp1d(recoil_energies, measured_rates)
	discrete_rates = []
	delta = 0
	for i, energy in enumerate(SK_bin_left):
		sol = sp.integrate.romberg(rate_interp, energy+delta,energy+SK_widths[i]+delta) /SK_widths[i]
		discrete_rates.append(sol)
	return np.array(discrete_rates)

def generateSpectrum(beta,doMixing):
	"""
	Returns the true theory spectrum of scattered electrons in Super-K due to 8B neutrinos produced
	in the solar core, sans unitful prefactors.

		Parameters:
			beta (float): Temperature scaling parameter controlling the neutrino production zone.

			doMixing (boolean): Whether or not to introduce neutrino mixing physics while calculating 
								the spectrum of scattered electrons. 

								If False, survival probabilities are set to unity for all neutrino 
								energies, and the choice of beta does not enter.

								If True, survival probabilities are calculated using the MSW effect 
								and reactor mixing parameters given by KamLAND. The choice of beta
								will change the neutrino production zone and thus probabilities.

		Returns:
			rates (numpy.ndarray): 1D array containing the true theory spectrum.


	"""
	zone = generate_production_zone(beta)

	if doMixing == True:
		zone = generate_production_zone(beta)
		probabilities = []
		for E_nu in neutrino_energies:
			probabilities.append(p_ee(E_nu*u.MeV,zone))
		probabilities = np.array(probabilities)
	elif doMixing == False:
		probabilities = np.ones_like(neutrino_energies)

	flux = phi_tot_8B * f_8B(neutrino_energies) 
    
	rates = []
	for i,T in enumerate(recoil_energies):
		if T >= E_max_8B:
			rates.append(0)
			continue

		integrand = flux*(differential_cross_sections_e[i]*probabilities + differential_cross_sections_mu_tau[i]*(1-probabilities))
		E_nu_min = (T + np.sqrt(2*m_e*T + T**2))/2
		lower_bound_array = np.absolute(neutrino_energies - E_nu_min)
		upper_bound_array = np.absolute(neutrino_energies - E_max_8B)
		low_index = lower_bound_array.argmin()
		high_index = upper_bound_array.argmin()
		if low_index >= high_index:
			rates.append(0)
			continue
        
		sol = sp.integrate.simpson(integrand[low_index:high_index],neutrino_energies[low_index:high_index])
		rates.append(sol)
	return np.array(rates)


def spectrum_model(beta,doMixing):
	"""
	Returns the theoretical prediction for the spectrum of scattered electrons in Super-K.

	Parameters:
			beta (float): Temperature scaling parameter controlling the neutrino production zone

			doMixing (boolean): Whether or not to introduce neutrino mixing physics while calculating 
								the spectrum of scattered electrons. 

								If False, survival probabilities are set to unity for all neutrino 
								energies, and the choice of beta does not enter.

								If True, survival probabilities are calculated using the MSW effect 
								and reactor mixing parameters given by KamLAND. The choice of beta
								will change the neutrino production zone and thus probabilities.

		Returns:
			final_spectrum (numpy.ndarray): 1D array containing the measured, discretized, unitful
											prediction for the spectrum of scattered electrons.
	"""
	electron_spectrum = generateSpectrum(beta,doMixing) #Generate the prompt spectrum
	measured_electron_spectrum = smear_numeric_sampled(electron_spectrum,recoil_energies) #Apply effects due to energy resolution
	final_spectrum = discretize(recoil_energies,measured_electron_spectrum)*unit_factor #Discretize according to Super-K and combine with prefactor
	return final_spectrum
#---------------------------------------------------------------------------------------------------------------------------------------

#Functions related to fitting the neutrino production zone parameter given the observed spectrum in Super-K

#Set the prior on beta
prior_low = 10
prior_high = 40
prior_span = prior_high - prior_low

#Set the grid size of betas to try
num_betas = 200

def log_prior(theta):
    beta = theta
    if prior_low-0.1 < beta < prior_high+0.1:
        return 0.0
    return -np.inf

def log_likelihood(theta):
    beta = theta
    theory = spectrum_model(beta,True)
    sigma2 = SK_err**2
    return -0.5*np.sum((SK_observed - theory)**2 /sigma2 + np.log(2*np.pi*sigma2))

def log_probability(theta):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta)

def contour_levels(grid,sigma):
    _sorted = np.sort(grid.ravel())[::-1]
    pct = np.cumsum(_sorted) / np.sum(_sorted)
    cutoffs = np.searchsorted(pct, np.array(sigma))
    return _sorted[cutoffs]

def fitBeta(filename, generate=True, show=True):
    """
    Performs a fit for beta, the neutrino production zone parameter, given the observed spectrum
    of scattered electron in Super-K.
    
        Parameters:
            filename (string): Location + name of the fit results to be stored/extracted, relative to
                               the home directory of this repository.
    
            generate (boolean): Whether or not to generate a new fit. Set to True (default) to generate a new
                                fit or overwrite an existing fit. Set to False to get an already existing fit.
    
            show (boolean): Whether or not to plot the fit results. Set to True (default) to create and save a
                            plot. Plot saves to the "figures/"" directory. Set to False to do nothing.
    
        Regardless of the values of the above booleans, this function will grab a fit result from file and
        return the primary characteristics of the fit.
    
        Returns:
            beta_max (float): The best-fit value of beta.
    
            low_bound (float): The -1sigma lower bound from the best-fit of beta.
    
            high_bound (float): The +1sigma upper bound from the best-fit of beta.
    
    """
    if generate == True:
        nll = lambda beta: log_probability(beta)
        beta_range = np.linspace(prior_low,prior_high,num_betas)
        log_P1 = [nll(beta_val) for beta_val in beta_range]
        #print(log_P1)
        log_P1_1 = log_P1 - np.max(log_P1)
        
        P1 = np.exp(log_P1 - np.max(log_P1))
        sigma_contours = contour_levels(P1,0.68)
        
        # Find the max likelihood and the contours
        beta_max = beta_range[P1==1.][0]
        err_beta_min = np.min(beta_range[P1>sigma_contours])
        err_beta_max = np.max(beta_range[P1>sigma_contours])
        low_bound = beta_max - err_beta_min
        high_bound = err_beta_max - beta_max
        
        data = {'Beta':beta_range.tolist(), 'PDF': P1.tolist(),'Best Fit': beta_max.tolist(), 'err_beta_min': err_beta_min.tolist(),'err_beta_max': err_beta_max.tolist(),'-1sigma':low_bound.tolist(),'+1sigma':high_bound.tolist()}
        df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in data.items()]))
        df.to_csv(filename,index=False)
    else:
        df = pd.read_csv(filename)
        beta_range = df['Beta'].to_numpy()
        P1 = df['PDF'].to_numpy()
        beta_max = df['Best Fit'].to_numpy()[0]
        err_beta_min = df['err_beta_min'].to_numpy()[0]
        err_beta_max = df['err_beta_max'].to_numpy()[0]
        low_bound = df['-1sigma'].to_numpy()[0]
        high_bound = df['+1sigma'].to_numpy()[0]
        sigma_contours = contour_levels(P1,0.68) #1 sigma uncertainties on beta

    if show == True:
        norm = sp.integrate.simpson(P1,beta_range)
        fig, ax = plt.subplots(1, 1, figsize=(figsize, figsize),dpi=dpi)
        ax.plot(beta_range,P1/norm,'-k',lw=1.5)
        ax.set_xlabel(r'$\beta$') 
        ax.fill_between(beta_range, P1/norm, where=P1>=sigma_contours, interpolate=True, alpha=0.3,label=r'1$\sigma$ limits')
        ax.set_ylim(0,0.075)
        ax.set_xlim(prior_low,prior_high)
        ax.legend(loc='upper right',fontsize=14)
        ax.set_ylabel('Probability Density',labelpad=1)
        ax.tick_params(axis='x', pad=7)
        ax.tick_params(axis='y', pad=2)
        ax.set_xticks([10,15,20,25,30,35,40])
        plt.tight_layout()
        ax.set_box_aspect(1)
        fig.savefig('figures/fitResult.pdf')
        plt.show()
    print(f'beta = {beta_max:>5.5f} - {low_bound:>5.5f}/ + {high_bound:>5.5f}')
    return beta_max, low_bound, high_bound

def determineNuProductionExtent(beta):
	"""
	Determines and returns the radial extent of 68% (1 sigma) of neutrino production for a given beta.

		Parameters:
			beta (float): Temperature scaling parameter controlling the neutrino production zone

		Returns:
			r_max (float): The fraction of the solar radius corresponding to the peak of neutrino production.

			low_bound (float): The -1sigma lower bound from the peak of production.

			high_bound (float): The +1sigma upper bound from the peak of production.
	"""
	zone = generate_production_zone(beta)(SSM_r)
	norm_zone = zone/np.max(zone)
	sigma_contours = contour_levels(norm_zone,0.68)

	r_max = SSM_r[norm_zone==1.][0]
	err_r_min = np.min(SSM_r[norm_zone>sigma_contours])
	err_r_max = np.max(SSM_r[norm_zone>sigma_contours])
	low_bound = r_max - err_r_min
	high_bound = err_r_max - r_max

	print(f'r = {r_max:>5.5f} - {low_bound:>5.5f}/ + {high_bound:>5.5f}')
	return r_max, low_bound, high_bound

#---------------------------------------------------------------------------------------------------------------------------------------
#Functions related to the differential cross sections

fsc = const.alpha #Fine structure constant
G_F_natural = 1.16639e-5 * u.GeV**-2 #Fermi constant in natural units
sin_2_W = 0.2317 #Weinberg angle
rho_NC = 1.0126 #Neutral current parameter

#Astropy conversion rules to go from natural units to astronomy units and back
cross_section = [(u.GeV**-3, u.cm**2 * u.GeV**-1, lambda x: x * 0.3894 * 1e-27, lambda x: x / (0.3894 * 1e-27))] #GeV^-2 <--> 0.3894 mb, 1 mb <--> 1e-27 cm^2

def dSigma_dT_corrections(flavor, E_nu, E_e):
    """
    Computes and returns the differential scattering cross sections between neutrino flavors and electrons according to radiative corrections and QED effects as a function of incident scattered electron energy, taking incident neutrino energy as a parameter. Source: https://arxiv.org/pdf/astro-ph/9502003

        Parameters:
            flavor (string): The neutrino flavor, must be 'e' or 'mu/tau'.
            E_nu (astropy.units.quantity.Quantity): Incident neutrino energy with astropy units.
            E_e (astropy.units.quantity.Quantity): Scattered electron energies with astropy units.

        Returns:
            dSigma_dT (float): Differential scattering cross section in implicit units of cm^2 MeV^-1. 
    """
    #E_nu is neutrino energy
    #E_e is electron kinetic energy
    z_val = E_e/E_nu
    max_z = (2*E_nu)/(2*E_nu + m_electron)

    if z_val.value >= max_z.value:
        return 0
            
    #print(z)
    xsc = 2*G_F_natural**2 * m_electron/np.pi * (g_L(E_e,flavor)**2 * (1 + fsc/np.pi * fminus(z_val,E_nu)) + g_R(E_e,flavor)**2 * (1-z_val)**2 * (1 + fsc/np.pi * fplus(z_val,E_nu))\
                                                 - g_R(E_e,flavor)*g_L(E_e,flavor)*m_electron*z_val/E_nu * (1 + fsc/np.pi * fpm(z_val,E_nu)))

    return xsc.to(u.cm**2 /u.MeV, equivalencies=cross_section).value

def fminus(z,q):
    #QED term
    T = z*q
    E = T + m_electron
    l = np.sqrt(E**2 - m_electron**2)
    beta = (l/E).value
    #print(l,beta)
    out = (E/l * np.log((E + l)/m_electron) - 1) * (2 * np.log(1 - z - m_electron/(E + l)) - np.log(1 - z) - 0.5*np.log(z) - 5/12)\
    + 0.5*(-sp.special.spence(1 - z.value) + sp.special.spence(1 - beta)) - 0.5*np.log(1-z)**2 - (11/12 + z/2)*np.log(1-z)\
    + z*(np.log(z) + 0.5*np.log(2*q/m_electron)) - (31/18 + 1/12 * np.log(z))*beta - 11*z/12 + z**2 /24
    return out

def fplus(z,q):
    #QED term
    T = z*q
    E = T + m_electron
    l = np.sqrt(E**2 - m_electron**2)
    beta = (l/E).value
    out = (E/l * np.log((E + l)/m_electron) - 1) * ((1-z)**2 * (2*np.log(1 - z - m_electron/(E + l)) - np.log(1 - z) - np.log(z)/2 - 2/3) - 0.5*(z**2 * np.log(z) + 1 - z))\
    - 0.5*(1-z)**2 * (np.log(1-z)**2 + beta * (-sp.special.spence(1-(1-z.value)) - np.log(z)*np.log(1-z))) \
    + np.log(1-z) * (0.5*z**2 * np.log(z) + (1-z)/3 * (2*z - 0.5)) + 0.5*z**2 * sp.special.spence(1-(1-z.value)) - z*(1-2*z)/3 * np.log(z) - z*(1-z)/6 \
    - beta/12 * (np.log(z) + (1-z)*(115 - 109*z)/6)
    return out

def fpm(z,q):
    #QED term
    T = z*q
    E = T + m_electron
    l = np.sqrt(E**2 - m_electron**2)
    out = 2*(E/l * np.log((E+l)/m_electron) - 1) * np.log(1 - z - m_electron/(E+l))
    return out

def g_R(T,flavor):
    #Radiative correction term
    if flavor == 'e':
        return -rho_NC*kappa(T,'e')*sin_2_W
    if flavor == 'mu/tau':
        return -rho_NC*kappa(T,'mu/tau')*sin_2_W

def g_L(T,flavor):
    #Radiative correction term
    if flavor == 'e':
        return rho_NC*(0.5 - kappa(T,flavor)*sin_2_W) - 1
    if flavor == 'mu/tau':
        return rho_NC*(0.5 - kappa(T,flavor)*sin_2_W)
        
def eye(T):
    #Radiative correction term
    x = np.sqrt(1 + 2*m_electron/T)
    return 1/6 * (1/3 + (3-x**2)*(0.5*x*np.log((x+1)/(x-1))-1))

def kappa(T,flavor):
    #Radiative correction term
    if flavor == 'e':
        return 0.9791 + 0.0097*eye(T)
    if flavor == 'mu/tau':
        return 0.9970 - 0.00037*eye(T)

#------------------------------------------------------------------------------------------------------------------------------------------------------
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#:::::::::::::::::::::::::::::::::::::::::::::#:::-::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#::::::::::::::::::::::::::::::::::::::::::+::-::::-::==:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#::::::::::::::::::::::::::::-=::-:::::::::-*+:::+-+--=::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#::::::::::::::::::::::::::::-%*##@#@%#:-+#@%%@@%@%---:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#:::::::::::::::::::::=:+=#*+%%*%%@@@@@@@@@%@@@@@%===%#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#::::::::::::::::::::::--=#%#%@@@@@@@@@@@@@@@@@@@%@%#*--:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#::::::::::::::::::::-::++%@%@@%@@@@@#@@@@@@@@@@@@@%@@#-*-:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#:::::::::::::::::::::-+#%%%@@@@%@@@@@@@@%@@@@@@@@@@@%*-%::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#::::::::::::::::::::::%*#@%@@@@%%@@%%@@%@@@@@@@@@@@@@@=-:=::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#::::::::::::::::::::#-*%@@@@@@@@@@@@@%%@%#%@%@@@@@@@@@*%=:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#::::::::::::::::::::-##%#@@@@@@@@@@@%%#=%+@%@@@@@@@@@@%@=:-:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#::::::::::::::::::::=:%*%@@@@@%@@@@@+@@%---@%%@@@@@@@@#%#-::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#:::::::::::::::::::-:=+#@@@@@@@%@%%*%=%----*#@%@@@@@@@@#%+::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#:::::::::::::::::::::=%*@@@@@@@@*=---:::::-:-=*@@@@@@%@+--+-::::::::::::::::::::::::::::::::::::::=:::::::::::::@:::::::::::::=:::::::::::::::::::::::
#::::::::::::::::::::==*%@@@@@@@%+:::::::::-:::-%@@@@@@%+%*::::::::::::::::::::::::::::::::::::::::=:@@::@*+@:*@:@:%@:@@::@@:::=:::::::::::::::::::::::
#::::::::::::::::::::-*%@@@@@@@@##-::::::::::::-@@@@@@%%%--::::::::::::::::::::::::::::::::::::::::=:@@%@@*+@:*@:@:@%@@@:@@%%::=:::::::::::::::::::::::
#:::::::::::::::::::-#%%%%@%@@@@@@-::::::::-:::#@@@@@@@=*=:::::::::::::::::::::::::::::::::::::::::=:@:@+@*:@@@#:@:%%:@@:@::%+:=:::::::::::::::::::::::
#:::::::::::::::::::=+#@@@@@@@@@@%-::::-+--+=:+@@@@@%@%*:-+::::::::::::::::::::::::::::::::::::::::=:::::::::::::@:::::::::::::=:::::::::::::::::::::::
#::::::::::::::::::-=#*@@@@%@@@%=-::-:::-----*@@@@@@@%#%:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#::::::::::::::::::::@@@@%=#%%-+:::::--::::-@@@@@@@@@**@+::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#:::::::::::::::::::%+--===###*-:::::::--+*%@@#%@@=@@@*+=::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#::::::::::::::::::===+=+++++**+:::::::--=#*+**+=*@@#:*::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#::::::::::::::::-=++****++*+##**#:::::::::::*++*+++=#--:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#:::::::::::::::-=++*****#****###+::::::::---**=*#***++::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#:::::::::::::::=+******+##**#*###+::*-::::-*+++######**+::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#:::::::::::::::=*+#*#****##*#**###*:+**::#*++*#######*#*+:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#::::::::::::::-=***+*#**+######*####*#####*+*#######*###*=::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#::::::::::::::=+####******#**####+%%#*#**+#**######+##*@*:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#::::::::::::::==#####*++*+***###***####+###########%#***#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#:::::::::::::=+=##*#***++=****#*##***#############+**%#*#*::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#:::::::::::::=*+**##**++*=*+*##**##*#**#########*#%#%**+++::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#:::::::::::::=#*+#***++=*=********#*##**#########*##+***+*::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#:::::::::::::=**+#+***+=*+****##****####**#######+*+*#***#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#::::::::::::-=***+#=#**=*+*****#****#%##**+######%#++*#*#+::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#::::::::::::==***=#*=*+=*+*#+*#***+*####+*#+*###*%**%#**#%::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#::::::::::::=+***+*+*===**+*++**#*=*###+*#*#+###*#*#*###*#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::-:--:-----
#-----------------------:--::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#:::::::::::::::::::::::::::::::#%@@%@@@@#@@#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::-%####*=::::::::::::::::::::::::::::::
#:::::::::::::::::::::::::::::%@@@@@@@#@@@%%%%@=::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::@@#@@@@#*@+**::::::::::::::::::::::::::::
#:::::::::::::::::::::::::::%@@@@%@#%+*=@@%@@@#@+:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#@#+#@@@%@*#=+*%-::::::::::::::::::::::::::
#:::::::::::::::::::::::::*:@@@@@=:::::::-=%@@@@@#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::@-:::::-*#+%+++%*@-:::::::::::::::::::::::::
#:::::::::::::::::::::::::#@#@@@@%:::::::::#%@%@@@#:::::::::::::::::::::::::::::::::::::::::::::::::::::::@-:::::::=*#+#*=+@%@-::::::::::::::::::::::::
#::::::::::::::::::::::-#*@@@@@@@@:::::::::=@%@@@@@::::::::::::::::::::::::::::::::::::::::::::::::::::::@%=::::::::=*+#*++%#%*::::::::::::::::::::::::
#::::::::::::::::::::::@@@@@@@@@+-=*:::::::-@#@@@@@:::::::::::::::::::::::::::::::::::::::::::::::::::::@@**+=::-=----*+%@@+%@@::::::::::::::::::::::::
#:::::::::::::::::::::-@@@@@@@=---*--::--::-@@@%@@@::::::::::::::::::::::::::::::::::::::::::::::::::::@%%--::::::-++:-**+%@@%@-:::::::::::::::::::::::
#:::::::::::::::::::::::%@@@@%-:::::-::----=@@@@@@:::::::::::::::::::::::::::::::::::::::::::::::::::::=@@:::::::::::::-*%@@@@@@:::::::::::::::::::::::
#::::::::::::::::::::::%@@@@@=--::::-:--:::-=@@@@@::::::::::::::::::::::::::::::::::::::::::::::::::::::%@::=::::::::::-=*@@@@@@-::::::::::::::::::::::
#::::::::::::::::::::::-#@@@@@--:::-+-=::::-@@@@@@@#::::::::::::::::::::::::::::::::::::::::::::::::::::%%---:::::::::----@%@%@@#::::::::::::::::::::::
#:::::::::::::::::::::::=@@@@@=-----:::::--=@@@@@@@+::::::::::::::::::::::::::::::::::::::::::::::::::##+%---:--::::::-**@%@*%#@%*:::::::::::::::::::::
#:::::::::::::::::::::::+@@@@@@------------%@@@@@@@::::::::::::::::::::::::::::::::::::::::::::::::::-@%-@@-::::::::-=*%@@@@@@@@@##::::::::::::::::::::
#::::::::::::::::::::::::-@@@@@@--:::-----#@@@@@@@@:::::::::::::::::::::::::::::::::::::::::::::::::::@%@%@@*----==--:%@%@*@@@@%%@*::::::::::::::::::::
#::::::::::::::::::::::::::@@@@@+=-----=+-=@%@@@@@=--------:::::::::::::::::::::::::::::::::::::::::::*+%%@@@:=@=--::::=@@@@@@@@%@+::::::::::::::::::::
#:::::::::::::::::::::::::::::=++===----::-*@+*%-------------::::::::::::::::::::::::::::::::::::::::-=%@+%-+:#@=-::::::*@%%%--@@@-::::::::::::::::::::
#:::::::::::::::::::::::::::-----------::::-------------------:::::::::::::::::::::::::::::::::::::::::-%%@===*%@:-:::::=------==+@::::::::::::::::::::
#::::::::::::::::::::::::::------------::::--------------------::::::::::::::::::::::::::::::::::::::::::=*=--+==:---::=--::::---==+:::::::::::::::::::
#::::::::::::::::::::::--------------::::::-=-::----------------::::::::::::::::::::::::::::::::::::::::+=----=-----::=--:::::::--=+-::::::::::::::::::
#:::::::::::::::::::---------------:-::::::----------------------::::::::::::::::::::::::::::::::::::::=-:::%=-------+---:::::::--+=:::::::::::::::::::
#::::::::::::::::::=---------:---+--::::::-=---------------------::::::::::::::::::::::::::::::::::::::-:::#-------==-:--:::::::-===-::::::::::::::::::
#:::::::::::::::::----------:-*-----::::::------------------------::::::::::::::::::::::::::::::::::::--:-=---=---==::--:::::::--==+*::::::::::::::::::
#::::::::::::::::--------=-----------:::::-------------------------::::::::::::::::::::::::::::::::::--::===---=*==-::--:::::::---=+*::::::::::::::::::
#:::::::::::::::-------------------+-::::::------------------------::::::::::::::::::::::::::::::::::--:======--=-===:--:::::::----=*::::::::::::::::::
#::::::::::::::---------------------=::::::------------------------=::::::::::::::::::::::::::::::::--::-==-====-=====---::::::---==*::::::::::::::::::
#:::::::::::::-----:------------------:::::--------------=---------=::::::::::::::::::::::::::::::::--:=======--======--::::::----++=::::::::::::::::::
#:::::::::::::----------------------:-:::::--------------=----------:::::::::::::::::::::::::::::::=-:=+===-===---==-+--::::::----+*:::::::::::::::::::
#::::::::::::-------------------------:::::--------------=-----------::::::::::::::::::::::::::::::---+*++++++=--=-====-:::::::--+=+:::::::::::::::::::
#:::::::::::---------=---------------::::::::------==+---=----------=:::::::::::::::::::::::::::::=--:=##****++========-::::::--=+*::::::::::::::::::::
#::::::::::----------=---------------:::::::::-----------=--------=-=::::::::::::::::::::::::::::=--::-+##*##**++*+*==--::::::--%@@*:::::::::::::::::::
#:::::::::---------------------------:::::::::-----------=----------=::::::::::::::::::::::::::::--:::-=#:@@%%%@@%*+=+--:::::--=@@@::::::::::::::::::::
#::::::::-------------+-----------=--:::::::::-----------=-------=--=:::::::::::::::::::::::::::=-::::-=::*******#%#%*--:::::--%@@:::::::::::::::::::::
#::::::--------------::----=-------------:::::-----------+=---------=:::::::::::::::::::::::::::--::---:::--+-+##***#=--::::--=@*:%::::::::::::::::::::
#::::::*-----------*:::--------------:::::::::-----------+----------=::::::::::::::::::::::::::--:::--=:::---=--::::---:::::--*+*:#::::::::::::::::::::
#::::--------------::::=-----------=-:-:::::::-----------+=---------=::::::::::::::::::::::::::------=::::=------::----::::--=+*:::+:::::::::::::::::::
#------------------------------------------------------------------------------------------------------------------------------------------------------
#The Greatest Band in the World