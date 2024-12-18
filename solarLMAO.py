#Melanie A. Zaidel
#This file contains functions and variables that various Jupyter Notebooks in this repository depend on.
#Mixing parameters and constants are defined here.

import numpy as np
import astropy.constants as const
from astropy import units as u
import pandas as pd
import scipy as sp
#---------------------------------------------------------------------------------------------------------------------------------------

#Set the mixing parameters
delta2_m_12 = 7.58e-5 * u.eV**2 #source: https://arxiv.org/pdf/0801.4589
tan2_theta_12 = 0.436			#source: https://arxiv.org/pdf/1009.4771
sin2_theta_13 = 0.032 			#source: same as above

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

all_SK_observed = np.array([94.1,82.6,80.5,69.7,61.4,54.4,48.3,41.1,35.3,28.7,23.2,18.4,14.3,11.3,8.6,6.17,4.76,3.13,2.07,1.39,1.54,0.57,0.18])/(365 * all_SK_widths)
all_SK_stat_err = np.array([8.35,3.9,2.4,1.45,1.1,1.0,0.9,0.8,0.7,0.6,0.55,0.5,0.4,0.35,0.295,0.245,0.205,0.165,0.135,0.115,0.115,0.075,0.045])/(365 * all_SK_widths)

all_SK_uncorr_sys_err_weights = np.array([4.85, 2.4, 2.3, 1.1, 0.9, 1.2, 1.7, 1.75,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9])/100 #Source: 
all_SK_uncorr_sys_err = all_SK_uncorr_sys_err_weights * all_SK_observed
all_SK_err = np.sqrt(all_SK_stat_err**2 + all_SK_uncorr_sys_err**2)

#Cut the last three bins from 13.49--19.49 MeV to remove contributions from the hep flux:
SK_bin_center = all_SK_bin_center[0:-3]
SK_bin_left = all_SK_bin_left[0:-3]
SK_widths = all_SK_widths[0:-3]

SK_observed = all_SK_observed[0:-3]
SK_err = all_SK_stat_err[0:-3] #ignoring systematic errors
#---------------------------------------------------------------------------------------------------------------------------------------

#Obtain the reference solar model as a DataFrame
def getSolarModel(path):
	"""
	Return a DataFrame containing the reference solar model.

		Parameters:
			path (string): The location + name of the .csv file containing the reference solar model relative to the home directory of this repository.

		Returns:
			solarModel (pandas.core.frame.DataFrame): DataFrame containing the reference solar model.
	"""
	return pd.read_csv(path,names=['m','r','T','rho','P','l','X_H','X_He4','X_He3','X_C12','X_N14','X_O16','X_7Be','nu pp','nu B8','nu N13','nu O15','nu F17','nu Be7','nu pep','nu hep'],skiprows=1,index_col=False)

#Get the solar model and relevant variable profiles
solarModel = getSolarModel("data/Smoothed_Solar_Model.csv")
SSM_r = np.array(solarModel['r'])
SSM_T = np.array(solarModel['T'])
SSM_rho = np.array(solarModel['rho'])
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
	n_e_profile = pd.read_csv(path,names=['r','n_e'])
	r,n_e = np.array(n_e_profile['r']), 10**(np.array(n_e_profile['n_e'])) * const.N_A.value
	return sp.interpolate.interp1d(r,n_e,fill_value=(n_e[0],n_e[-1]),bounds_error=True)

n_r = getElectronNumberDensity("data/n_e.csv")
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
	power_law = (SSM_T)**beta * SSM_r**2
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
	radius_grid = np.linspace(0.0015985,0.3,500) #Smallest radius where electron number density is described, out to 30% of the solar radius.
	integrand = lambda ell: np.cos(2*matterAngle(n_r(ell)*u.cm**-3,energy)).value * zone(ell) #ell = r/R_sun
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

#Initialize theory neutrino and electron energies
recoil_energies = np.linspace(0.01,30,num_points) #implicit units of MeV
neutrino_energies = np.linspace(0.5,35,num_points) #implicit units of MeV

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
	    sol = integrate.simpson(integrand,true_energies)
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
        
		sol = integrate.simpson(integrand[low_index:high_index],neutrino_energies[low_index:high_index])
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
	electron_spectrum = generateSpectrum(beta,doMixing)
	measured_electron_spectrum = smear_numeric_sampled(electron_spectrum,recoil_energies)
	final_spectrum = discretize(recoil_energies,measured_electron_spectrum)*unit_factor
	return final_spectrum
#---------------------------------------------------------------------------------------------------------------------------------------

#Functions related to fitting the neutrino production zone parameter given the observed spectrum in Super-K

#Set the prior on beta
prior_low = 10
prior_high = 20
prior_span = prior_high - prior_low

#Set the grid size of betas to try
num_betas = 200

def log_prior(theta):
    beta = theta
    if prior_low < beta < prior_high:
        return 0.0
    return -np.inf

def log_likelihood(theta):
    beta = theta
    theory = spectrum_model(beta,True)
    sigma2 = SK_err**2
    return -0.5*np.sum((SK_observed - theory)**2 /sigma2 + np.log(sigma2))

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

		data = {'Betas':beta_range.tolist(), 'PDF': P1.tolist(),'Best Fit': beta_max.tolist(), 'err_beta_min': err_beta_min.tolist(),'err_beta_max': err_beta_max.tolist(),'-1sigma':low_bound.tolist(),'+1sigma':high_bound.tolist()}
		df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in data.items()]))
		df.to_csv(filename,index=False)
	else:
		df = pd.read_csv(filename)
		beta_range = df['Betas'].to_numpy()
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
		ax.set_ylim(0,0.4)
		ax.set_xlim(10,20)
		ax.legend(loc='upper right',fontsize=14)
		ax.set_ylabel('Probability Density')
		ax.tick_params(axis='both', pad=7)
		ax.set_xticks([10,15,20])
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
	zone = generate_production_zone(beta)
	norm_zone = zone/np.max(zone)
	sigma_contours = contour_levels(norm_zone,0.68)

	r_max = SSM_r[norm_zone==1.][0]
	err_r_min = np.min(SSM_r[norm_zone>sigma_contours])
	err_r_max = np.max(SSM_r[norm_zone>sigma_contours])
	low_bound = r_max - err_r_min
	high_bound = err_r_max - r_max

	print(f'r = {r_max:>5.5f} - {low_bound:>5.5f}/ + {high_bound:>5.5f}')
	return r_max, low_bound, high_bound


#*%@@@@@@@@@%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%##%#%%%#%%%%#%%%%%#%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#@@%*#%%@%%%%%%%%%%%%%%%%%%%%%%%%%%%%#%##%#########################################%########%%%%%%%%%
#@#+*+====#**%%%%%%%%%%%%%#%%%%%%#%#%###########################################################%%%%%
#%%#*%%**+=-=*#%%%%%%%##%########################################################**:...:**#########%%
#%%%%%%%*--:=.*#%##############################################################*+**=::::-=+#########%
#*%#%*=-..:::..:=########################*#****#*#**#***#************=-=:-=+::+*******+*#**##########
#*=###+=--:::.:::::-*###*#####*******@#*****************************--:::.:::.:-****************#####
#%%##=-=----::.....===*******+==@=:#@*@#************************=****%@@----::+********************##
#%%###*==-----::......:=+***++*@@@@@@@@@**+:**********************@@@@@@@@@+=***********************#
#########*******++-:::-++***#@%@@@@@@@#@@@@%****++++*++++*+++*+++*@@@*@@%@#@+++++++++++++++**********
######*#*************+**##%@@@@@@@@@@@*#*@@@@#*++++++++++++++++++@@@:..J+@@@@++++++++++++++*++*******
####*******+=-...:++*+++#@@@@@@@@@##+==:.#@@@@#++++++++++++++++++@@@=..-=@@@@-..++++++++++++++++*****
#*********=..:::::.=++*#@@@@@@@@@*.. :..:.%@@@##+==========+===++@@@@*...@#@@@:::-+++++++++++++++****
#******=+*+++--=--...:**@@@@@@@@*+:. N @..@@@@@#@==============:-@@@@*.#.@@@@@%+++++++++++++++++++***
#*****==-=+++=++:.....+@@@@@@@@@.=....==..@@@@**#=====:-=....:===@@@@@@@%%@@@@%+====++++++++++++++***
#********+++++++:----@@@@*** @@#@@+:.  :-@@@@@@-=+=--:-:---:.....%@@@@@@@@@@@@.  .====+++++++++++++*+
#*********++++++++++%@%%%%#*#  @%#@@@+-:%@@@@%#=.:=:....    .:--=#@@@@@@*@@@@@@. .======+++++=-=...::
#******+++++++++++++  %%%##*@ +#@@@@@@@@@@@@@@*-=-:...........-:::*@@@@@@@@@@@@@...=======+++=::::::=
#*****+++++++++===        +*@#@%@%@@@@@@@@@@@@*-=---::::-:-----*. .%@@@@@@@@@@@@-..=======++=++++====
#***+++++++++===+ ..         #@*%@@@@@@@@@%@@++=---------------=.  .@@@@@@@@@@@@+..:========+++==++**
#**+++++++======@*.          %%*%@*@@#@@%@@#@*----------------=:.   @@@@@@@=@@@@+...========+++++++**
#++++-:=+======#@@@.%:       .@ ##@@-*@@%@*-*-:----------:-----..   @@+@@@@:+%@@@...==========+++++**
#++++=::..=====@@@@@%###**        #---#@-*@-:::::::::::::::::::..   @@%@%@@@@#%:*...:=========+=++++*
#++++++======-@@@@@@@###***   .    .:-+---+::::::::::::::::::::..  .:::-:%--::..:-...===========+++++
#+++++======-   @@@@@@%%#**#-       :::::--:::::####:.#+::::.:::. ..-*%+:-%*@#@=-%+..:-==========++++
#++++=======.  ..::       *#@@      :..::::::::###.    .#:....:*. .:+-=#+*+**@*+#%-..:-======::+===++
#++========.   ..:         @@###%@@%#:::::::::%%#=:*+K.+:#:::::+.  :@++====+++@*++@..:-------=======+
#=======---.   ..:.        -@###@@%@@.....:::%%@#.+......#::::::.. .@@%**++++*@###*+..------====++#*%
#=======--@@@%...:.         @@%%@@@% :::::::-@@@%:.  *+  *#:::::=-. @@@*+#@@@@@*@%#@..*+++++*****##%@
#=======-@@@@@@@--@@@%%%#@#  .=%%#: . ======%@@@@*-.=###.@#======:. *#*+=++***@*%@%@..***#**#**#**###
#****+++@@@+*+%@@@@@@%#####@..==@@%%%@*===++@@@@@#=..   .@%+=====*:..**+++**#%#@@@@...+*++++++*******
#+++*+*@@@@@@@@@@@@@@++%%##@ ===      @@@@+++@@@@**##%%**%#++++===+..-+***#%%@@@@@@.@.:*##+++++++++++
#++++=@@@@@@@@@@@@@.   .   .@@@@.-@@%@.@@@@=-=-#%==++*@@#+++++++*+-...+**###@@@@@#@@+:#%%###@+++++***
#+++*@@@@@@@@@@@@@.        .@@@@@@%%#%@@@*+==---:-+*##@@+-+++++++:.....*@%%%@@@@@@@@@@@@@@%#%%@+++***
#****@@@@@@@@@@@@@@@-:@@@%# @@@@@@%%%%@@###++=:--+**:#@=**=@@@@%. .:.+.###@@@@@@@++@@@%@@@@@@@@@@*#%%
#***@@@@@@@@@@@@@@@@@@@@%@@%@@@@@:...#@###**#++=---.--@#***@@%%@.%.@.+@##@@@@@@@@+++@@@@@@@@%@@%@@@@%
#*##@@@@@@@@@@@@@@@@@@@@@@@#@@@@=....@%@#%#%%@*++=-=-.*@@@+=%%@@@@++@++**@@@@@@@@+++++#@@@@@@@@@@@@@@
##*#@@@@@@@@@@@@@@@@@@@@@@%%@@@*%....=%#@@%@@@%%%**===@@%**+%@@@@@@@#++@@#+@@@@@@++++****@@@@@@@@@@@@
####@@@@@@@@@@@@@@@@@@@@@@%%#@@:+.:::@%%@@*@@@@@@#@+=+=#%#++*@@@@@@##@@@@@@@@@@@@++++****##@@@@@@@@@@
####@@@@@@@@@@@@@@@@@@@@@%@@#+*++:---@@@@@@@@@@@%@@=+**#@%#++@@@@@@@#%@@%@@@@@@@@@%*++++*##*#*@@@@@@@
##%%#@@@@@@@@@@@@@@@@@@@@@@%@#++#+*@@@@@@@@@@@@@@%#***%*%@*+#*@@@@@@%##@@@@@@@@@@@@@@@@@####**##@@@@@
#@%##%@@@@@@@@@@@@@@@@@@@-.....:@@%@@@%@@@%@@@@@@@@@%%#*+*@#%+*@@@@@@%%%%%%@@@@@@@@@@@@@@@@@%#**@%@@@
#*+++##@@@@@@@@@@@@@@@@@.........:@@@@@@@#@@@@@@@@@##%%%%####@*+@@@@@@%%@%@@@@@@@=*%%@@@@@@@@@@@@%%##
##++++++*@@@@@@@@@@@@@@@@-.:.:...:.@@@@@@@@@@@@@%%%#%@@@@####%#**@@@@@@@@@@@@@@@@++=++==%@@@@@@@@@@@@
#@+++++++++@@@@@@@@@@@@@@-.*:@-:@:-=@@@@@@@@@@@@@@@@@@@@@@%%%%%%*#@@@@@@@@@@@@@@@========@*++#@@@@@@@
#@++**+++*+++@@@@@@@@@@@@+:@-+@..+.@@@@@@#+@@@@@@@@@@@@@@@%@@%%@@#@+@@@@@@@@@@@@@========+*++**+***@@
#@**++*++*+++@@@@@@@@@@@@*:@--@+:@.@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@*@@@@@@@@@@@@@@++=====+++++********
#@%+++*+**+++*@@@@@#@@@@@@%@*-@@%@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@%@#*@@@@@@@+*=+*++#*===+==+**++++***
#@@*******+++*@@@@%=--#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@##@@@@@@@@%#*@@@@@@+=****+#**+===++=++++****
#@@*##**#***+*@@@@#--=-==@@@@@@@@@@@@@@@@@@@@@@@@@@@@@*#%@@@@@@@@@@@@@::%@@@@+*##*****+=++*+++***+*+*
#@@###*****#**@@@@*-==+=@@@@@@@@@@@@@@@@@@@@@@@@@@@%##@@@@@@@@@@@@@@@@@*=:@@@++==+****+++++++****+***
#@@@###*##**#*#@@@+===++=@@@@@@@@@@@@@@@@@@@@@%%%%%%@@@@@@@@@@@@@@@@%+@##-:@@=++*##**#*#+****+*+**#*#
#@@@###*##**#*%@@@+==+++@@@@@@@@@@@@@@@@@@@@@@%%%%@@@@@@@@@@@@@@@@@@+::%%@+=+**==*###**#####**#*#%#%#
#@@@%#######%#%@@@++=+++==@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@--:=@@=++++***#%%%%#%######%%%%%
#@@@@##%##%####@@@*++=++==@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@+#%-*%****######*##*##*#*####%%#
#The Greatest Band in the World