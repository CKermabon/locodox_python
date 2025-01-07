import matplotlib.pyplot as plt
from m_argo_data import read_argo_data
from m_NCEP_read import read_NCEP
from m_fonctions import watervapor
import numpy as np
import xarray as xr
import warnings
warnings.filterwarnings("ignore",category=RuntimeWarning)
from scipy.optimize import curve_fit

if __name__ == "__main__":
	rep_argo_data = '/Users/chemon/ARGO_NEW/NEW_LOCODOX/data_test/'
	rep_NCEP_data = '/Users/chemon/ARGO_NEW/LOCODOX/DATA/LOCODOX_EXTERNAL_DATA/NCEP/'
	num_float = '6902818'
	z0q = 1e-4
	ds_argo_inair, ds_argo_inwater,ds_argo_Sprof, optode_height, launch_date = read_argo_data(num_float,rep_argo_data)
	print(optode_height) 
	#
	# On garde les cycles communs entre InWater et InAIR.
	#
	cycles_communs = xr.DataArray(np.intersect1d(ds_argo_inair['CYCLE_NUMBER'], ds_argo_inwater['CYCLE_NUMBER']), dims='N_CYCLE')
	ds_argo_inair = ds_argo_inair.where(ds_argo_inair['CYCLE_NUMBER'].isin(cycles_communs), drop=True)
	ds_argo_inwater = ds_argo_inwater.where(ds_argo_inwater['CYCLE_NUMBER'].isin(cycles_communs), drop=True)

	#
	# Lecture NCEP et Interpolation sur positions/dates ARGO
	#
	ds_NCEP_air,ds_NCEP_slp,ds_NCEP_rhum = read_NCEP(rep_NCEP_data,ds_argo_inair['LONGITUDE'],ds_argo_inair['LATITUDE'],ds_argo_inair['JULD'])

	bid=watervapor(ds_argo_inwater['TEMP'],ds_argo_inwater['PSAL'])
	SSph20 = bid * 1013.25 #mbar, seasurface water vapor pressure
	ncep_phum = watervapor(ds_NCEP_air['air'],0) * ds_NCEP_rhum['rhum']/100*1013.25 #ncep water vapor pressure
	ncep_phum_optode_height = (SSph20 + (ncep_phum - SSph20) * np.log(np.abs(optode_height)*100/z0q)/np.log(10/z0q))
	ncep_Po2 = (ds_NCEP_slp['slp'] - ncep_phum_optode_height) * 0.20946
                    
	#
	# Figures/Traces
	#
	plt.figure()
	plt.plot(ncep_Po2,'*b')
	plt.plot(ds_argo_inair['PPOX_DOXY'],'r+')

	#
	# Calcul Gain
	#
	Gain = np.mean(ncep_Po2/ds_argo_inair['PPOX_DOXY']).values
	print(f'Gain a appliquer : {Gain}') 

	#
	# Calcul Gain et Derive.
	#
	PPOX1 = ds_argo_inair['PPOX_DOXY'].values

	def model(delta_T, G, D):
		return G * (1 + D / 100 * delta_T/365) * PPOX1

	# Ajustement des paramètres G et D
	initial_guess = [1, 0]  # Valeurs initiales pour G et D
	delta_T = (ds_argo_inair['JULD'].values - launch_date)
	delta_T = delta_T.astype(float)
	delta_T = delta_T/1e9/86400 # Difference en jour
	params, covariance = curve_fit(model, delta_T, ncep_Po2, p0=initial_guess)

	# Recuperation 
	Gain_estime, Drift_estime = params
	print(f'Gain a appliquer : {Gain_estime} avec Derive {Drift_estime}') 

	plt.show()
