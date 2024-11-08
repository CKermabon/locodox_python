""" Module dedie a la lecture des donnees ARGO """
# Importation des modules necessaires
import argopy
import xarray as xr
import glob
import numpy as np

# Fonction de lecture
def read_argo_data(num_float,rep_data_argo):
	""" 
	Cette fonction lit les fichiers Rtraj et Sprof d'un flotteur.
	En entree : 
 		num_float : numero WMO du flotteur
 		rep_data_argo : repertoire ou sont les donnees ARGO
	En sortie :
 		ds_argo_Rtraj_inair : donnees Rtraj contenant les donnees PPOX_DOXY dans l'air
 		ds_argo_Rtraj_inwater : donnees Rtraj contenant les donnees PPOX_DOXY proche de la surface
 		ds_argo_Sprof : donnees synthetique
		optode_height : hauteur de l'optode
	"""
	fic_argo_Sprof = glob.glob(rep_data_argo + num_float + '/*Sprof.nc')
	fic_argo_Rtraj = glob.glob(rep_data_argo + num_float + '/*Rtraj.nc')
	fic_argo_meta = glob.glob(rep_data_argo + num_float + '/*meta.nc')
	ds_argo_Rtraj = xr.open_dataset(fic_argo_Rtraj[0],engine='argo')
	ds_argo_Sprof = xr.open_dataset(fic_argo_Sprof[0],engine='argo')
	ds_argo_meta = xr.open_dataset(fic_argo_meta[0],engine='argo')

	# Lecture de la hauteur de l'optode
	bid = ds_argo_meta['LAUNCH_CONFIG_PARAMETER_NAME'].str.strip()=='CONFIG_OptodeVerticalPressureOffset_dbar'
	optode_height = ds_argo_meta['LAUNCH_CONFIG_PARAMETER_VALUE'][bid].values

	# Dans le fichier Rtraj, il n'y a pas de longitude affectee aux donnees PPOX_DOXY dans l'air.
	# On remplit les donnees de LONGITUDE à partir des donnees LONGITUDE des donnees synthetiques
	# en faisant le lien entre les donnees Rtraj et Sprof via le numero de cycle.
	ds_argo_Rtraj =  ds_argo_Rtraj.groupby("CYCLE_NUMBER").apply(
    	 lambda group: group.assign_coords(
        	LONGITUDE=("N_MEASUREMENT",np.full(len(group["N_MEASUREMENT"]),ds_argo_Sprof.LONGITUDE.sel(N_PROF=group.CYCLE_NUMBER[0]).item()))
         )
	)
	ds_argo_Rtraj =  ds_argo_Rtraj.groupby("CYCLE_NUMBER").apply(
    	 lambda group: group.assign_coords(
        	LATITUDE=("N_MEASUREMENT",np.full(len(group["N_MEASUREMENT"]),ds_argo_Sprof.LATITUDE.sel(N_PROF=group.CYCLE_NUMBER[0]).item()))
         )
	)
	# On ne garde que les donnees Rtraj qui nous interessent, à savoir les donnees dans l'air et proche de la surface. 
	ds_argo_Rtraj_inair = ds_argo_Rtraj.where(ds_argo_Rtraj['MEASUREMENT_CODE'].isin([699,711,799]),drop=True)
	ds_argo_Rtraj_inwater = ds_argo_Rtraj.where(ds_argo_Rtraj['MEASUREMENT_CODE'].isin([690,710]),drop=True)

	return ds_argo_Rtraj_inair, ds_argo_Rtraj_inwater, ds_argo_Sprof, optode_height

