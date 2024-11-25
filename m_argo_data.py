""" Module dedie a la lecture des donnees ARGO """
# Importation des modules necessaires
import argopy
import xarray as xr
import glob
import numpy as np
import pandas as pd

# Fonction de lecture
def read_argo_data_for_NCEP(num_float,rep_data_argo,which_psal,code_inair,code_inwater):
	""" 
	Cette fonction lit les fichiers Rtraj et Sprof d'un flotteur.
	En entree : 
 		num_float : numero WMO du flotteur
 		rep_data_argo : repertoire ou sont les donnees ARGO
        which_psal : Indique si l'utilisateur souhaite lire PSAL (1) ou PSAL_ADJUSTED (2).
	En sortie :
 		ds_argo_Rtraj_inair : donnees Rtraj dans l'air (InAir) moyennees par cycle
 		ds_argo_Rtraj_inwater : donnees Rtraj InWater moyennees par cycle 
        Dans ces 2 variables, apres disucussion avec VT et CC le 21/11/2024 :
            la temperature est issue de la temperature du fichier Rtraj moyenne par cycle. Pour le calcul de la vapeur d'eau,
            necessaire au calcul de PPOX NCEP, la temperature a un fort impact. On decide donc d'affecter
            les donnnees dans l'air et inwater a la temperature du Rtraj, plus proche de la mesure. 
            On compare cette temperature moyenne par cycle avec la temperature du Sprof dont le QC vaut 1 ou 2 et pour la pression la plus
            proche de la surface entre 0 et 10m. Si les 2 valeurs differe de plus de 0.5 degre, on affiche un message d'alerte.
        
            Pour la salinite, comme pour les donnees inwater et inair, la pompe est arretee. On decide d'affecter a ces donnees la
            valeur de PSAL (ou PSAL_ADJUSTED) issue du fichier Sprof. On prend la salinite dont le QC et 1 ou 2 et associee a la profondeur
            minimale entre O et 10m (ie a la pression la plus proche de la surface entre 0 et 10m).
            
		optode_height : hauteur de l'optode
        launch_date : date de deploiement du flotteur
	"""
	fic_argo_Sprof = glob.glob(rep_data_argo + num_float + '/*Sprof.nc')
	fic_argo_Rtraj = glob.glob(rep_data_argo + num_float + '/*Rtraj.nc')
	fic_argo_meta = glob.glob(rep_data_argo + num_float + '/*meta.nc')
	ds_argo_Rtraj = xr.open_dataset(fic_argo_Rtraj[0],engine='argo')
	ds_argo_Sprof = xr.open_dataset(fic_argo_Sprof[0],engine='argo')
	ds_argo_meta = xr.open_dataset(fic_argo_meta[0],engine='argo')

	# Dans le Sprof, on ne garde que les QC 1 ou 2 pour la salinite.
	if which_psal == 1:
		valid_qc = (ds_argo_Sprof['PSAL_QC']==1) | (ds_argo_Sprof['PSAL_QC']==2)
	else:
		valid_qc = (ds_argo_Sprof['PSAL_ADJUSTED_QC']==1) | (ds_argo_Sprof['PSAL_ADJUSTED_QC']==2)

    	############################################################################################
    	# Recherche de la salinite valide la plus proche de la surface (entre 0 et 10m) dans le fichier Sprof.
    	#
	valid_pres_range = (ds_argo_Sprof.PRES >= 0) & (ds_argo_Sprof.PRES <= 10)

    	# Masque combiné pour valider les 2 conditions (QC et niveau de pression)
	valid_mask = valid_qc & valid_pres_range
    	# On force les pressions non 'valides' a une valeur infinie.
	valid_pres = ds_argo_Sprof.PRES.where(valid_mask, other=np.inf)
    	# recherche des indices associes a la pression minimale 'valide' pour chaque profil
	min_pres_idx = valid_pres.argmin(dim="N_LEVELS")

    	# Extraire les valeurs de PSAL correspondantes
	if which_psal == 1:
		psal_results = ds_argo_Sprof['PSAL'].isel(N_LEVELS=min_pres_idx)
	else:
		psal_results = ds_argo_Sprof['PSAL_ADJUSTED'].isel(N_LEVELS=min_pres_idx)
        
    	# On garde les valeurs de salinite repondant aux criteres de QC et de niveau de pression.   
	psal_results = psal_results.where(valid_pres.min(dim="N_LEVELS") != np.inf)
	cycle_results = ds_argo_Sprof['CYCLE_NUMBER']
	psal_results = psal_results.to_numpy()
	cycle_results = cycle_results.to_numpy()
	############################################################################################

	############################################################################################
	# Recherche de la temperature valide la plus proche de la surface (entre 0 et 10m) dans le fichier Sprof.
	#
	# Dans le Sprof, recherche de la temperature valide la plus proche de la surface.
	# Remarque : Dans le Rtraj, tous les TEMP_QC sont a 3
	# Dans le fichier synthetique, le profil TEMP contient les donnees de profils
	# et du near surface. Les donnees de TEMP_ADJUSTED ne comprennent pas les donnees Near
	# (elles sont vides). On travaille donc sur les donnees TEMP.
	valid_qc = (ds_argo_Sprof['TEMP_QC']==1) | (ds_argo_Sprof['TEMP_QC']==2) | (ds_argo_Sprof['TEMP_QC']==3)

	# Dans le Sprof, on ne garde que les valeurs entre O et 10m.
	valid_pres_range = (ds_argo_Sprof.PRES >= 0) & (ds_argo_Sprof.PRES <= 10)

	# Masque combiné pour valider les 2 conditions (QC et niveau de pression)
	valid_mask = valid_qc & valid_pres_range
	# On force les pressions non 'valides' a une valeur infinie.
	valid_pres = ds_argo_Sprof.PRES.where(valid_mask, other=np.inf)
	# recherche des indices associes a la pression minimale 'valide' pour chaque profil
	min_pres_idx = valid_pres.argmin(dim="N_LEVELS")

	temp_results = ds_argo_Sprof['TEMP'].isel(N_LEVELS=min_pres_idx)
	# On garde les valeurs de salinite repondant aux criteres de QC et de niveau de pression.   
	temp_results = temp_results.where(valid_pres.min(dim="N_LEVELS") != np.inf)

	############################################################################################

	# Lecture de la hauteur de l'optode
	bid = ds_argo_meta['LAUNCH_CONFIG_PARAMETER_NAME'].str.strip()=='CONFIG_OptodeVerticalPressureOffset_dbar'
	optode_height = ds_argo_meta['LAUNCH_CONFIG_PARAMETER_VALUE'][bid].values

	# Lecture de la date de deploiement du flotteur
	launch_date = ds_argo_meta['LAUNCH_DATE'].values

    
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
	ds_argo_Rtraj_inair = ds_argo_Rtraj.where(ds_argo_Rtraj['MEASUREMENT_CODE'].isin(code_inair),drop=True)
	ds_argo_Rtraj_inwater = ds_argo_Rtraj.where(ds_argo_Rtraj['MEASUREMENT_CODE'].isin(code_inwater),drop=True)

	# pour le carry-over, on ne garde que les cycles ou on a des donnees InAir et des donnees InWater.
	cycles_communs = xr.DataArray(np.intersect1d(ds_argo_Rtraj_inair['CYCLE_NUMBER'], ds_argo_Rtraj_inwater['CYCLE_NUMBER']), dims='N_CYCLE')
	ds_argo_Rtraj_inair = ds_argo_Rtraj_inair.where(ds_argo_Rtraj_inair['CYCLE_NUMBER'].isin(cycles_communs), drop=True)
	ds_argo_Rtraj_inwater = ds_argo_Rtraj_inwater.where(ds_argo_Rtraj_inwater['CYCLE_NUMBER'].isin(cycles_communs), drop=True)

	# On calcule la mediane par cycle.
	# Pour moyenner les dates, il faut au prealable les transformer en int.
	# On souhaite garder les lon/lat. On les transforme de coordonnees en variables.
	ds_argo_Rtraj_inair['LATITUDE_ARGO']=('N_MEASUREMENT',ds_argo_Rtraj_inair['LATITUDE'].values)
	ds_argo_Rtraj_inair['LONGITUDE_ARGO']=('N_MEASUREMENT',ds_argo_Rtraj_inair['LONGITUDE'].values)
	ds_argo_Rtraj_inwater['LATITUDE_ARGO']=('N_MEASUREMENT',ds_argo_Rtraj_inwater['LATITUDE'].values)
	ds_argo_Rtraj_inwater['LONGITUDE_ARGO']=('N_MEASUREMENT',ds_argo_Rtraj_inwater['LONGITUDE'].values)
	ds_argo_Rtraj_inair['JULD_INT'] = ('N_MEASUREMENT',ds_argo_Rtraj_inair['JULD'].astype(int).values) 
	ds_argo_Rtraj_inwater['JULD_INT'] = ('N_MEASUREMENT',ds_argo_Rtraj_inwater['JULD'].astype(int).values) 

	# Calcul de la mediane  par cycle
	ds_argo_Rtraj_inair = ds_argo_Rtraj_inair.groupby('CYCLE_NUMBER').median(skipna=True)
	ds_argo_Rtraj_inwater = ds_argo_Rtraj_inwater.groupby('CYCLE_NUMBER').median(skipna=True)
	# On transforme les dates de int en datetime
	ds_argo_Rtraj_inair['JULD']=('CYCLE_NUMBER',pd.to_datetime(ds_argo_Rtraj_inair['JULD_INT'].values))
	ds_argo_Rtraj_inwater['JULD']=('CYCLE_NUMBER',pd.to_datetime(ds_argo_Rtraj_inair['JULD_INT'].values))
	#ds_argo_Rtraj_inair = ds_argo_Rtraj_inair.assign(JULD=pd.to_datetime(ds_argo_Rtraj_inair['JULD_INT'].values))
	#ds_argo_Rtraj_inwater = ds_argo_Rtraj_inwater.assign(JULD=pd.to_datetime(ds_argo_Rtraj_inwater['JULD_INT'].values))

	# On remplace les donnees de PSAL issues de Rtraj avec  la salinite valide 
	# la plus proche de la surface issue du Sprof.
	for i_data in range(ds_argo_Rtraj_inair['PSAL'].size): 
		isok = np.where(cycle_results==ds_argo_Rtraj_inair['CYCLE_NUMBER'][i_data].values)[0]
		if isok.size > 0:
			ds_argo_Rtraj_inair['PSAL'][i_data] = psal_results[isok][0]
			ds_argo_Rtraj_inwater['PSAL'][i_data] = psal_results[isok][0]
		else:
			ds_argo_Rtraj_inair['PSAL'][i_data]=np.nan
			ds_argo_Rtraj_inwater['PSAL'][i_data]=np.nan
            
	if (np.abs(ds_argo_Rtraj_inair['TEMP'][i_data] - temp_results[isok][0]) > 0.5):
		print(f'Cycle {cycle_results[isok][0]}\n \
		La temperature du RTraj differe de plus de 0.5 degre de la temperature du Sprof')

	# On ne garde que les variables utiles.
	ds_argo_Rtraj_inair = ds_argo_Rtraj_inair[['LONGITUDE_ARGO','LATITUDE_ARGO','PPOX_DOXY','TEMP','PSAL','JULD','CYCLE_NUMBER']]
	ds_argo_Rtraj_inwater = ds_argo_Rtraj_inwater[['LONGITUDE_ARGO','LATITUDE_ARGO','PPOX_DOXY','TEMP','PSAL','JULD','CYCLE_NUMBER']]

	return ds_argo_Rtraj_inair, ds_argo_Rtraj_inwater, optode_height, launch_date

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

	# Lecture de la date de deploiement du flotteur
	launch_date = ds_argo_meta['LAUNCH_DATE'].values
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

	# On ne garde que les variables utiles.
	ds_argo_Rtraj_inair = ds_argo_Rtraj_inair[['PPOX_DOXY','TEMP','PSAL','JULD','CYCLE_NUMBER']]
	ds_argo_Rtraj_inwater = ds_argo_Rtraj_inwater[['PPOX_DOXY','TEMP','PSAL','JULD','CYCLE_NUMBER']]

	return ds_argo_Rtraj_inair, ds_argo_Rtraj_inwater, ds_argo_Sprof, optode_height, launch_date

