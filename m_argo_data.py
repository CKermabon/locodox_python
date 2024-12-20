""" Module dedie a la lecture des donnees ARGO """
# Importation des modules necessaires
import argopy
import xarray as xr
import glob
import numpy as np
import pandas as pd
from m_fonctions import O2ctoO2p
import matplotlib.pyplot as plt

def read_argo_launch_date(num_float,rep_data_argo):
    """ 
    Fonction permettant de recuperer la date de deploiement du flotteur.
    En entree :
        num_float : numero WMO du flotteur
        rep_data_argo : repertoire des donnees ARGO

    En sortie :
        launch_date : date de deploiement du flotteur
    """
    fic_argo_meta = glob.glob(rep_data_argo + num_float + '/*meta.nc')
    ds_argo_meta = xr.open_dataset(fic_argo_meta[0],engine='argo')
    launch_date = ds_argo_meta['LAUNCH_DATE'].values
    return launch_date
    
# Fonctions de lecture des donnees ARGO en vue d'une correction via WOA
def read_argo_data_for_WOA(num_float,rep_data_argo,which_var,temp_qc,sal_qc,doxy_qc):
    """
    Cette fonction lit le fichier Sprof ARGO.
    En entree :
        num_float : numero WMO du flotteur
        rep_data_argo :  repertoire des donnees ARGO
        which_var : Indique si on utilise les donnees ajustees (2) ou les donnees 'temps reel' (1) 
        ou les donnees ajustees si elles existent sinon les donnees 'temps reel' pour le calcul de le % de saturation d'oxygene
        temp_qc,sal_qc, doxy_qc : flag de qualite (QC) a garder pour

    En sortie :
         ds_argo_Sprof : donnees ARGO (lon/lat/time) issues du Sprof utiles a la correction WOA
    """
    fic_argo_Sprof = glob.glob(rep_data_argo + num_float + '/*Sprof.nc')
    ds_argo_Sprof = xr.open_dataset(fic_argo_Sprof[0],engine='argo')

    # On garde uniquement les profils remontee (le premeir profil est souvent un profil descente qui ne commence pas en surface)
    #ds_argo_Sprof = ds_argo_Sprof.where(ds_argo_Sprof['DIRECTION']=='A',drop=True)
    if which_var == 1:
        ds_argo_Sprof['PSAL_ARGO'] = ds_argo_Sprof['PSAL']
        ds_argo_Sprof['PSAL_ARGO_QC'] = ds_argo_Sprof['PSAL_QC']
        ds_argo_Sprof['PRES_ARGO'] = ds_argo_Sprof['PRES']
        ds_argo_Sprof['PRES_ARGO_QC'] = ds_argo_Sprof['PRES_QC']
        ds_argo_Sprof['TEMP_ARGO'] = ds_argo_Sprof['TEMP']
        ds_argo_Sprof['TEMP_ARGO_QC'] = ds_argo_Sprof['TEMP_QC']
    elif which_var == 2:
        ds_argo_Sprof['PSAL_ARGO'] = ds_argo_Sprof['PSAL_ADJUSTED']
        ds_argo_Sprof['PSAL_ARGO_QC'] = ds_argo_Sprof['PSAL_ADJUSTED_QC']
        ds_argo_Sprof['PRES_ARGO'] = ds_argo_Sprof['PRES_ADJUSTED']
        ds_argo_Sprof['PRES_ARGO_QC'] = ds_argo_Sprof['PRES_ADJUSTED_QC']
        ds_argo_Sprof['TEMP_ARGO'] = ds_argo_Sprof['TEMP_ADJUSTED']
        ds_argo_Sprof['TEMP_ARGO_QC'] = ds_argo_Sprof['TEMP_ADJUSTED_QC'] 
    else:
        ds_argo_Sprof['PSAL_ARGO'] = ds_argo_Sprof['PSAL_ADJUSTED']
        ds_argo_Sprof['PSAL_ARGO'] = ds_argo_Sprof['PSAL_ARGO'].where((np.isfinite(ds_argo_Sprof['PSAL_ARGO'])),ds_argo_Sprof['PSAL'])
        ds_argo_Sprof['PSAL_ARGO_QC'] = ds_argo_Sprof['PSAL_ADJUSTED_QC']
        ds_argo_Sprof['PSAL_ARGO_QC'] = ds_argo_Sprof['PSAL_ARGO_QC'].where((np.isfinite(ds_argo_Sprof['PSAL_ADJUSTED'])),ds_argo_Sprof['PSAL_QC'])

        ds_argo_Sprof['PRES_ARGO'] = ds_argo_Sprof['PRES_ADJUSTED']
        ds_argo_Sprof['PRES_ARGO'] = ds_argo_Sprof['PRES_ARGO'].where((np.isfinite(ds_argo_Sprof['PRES_ARGO'])),ds_argo_Sprof['PRES'])
        ds_argo_Sprof['PRES_ARGO_QC'] = ds_argo_Sprof['PRES_ADJUSTED_QC']
        ds_argo_Sprof['PRES_ARGO_QC'] = ds_argo_Sprof['PRES_ARGO_QC'].where((np.isfinite(ds_argo_Sprof['PRES_ADJUSTED'])),ds_argo_Sprof['PRES_QC'])

        ds_argo_Sprof['TEMP_ARGO'] = ds_argo_Sprof['TEMP_ADJUSTED']
        ds_argo_Sprof['TEMP_ARGO'] = ds_argo_Sprof['TEMP_ARGO'].where((np.isfinite(ds_argo_Sprof['TEMP_ARGO'])),ds_argo_Sprof['TEMP'])
        ds_argo_Sprof['TEMP_ARGO_QC'] = ds_argo_Sprof['TEMP_ADJUSTED_QC'] 
        ds_argo_Sprof['TEMP_ARGO_QC'] = ds_argo_Sprof['TEMP_ARGO_QC'].where((np.isfinite(ds_argo_Sprof['TEMP_ADJUSTED'])),ds_argo_Sprof['TEMP_QC'])


    # On garde les temperatures, DOXY et salinites avec un QC corrects. On force les autres valeurs a NaN
    ds_argo_Sprof['TEMP_ARGO'] = ds_argo_Sprof['TEMP_ARGO'].where((ds_argo_Sprof['TEMP_ARGO_QC'].isin(temp_qc)),np.nan)
    ds_argo_Sprof['PSAL_ARGO'] = ds_argo_Sprof['PSAL_ARGO'].where((ds_argo_Sprof['PSAL_ARGO_QC'].isin(sal_qc)),np.nan)
    ds_argo_Sprof['PRES_ARGO'] = ds_argo_Sprof['PRES_ARGO'].where((ds_argo_Sprof['PRES_ARGO_QC'].isin(temp_qc)),np.nan)
    ds_argo_Sprof['DOXY'] = ds_argo_Sprof['DOXY'].where((ds_argo_Sprof['DOXY_QC'].isin(doxy_qc)),np.nan)
    ds_argo_Sprof = ds_argo_Sprof.rename_vars({'DOXY' :'DOXY_ARGO','DOXY_QC':'DOXY_ARGO_QC'})
                                               
    if ds_argo_Sprof['PRES_ARGO'].isnull().all():
        print(f'Attention : Aucune donnees de pression valide ...\nArret')
        return None

    if ds_argo_Sprof['TEMP_ARGO'].isnull().all():
        print(f'Attention : Aucune donnees de temperature valide ...\nArret')
        return None

    if ds_argo_Sprof['PSAL_ARGO'].isnull().all():
        print(f'Attention : Aucune donnees de salinite valide ...\nArret')
        return None
        
    if ds_argo_Sprof['DOXY_ARGO'].isnull().all():
        print(f'Attention : Aucune donnees de O2 valide ...\nArret')
        return None
   
    # On garde les variables utiles au calcul de PSAT.
    ds_argo_Sprof = ds_argo_Sprof[['TEMP_ARGO','TEMP_ARGO_QC','PSAL_ARGO','PSAL_ARGO_QC','DOXY_ARGO','DOXY_ARGO_QC','PRES_ARGO','PRES_ARGO_QC','JULD','LONGITUDE','LATITUDE','CYCLE_NUMBER']]


    return ds_argo_Sprof
    
    
def read_argo_data_for_WOA_old(num_float,rep_data_argo,which_var,qc_psal):
    """
    Cette fonction lit le fichier Sprof ARGO.
    En entree :
        num_float : numero WMO du flotteur
        rep_data_argo :  repertoire des donnees ARGO
        which_var : Indique si on utilise les donnees ajustees (2) ou non (1) pour le calcul de le % de saturation d'oxygene

    En sortie :
         ds_argo_Sprof : donnees ARGO (lon/lat/time) issues du Sprof utiles a la correction WOA
    """
    fic_argo_Sprof = glob.glob(rep_data_argo + num_float + '/*Sprof.nc')
    ds_argo_Sprof = xr.open_dataset(fic_argo_Sprof[0],engine='argo')
    fic_argo_meta = glob.glob(rep_data_argo + num_float + '/*meta.nc')
    ds_argo_meta = xr.open_dataset(fic_argo_meta[0],engine='argo')
    launch_date = ds_argo_meta['LAUNCH_DATE'].values

    # On garde uniquement les profils remontee (le premeir profil est souvent un profil descente qui ne commence pas en surface)
    #ds_argo_Sprof = ds_argo_Sprof.where(ds_argo_Sprof['DIRECTION']=='A',drop=True)
    
    if which_var == 1:
        var_sal = 'PSAL'
        var_temp = 'TEMP'
        var_pres = 'PRES'
    else:
        var_sal = 'PSAL_ADJUSTED'
        var_temp = 'TEMP_ADJUSTED'
        var_pres = 'PRES_ADJUSTED'
    var_sal_qc = var_sal + '_QC'
    var_temp_qc = var_temp + '_QC'
    var_pres_qc = var_pres + '_QC'

    finite_mask = (np.isfinite(ds_argo_Sprof[var_pres].values)) & ((ds_argo_Sprof[var_pres_qc].values == 1) | (ds_argo_Sprof[var_pres_qc].values == 2))
    nb_indices = np.count_nonzero(finite_mask) 
    if nb_indices==0:
        print(f'Attention : Aucune donnees de pression valide ...\nArret')
        return None, None, None, None
        
    finite_mask = (np.isfinite(ds_argo_Sprof[var_temp].values)) & ((ds_argo_Sprof[var_temp_qc].values == 1) | (ds_argo_Sprof[var_temp_qc].values == 2))
    nb_indices = np.count_nonzero(finite_mask) 
    if nb_indices==0:
        print(f'Attention : Aucune donnees de temperature valide ...\nArret')  
        return None, None, None, None

    finite_mask = (np.isfinite(ds_argo_Sprof[var_sal].values)) & ((ds_argo_Sprof[var_sal_qc].values == 1) | (ds_argo_Sprof[var_sal_qc].values == 2))
    nb_indices = np.count_nonzero(finite_mask) 
    if nb_indices==0:
        print(f'Attention : Aucune donnees de salinite valide ...\nArret') 
        return None, None, None, None
        
    # Conversion de DOXY en PPOX
    # On garde les donnees PPOX dont le QC (en temperature/salinite et DOXY) est invalide.
    #ds_argo_Sprof['PPOX_DOXY'] = O2ctoO2p(ds_argo_Sprof['DOXY'],ds_argo_Sprof[var_temp],ds_argo_Sprof[var_sal]) # On calcule PPOX pour P=0
    #ds_argo_Sprof['PPOX_DOXY'] = ds_argo_Sprof['PPOX_DOXY'].where((ds_argo_Sprof[var_temp_qc].isin([1,2,8])),np.nan) # On garde QC temp 1/2/8
    #ds_argo_Sprof['PPOX_DOXY'] = ds_argo_Sprof['PPOX_DOXY'].where((ds_argo_Sprof[var_sal_qc].isin(qc_psal)),np.nan) # On garde QC psal fourni par l'utilisateur
    #ds_argo_Sprof['PPOX_DOXY'] = ds_argo_Sprof['PPOX_DOXY'].where((ds_argo_Sprof['DOXY_QC'].isin([1,2,3,8])),np.nan) # On garde QC DOXY 1/2/3/8

    # Interpolation sur une grille reguliere en pression (definie par l'utilisateur).
    # On interpole les donnees de PPOX ARGO sur une grille reguliere avant de faire la moyenne sur l'intervalle de pression fournie par l'utilisateur.
    # Cela permet d'avoir un poids identique pour chaque pression.
    #ppox_argo = np.empty(shape=(ds_argo_Sprof['N_PROF'].size,max_pres-min_pres+1))
    #new_pres = np.arange(min_pres,max_pres+1,1)
    #for i_cycle in (range(0,len(ds_argo_Sprof['N_PROF']))):
    #    data_en_cours = ds_argo_Sprof['PPOX_DOXY'][i_cycle,:]
    #    isok = np.isfinite(data_en_cours)
    #    nb_indices_ok = np.count_nonzero(isok) 
    #    if nb_indices_ok>0:
    #        ppox_argo[i_cycle] = np.interp(new_pres,ds_argo_Sprof['PRES'][i_cycle,isok],data_en_cours[isok])
    #    else:
    #        ppox_argo[i_cycle] = np.nan
        
    #ppox_argo = np.nanmean(ppox_argo,axis=1)    # On moyenne entre les pressions min_pres/max_pres. On obtient une valeur par cycle.
    
    #ds_argo_Sprof = ds_argo_Sprof.assign_coords({'N_PROF':range(len(ds_argo_Sprof.N_PROF)), 'N_LEVELS':range(len(ds_argo_Sprof.N_LEVELS))})
    #ds_argo_Sprof_interp = ds_argo_Sprof.argo.interp_std_levels(new_pres,axis=var_pres) # Ne conserve que les profils qui vont jusqu'à max(new_pres). Non OK pour nos besoins
    #ds_argo_Sprof_interp = ds_argo_Sprof.interp(new_pres,axis=var_pres)

    # On garde les temperatures, DOXY et salinites avec un QC corrects. On force les autres valeurs a NaN
    ds_argo_Sprof[var_temp] = ds_argo_Sprof[var_temp].where((ds_argo_Sprof[var_temp_qc].isin([1,2,8])),np.nan)
    ds_argo_Sprof[var_sal] = ds_argo_Sprof[var_sal].where((ds_argo_Sprof[var_sal_qc].isin(qc_psal)),np.nan)
    ds_argo_Sprof['DOXY'] = ds_argo_Sprof['DOXY'].where((ds_argo_Sprof['DOXY_QC'].isin([1,2,3,8])),np.nan)
    
    # Conversion de DOXY en PPOX
    # On garde les donnees PPOX dont le QC (en temperature/salinite et DOXY) est invalide.
    ds_argo_Sprof['PPOX_DOXY'] = O2ctoO2p(ds_argo_Sprof['DOXY'],ds_argo_Sprof[var_temp],ds_argo_Sprof[var_sal]) # On calcule PPOX pour P=0
    ds_argo_Sprof['PPOX_DOXY'] = ds_argo_Sprof['PPOX_DOXY'].where((ds_argo_Sprof[var_temp_qc].isin([1,2,8])),np.nan) # On garde QC temp 1/2/8
    ds_argo_Sprof['PPOX_DOXY'] = ds_argo_Sprof['PPOX_DOXY'].where((ds_argo_Sprof[var_sal_qc].isin(qc_psal)),np.nan) # On garde QC psal fourni par l'utilisateur
    ds_argo_Sprof['PPOX_DOXY'] = ds_argo_Sprof['PPOX_DOXY'].where((ds_argo_Sprof['DOXY_QC'].isin([1,2,3,8])),np.nan) # On garde QC DOXY 1/2/3/8
    
    # On garde les variables utiles au calcul de PSAT.
    ds_argo_Sprof = ds_argo_Sprof[[var_temp,var_sal,'DOXY',var_pres,'LONGITUDE','LATITUDE','JULD','PPOX_DOXY']] 
    #ds_argo_Sprof = ds_argo_Sprof.assign_coords({'N_PROF':range(len(ds_argo_Sprof.N_PROF)), 'N_LEVELS':range(len(ds_argo_Sprof.N_LEVELS))})
    #ds_argo_Sprof_interp = ds_argo_Sprof.argo.interp_std_levels(new_pres,axis=var_pres) # Ne conserve que les profils qui vont jusqu'à max(new_pres). Non OK pour nos besoins
    #ds_argo_Sprof_interp = ds_argo_Sprof.interp(new_pres,axis=var_pres)

    ds_argo_Sprof = ds_argo_Sprof.rename_vars({var_temp :'TEMP_ARGO',var_sal:'PSAL_ARGO',var_pres:'PRES_ARGO'})

    return ds_argo_Sprof, launch_date
    #return interpol_data,ds_argo_Sprof['LATITUDE'],ds_argo_Sprof['LONGITUDE'],ds_argo_Sprof['JULD']

# Fonctions de lecture des donnees ARGO en vue d'une correction via NCEP
def read_argo_data_for_NCEP(num_float,rep_data_argo,which_psal,code_inair,code_inwater,min_pres,max_pres):
	""" 
	Cette fonction lit les fichiers Rtraj et Sprof d'un flotteur.
	En entree : 
 		num_float : numero WMO du flotteur
 		rep_data_argo : repertoire ou sont les donnees ARGO
        which_psal : Indique si l'utilisateur souhaite lire PSAL (1) ou PSAL_ADJUSTED (2) ou PSAL_ADJUSTED si existe sinon PSAL (3).
        code_inair : Code associe aux donnees dans l'air dans le fichier Rtraj.nc ARGO (699/711/799)
        code_inwater : Code associe aux donnees InWater dans le fichier Rtraj.nc ARGO (690/710)
        min_pres/max_res : Minimum et Maximum de pression, utilises pour trouver la salinite valide la plus proche de la surface
        
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

	# Recherche de la salinite (PSAL et PSAL_ADJUSTED) correcte la plus proche de la surface
	valid_pres_range = (ds_argo_Sprof.PRES >= min_pres) & (ds_argo_Sprof.PRES <= max_pres)

	var_psal = ['PSAL','PSAL_ADJUSTED']

	for i_var in range(0,len(var_psal)):
		var_en_cours = var_psal[i_var]
		print(f'Recherche de la valeur de {var_en_cours} dans Sprof correcte la plus proche de la surface\
 entre {min_pres} et {max_pres}')
		valid_qc = (ds_argo_Sprof[var_en_cours+'_QC']==1) | (ds_argo_Sprof[var_en_cours+'_QC']==2)

		# Masque combiné pour valider les 2 conditions (QC et niveau de pression)
		valid_mask = valid_qc & valid_pres_range
		# On force les pressions non 'valides' a une valeur infinie.
		valid_pres = ds_argo_Sprof.PRES.where(valid_mask, other=np.inf)
		# Recherche des indices associes a la pression minimale 'valide' pour chaque profil
		min_pres_idx = valid_pres.argmin(dim="N_LEVELS")

		# Extraire les valeurs de PSAL correspondantes, repondant aux criteres de QC et de niveau de pression. 
		if i_var == 0:
			psal_results = ds_argo_Sprof['PSAL'].isel(N_LEVELS=min_pres_idx)
			psal_results = psal_results.where(valid_pres.min(dim="N_LEVELS") != np.inf)
		else:
			psal_adj_results = ds_argo_Sprof['PSAL_ADJUSTED'].isel(N_LEVELS=min_pres_idx)
			psal_adj_results = psal_adj_results.where(valid_pres.min(dim="N_LEVELS") != np.inf)

	# L'utilisateur peut vouloir prendre les donnees PSAL_ADJUSTED si elles existent, sinon il prend PSAL.        
	cycle_results = ds_argo_Sprof['CYCLE_NUMBER']
	psal_results = psal_results.to_numpy()
	psal_adj_results = psal_adj_results.to_numpy()
	psal_mixte_results = psal_adj_results.copy()
	isbad = np.isnan(psal_mixte_results)
	psal_mixte_results[isbad] = psal_results[isbad]
	cycle_results = cycle_results.to_numpy()

	############################################################################################

	############################################################################################
	# Recherche de la temperature valide la plus proche de la surface (entre min_pres et max_pres) dans le fichier Sprof.
	#
	# Dans le Sprof, recherche de la temperature valide la plus proche de la surface.
	# Remarque : Dans le Rtraj, tous les TEMP_QC sont a 3
	# Dans le fichier synthetique, le profil TEMP contient les donnees de profils
	# et du near surface. Les donnees de TEMP_ADJUSTED ne comprennent pas les donnees Near
	# (elles sont vides). On travaille donc sur les donnees TEMP.

	valid_qc = (ds_argo_Sprof['TEMP_QC']==1) | (ds_argo_Sprof['TEMP_QC']==2) | (ds_argo_Sprof['TEMP_QC']==3)

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

	plt.figure()
	plt.plot(ds_argo_Rtraj_inair['CYCLE_NUMBER'],ds_argo_Rtraj_inair['PPOX_DOXY'],'*-b')
	plt.plot(ds_argo_Rtraj_inwater['CYCLE_NUMBER'],ds_argo_Rtraj_inwater['PPOX_DOXY'],'o-r')

	# Calcul de la mediane  par cycle
	ds_argo_Rtraj_inair = ds_argo_Rtraj_inair.groupby('CYCLE_NUMBER').median(skipna=True)
	ds_argo_Rtraj_inwater = ds_argo_Rtraj_inwater.groupby('CYCLE_NUMBER').median(skipna=True)

	plt.plot(ds_argo_Rtraj_inair['CYCLE_NUMBER'],ds_argo_Rtraj_inair['PPOX_DOXY'],'*-c')
	plt.plot(ds_argo_Rtraj_inwater['CYCLE_NUMBER'],ds_argo_Rtraj_inwater['PPOX_DOXY'],'o-m')
	plt.grid()
	_ = plt.legend(['InAir','InWater','MedianInAir','MedianInWater'],loc='upper left',bbox_to_anchor=(1,1))
	plt.xlabel('CYCLE_NUMBER')
	plt.ylabel('PPOX')
	plt.show()

	# On transforme les dates de int en datetime
	ds_argo_Rtraj_inair['JULD']=('CYCLE_NUMBER',pd.to_datetime(ds_argo_Rtraj_inair['JULD_INT'].values))
	ds_argo_Rtraj_inwater['JULD']=('CYCLE_NUMBER',pd.to_datetime(ds_argo_Rtraj_inair['JULD_INT'].values))
	#ds_argo_Rtraj_inair = ds_argo_Rtraj_inair.assign(JULD=pd.to_datetime(ds_argo_Rtraj_inair['JULD_INT'].values))
	#ds_argo_Rtraj_inwater = ds_argo_Rtraj_inwater.assign(JULD=pd.to_datetime(ds_argo_Rtraj_inwater['JULD_INT'].values))


	# On remplace les donnees de PSAL issues de Rtraj avec  la salinite valide 
	# la plus proche de la surface issue du Sprof. La salinite et la temperature sont utilisees pour calculer NCEP PPOX.
	for i_data in range(ds_argo_Rtraj_inair['PSAL'].size): 
		isok = np.where(cycle_results==ds_argo_Rtraj_inair['CYCLE_NUMBER'][i_data].values)[0]
		if isok.size > 0:
			if which_psal == 1:
				if i_data==0: print(f'On conserve les donnees PSAL')
				ds_argo_Rtraj_inair['PSAL'][i_data] = psal_results[isok][0]
				ds_argo_Rtraj_inwater['PSAL'][i_data] = psal_results[isok][0]
			elif which_psal == 2:
				if i_data==0:print(f'On conserve les donnees PSAL_ADJUSTED')
				ds_argo_Rtraj_inair['PSAL'][i_data] = psal_adj_results[isok][0]
				ds_argo_Rtraj_inwater['PSAL'][i_data] = psal_adj_results[isok][0] 
			else:
				if i_data==0:print(f'On conserve les donnees PSAL_ADJUSTED quand elles existent.\n\
Sinon, on prend les donnees PSAL.')
				ds_argo_Rtraj_inair['PSAL'][i_data] = psal_mixte_results[isok][0]
				ds_argo_Rtraj_inwater['PSAL'][i_data] = psal_mixte_results[isok][0]
		else:
			ds_argo_Rtraj_inair['PSAL'][i_data]=np.nan
			ds_argo_Rtraj_inwater['PSAL'][i_data]=np.nan
            
	if (np.abs(ds_argo_Rtraj_inair['TEMP'][i_data] - temp_results[isok][0]) > 0.5):
		print(f'Cycle {cycle_results[isok][0]}\n \
		La temperature du RTraj differe de plus de 0.5 degre de la temperature du Sprof')

	# On ne garde que les variables utiles.
	ds_argo_Rtraj_inair = ds_argo_Rtraj_inair[['LONGITUDE_ARGO','LATITUDE_ARGO','PPOX_DOXY','TEMP','PSAL','JULD','CYCLE_NUMBER']]
	ds_argo_Rtraj_inwater = ds_argo_Rtraj_inwater[['LONGITUDE_ARGO','LATITUDE_ARGO','PPOX_DOXY','TEMP','PSAL','JULD','CYCLE_NUMBER']]

	return ds_argo_Rtraj_inair, ds_argo_Rtraj_inwater, optode_height

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

