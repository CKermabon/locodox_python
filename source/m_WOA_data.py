import xarray as xr
from m_fonctions import O2stoO2p
import numpy as np

# Fonction de lecture du fichier WOA et interpolation sur lon/lat/time ARGO
def read_WOA(fic_WOA,lon_argo,lat_argo,time_argo):
    """ 
    Cette fonction 'lit' le fichier WOA et interpole les données sur positions/date ARGO

    En entree :
        fic_WOA = Fichier WOA
        
    En sortie :
        ds_woa : donnees WOA interpolees aux postions/dates ARGO sur une grille reguliere en pression (min_pres:1:max_pres)
    """
    ds_woa = xr.open_dataset(fic_WOA)
    ds_woa = ds_woa.assign_coords(lon=("lon",ds_woa['longitude'].values))
    ds_woa = ds_woa.assign_coords(lat=("lat",ds_woa['latitude'].values))
    #
    # Passage des longitudes de [0 360] a [-180 180]
    #
    ds_woa['lon'] = xr.where(ds_woa['lon'] > 180, ds_woa['lon'] - 360, ds_woa['lon'])

    #
    # Dans WOA, les donnees sont affectees a un jour dans l'annee.
    # Les jours vont de 15 à 350. Pour interpoler sur les jours de l'annee (de 1 a 365),
    # on extrapole les donnees WOA.
    #

    var_to_extend = [var for var in ds_woa.data_vars if "time" in ds_woa[var].dims]
    new_time = xr.concat([
        ds_woa['time'][-1] - 365.25, 
        ds_woa['time'],                
        ds_woa['time'][0] + 365.25 # Après le dernier
    ], dim='time')
    
    extended_var = {}
    for var in var_to_extend:
        print(f'Extrapolation WOA temporelle en cours pour la variable : {var}')
        data_en_cours = ds_woa[var]
        first_data = data_en_cours.isel(time=0)
        last_data = data_en_cours.isel(time=-1)
        extended_data = xr.concat([first_data,data_en_cours,last_data],dim='time')
        extended_data = extended_data.assign_coords(time=new_time)
        extended_var[var] = extended_data 

    ds_woa_extended = ds_woa.copy()   
    for var, data in extended_var.items():
        ds_woa_extended[var] = data.assign_coords(time=new_time)

    #
    # Interpolation des donnees WOA sur lon/lat/time ARGO.
    #
    ds_woa_interp = ds_woa_extended.interp(lat=lat_argo,lon=lon_argo,time=time_argo.dt.dayofyear)
    ds_woa_interp = ds_woa_interp.transpose() # Les tableaux du dataset ont la dimension (N_CYCLE,N_DEPTH)

    return ds_woa_interp

# Fonction de lecture du fichier WOA et interpolation sur lon/lat/time ARGO
def read_WOA_old(fic_WOA,lon_argo,lat_argo,time_argo,min_pres,max_pres):
    """ 
    Cette fonction 'lit' le fichier WOA et interpole les données sur positions/date ARGO

    En entree :
        fic_WOA = Fichier WOA
        
    En sortie :
        ds_woa : donnees WOA interpolees aux postions/dates ARGO sur une grille reguliere en pression (min_pres:1:max_pres)
    """
    ds_woa = xr.open_dataset(fic_WOA)
    ds_woa = ds_woa.assign_coords(lon=("lon",ds_woa['longitude'].values))
    ds_woa = ds_woa.assign_coords(lat=("lat",ds_woa['latitude'].values))
    #
    # Passage des longitudes de [0 360] a [-180 180]
    #
    ds_woa['lon'] = xr.where(ds_woa['lon'] > 180, ds_woa['lon'] - 360, ds_woa['lon'])

    #
    # Dans WOA, les donnees sont affectees a un jour dans l'annee.
    # Les jours vont de 15 à 350. Pour interpoler sur les jours de l'annee (de 1 a 365),
    # on extrapole les donnees WOA.
    #

    var_to_extend = [var for var in ds_woa.data_vars if "time" in ds_woa[var].dims]
    new_time = xr.concat([
        ds_woa['time'][-1] - 365.25, 
        ds_woa['time'],                
        ds_woa['time'][0] + 365.25 # Après le dernier
    ], dim='time')
    
    extended_var = {}
    for var in var_to_extend:
        print(f'Extrapolation WOA temporelle en cours pour la variable : {var}')
        data_en_cours = ds_woa[var]
        first_data = data_en_cours.isel(time=0)
        last_data = data_en_cours.isel(time=-1)
        extended_data = xr.concat([first_data,data_en_cours,last_data],dim='time')
        extended_data = extended_data.assign_coords(time=new_time)
        extended_var[var] = extended_data 

    ds_woa_extended = ds_woa.copy()   
    for var, data in extended_var.items():
        ds_woa_extended[var] = data.assign_coords(time=new_time)

    #
    # Interpolation des donnees WOA sur lon/lat/time ARGO.
    #
    ds_woa_interp = ds_woa_extended.interp(lat=lat_argo,lon=lon_argo,time=time_argo.dt.dayofyear)
    ds_woa_interp = ds_woa_interp.transpose() # Les tableaux du dataset ont la dimension (N_CYCLE,N_DEPTH)
    
    # Interpolation sur une grille reguliere en pression.
    new_pres = np.arange(min_pres,max_pres+1,1)
    var_to_interpol = [var for var in ds_woa_interp.data_vars if "Depth" in ds_woa_interp[var].dims]
    interpol_var = {}
    interpol_data = np.zeros(shape=(len(ds_woa_interp['N_PROF']),max_pres-min_pres+1))
    for var in var_to_interpol:
        print(f'Interpolation de la variable WOA {var} sur grille reguliere en pression')
        for i_cycle in (range(0,len(ds_woa_interp['N_PROF']))):
            data_en_cours = ds_woa_interp[var][i_cycle,:]
            isok = np.isfinite(data_en_cours)
            nb_indices = np.count_nonzero(isok)
            if nb_indices>0:
                interpol_data[i_cycle,:] = np.interp(new_pres,ds_woa_interp['preswoa'][i_cycle,isok].values,data_en_cours[isok])
            else:
                interpol_data[i_cycle,:] = np.nan
        interpol_var[var] = interpol_data

    return interpol_var    