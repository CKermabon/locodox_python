import xarray as xr

# Fonction de lecture des 3 fichiers NCEP utiles au calcul de PPOX_NCEP
def read_WOA(fic_WOA,lon_argo,lat_argo,time_argo):
    """ 
    Cette fonction 'lit' le fichier WOA et interpole les données sur positions/date ARGO

    En entree :
        fic_WOA = Fichier WOA
        
    En sortie :
        ds_woa : donnees WOA interpolees aux postions/dates ARGO
    """
    ds_woa = xr.open_dataset(fic_WOA)
    ds_woa = ds_woa.assign_coords(lon=("lon",ds_woa['longitude'].values))
    ds_woa = ds_woa.assign_coords(lat=("lat",ds_woa['latitude'].values))
    #
    # Passage des longitudes de [0 360] a [-180 180]
    #
    ds_woa['lon'] = xr.where(ds_woa['lon'] > 180, ds_woa['lon'] - 360, ds_woa['lon'])
    #
    # Interpolation des donnees WOA sur lon/lat/time ARGO.
    #
    ds_woa_interp = ds_woa.interp(lat=lat_argo,lon=lon_argo,time=time_argo.dt.dayofyear)


    return ds_woa_interp
