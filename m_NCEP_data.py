import xarray as xr
import glob

# Fonction de lecture des 3 fichiers NCEP utiles au calcul de PPOX_NCEP
def read_NCEP(rep_data_ncep,lon_argo,lat_argo,time_argo):
    """ 
    Cette fonction 'lit' les fichiers air/rhum/slp et renvoie les dataset associes

    En entree :
        rep_data_ncep : repertoire NCEP
	lon_argo/lat_argo/time_argo : Position/date ARGO
    En sortie :
        ds_slp, ds_air, ds_rhum : Données NCEP interpolées aux dates/positions ARGO
    """
    fic_air = glob.glob(rep_data_ncep + 'air.sig*nc')
    ds_air = xr.open_mfdataset(fic_air)
    fic_rhum = glob.glob(rep_data_ncep + 'rhum.sig*nc')
    ds_rhum = xr.open_mfdataset(fic_rhum)
    fic_slp = glob.glob(rep_data_ncep + 'slp*nc')
    ds_slp = xr.open_mfdataset(fic_slp)
    #
    # On force les longitudes entre [-180 180], comme le sont les longitudes ARGO
    #
    if 'lon' in ds_air.coords:  # Vérifie que la coordonnée 'lon' existe
        ds_air['lon'] = xr.where(ds_air['lon'] > 180, ds_air['lon'] - 360, ds_air['lon'])
        
    if 'lon' in ds_rhum.coords:  # Vérifie que la coordonnée 'lon' existe
        ds_rhum['lon'] = xr.where(ds_rhum['lon'] > 180, ds_rhum['lon'] - 360, ds_rhum['lon'])
        
    if 'lon' in ds_slp.coords:  # Vérifie que la coordonnée 'lon' existe
        ds_slp['lon'] = xr.where(ds_slp['lon'] > 180, ds_slp['lon'] - 360, ds_slp['lon'])

    #
    # Interpolation sur les positions/date ARGO
    #
    ds_air = ds_air.interp(lat=lat_argo,lon=lon_argo,time=time_argo)
    ds_rhum = ds_rhum.interp(lat=lat_argo,lon=lon_argo,time=time_argo)
    ds_slp = ds_slp.interp(lat=lat_argo,lon=lon_argo,time=time_argo)

    #
    # Conversion d'unite.
    #
    ds_slp['slp']= ds_slp['slp']/100 # Passage de Pascal en HectoPascal/Millibar
    ds_air['air'] = ds_air['air'] - 273.15 # Passage Kelvin vers Celsius

    return ds_air,ds_slp,ds_rhum
