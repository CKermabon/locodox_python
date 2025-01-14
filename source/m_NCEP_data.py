import xarray as xr
import glob
import os
from ftplib import FTP
from m_users_fonctions import watervapor
import numpy as np
# Fonction qui telecharge les donnees NCEP si necessaire
def download_NCEP_force(annee: list,ftp_server: str,rep_ftp: str,rep_local:str,ncep_variables:list)->None:
    """ Function to download NCEP data 

    Parameters
    ----------
    annee : list
        year to download
    ftp_server : str
        Ncep ftp server
    rep_ftp : str
        directory on server from where to download the NCEP files
    rep_local : str
        directory to download the NCEP files
    ncep_variables : list
        NCEP variables name to download

    Returns
    -------
    None
    """
    # Creation repertoire de sortie si besoin.
    os.makedirs(rep_local, exist_ok=True)
    
    # Connexion au serveur ftp (en tant qu'anonyme) et on se deplace dans le repertoire NCEP ou sont les fichiers a telecharger.
    ftp = FTP(ftp_server)
    ftp.login()
    ftp.cwd(rep_ftp)
    min_year = annee[0]
    max_year = annee[-1]
    for iyear in range(min_year,max_year+1,1):
        for i_var in range(0,len(ncep_variables),1):
            fic_en_cours = ncep_variables[i_var] + '.' + str(iyear) + '.nc'
            local_file = os.path.join(rep_local,fic_en_cours)   
            with open(local_file,"wb") as f:
                ftp.retrbinary(f"RETR {fic_en_cours}", f.write) 
            print(f"File {local_file} downloaded")

    ftp.quit()
    
def download_NCEP_if_needed(time_argo:xr.DataArray,ftp_server:str,rep_ftp:str,rep_local:str,ncep_variables:list)-> None:
    """ Function to dowload NCEP file if needed.

    Parameters
    ----------
    time_argo : xr.DataArray (datetime64)
        ARGO JULD
    ftp_server : str
        NCEP ftp
    rep_ftp : str
        FTP NCEP directory
    rep_local : str
        Local directory to download NCEP files
    ncep_variables : list
        NCEP variables Name to download

    Returns
    -------
    None
    """
    # Creation repertoire de sortie si besoin.
    os.makedirs(rep_local, exist_ok=True)
    
    # Connexion au serveur ftp (en tant qu'anonyme) et on se deplace dans le repertoire NCEP ou sont les fichiers a telecharger.
    ftp = FTP(ftp_server)
    ftp.login()
    ftp.cwd(rep_ftp)
    min_year = int(time_argo[0].dt.year)
    max_year = int(time_argo[-1].dt.year)
    for iyear in range(min_year,max_year+1,1):
        for i_var in range(0,len(ncep_variables),1):
            fic_en_cours = ncep_variables[i_var] + '.' + str(iyear) + '.nc'
            remote_size = ftp.size(fic_en_cours) 
            local_file = os.path.join(rep_local,fic_en_cours)   
            if os.path.exists(local_file):
                local_size = os.path.getsize(local_file)
                if (local_size == remote_size):
                    print(f"File {fic_en_cours} already exists with the same size. No Download")
                else:
                    with open(local_file,"wb") as f:
                        ftp.retrbinary(f"RETR {fic_en_cours}", f.write) 
                    print(f"File {local_file} alreaddy exists but with a different size. Download")
                    
            else:
                with open(local_file,"wb") as f:
                    ftp.retrbinary(f"RETR {fic_en_cours}", f.write) 
                print(f"File {local_file} doesn't exist. Download")

    ftp.quit()
    

def open_NCEP_file(rep_data_ncep : str) -> xr.Dataset:
    """ Function to open NCEP File (air.sig*nc/rhu.sig*.nc/slp*.nc)

    Parameters
    ----------
    rep_data_ncep : str
        Local Directory where the NCEP files are located

    Returns
    -------
    ds_air/ds_rhum/ds_slp : xr.Dataset
        Data from the air/rhum/slp NCEP files
        Longitudes are forced to be [-180 180]

    """
    fic_air = glob.glob(rep_data_ncep + 'air.sig*nc')
    ds_air = xr.open_mfdataset(fic_air)
    fic_rhum = glob.glob(rep_data_ncep + 'rhum.sig*nc')
    ds_rhum = xr.open_mfdataset(fic_rhum)
    fic_slp = glob.glob(rep_data_ncep + 'slp*nc')
    ds_slp = xr.open_mfdataset(fic_slp)
    #
    # Force longitudes to be in [-180 180] (as ARGO Longitudes)
    #
    if 'lon' in ds_air.coords:  # Vérifie que la coordonnée 'lon' existe
        ds_air['lon'] = xr.where(ds_air['lon'] > 180, ds_air['lon'] - 360, ds_air['lon'])
        
    if 'lon' in ds_rhum.coords:  # Vérifie que la coordonnée 'lon' existe
        ds_rhum['lon'] = xr.where(ds_rhum['lon'] > 180, ds_rhum['lon'] - 360, ds_rhum['lon'])
        
    if 'lon' in ds_slp.coords:  # Vérifie que la coordonnée 'lon' existe
        ds_slp['lon'] = xr.where(ds_slp['lon'] > 180, ds_slp['lon'] - 360, ds_slp['lon'])  

    return ds_air, ds_rhum, ds_slp

# Fonction de lecture des 3 fichiers NCEP utiles au calcul de PPOX_NCEP
def interp_NCEP_on_ARGO(ds_air:xr.Dataset,ds_rhum : xr.Dataset, ds_slp : xr.Dataset,lon_argo : xr.DataArray,lat_argo:xr.DataArray,time_argo:xr.DataArray) -> xr.Dataset:
    """ Function to interpolate NCEP Dataset on ARGO lon/lat/time

    Parameters
    -----------
    ds_air/ds_rhum/ds_slp : xr.Dataset
    NCEP Data from air/rhum/slp NCEP file

    Returns
    --------
    ds_air/ds_rhum/ds_slp : xr.Dataset
    NCEP data interpolated on ARGO lon/lat/time

    """

    #
    # Interpolation on ARGO data
    #
    ds_air = ds_air.interp(lat=lat_argo,lon=lon_argo,time=time_argo)
    ds_rhum = ds_rhum.interp(lat=lat_argo,lon=lon_argo,time=time_argo)
    ds_slp = ds_slp.interp(lat=lat_argo,lon=lon_argo,time=time_argo)

    #
    # Units Conversion
    #
    ds_slp['slp']= ds_slp['slp']/100 # Transform Pascal to HectoPascal/Millibar
    ds_air['air'] = ds_air['air'] - 273.15 # Transform Kelvin to Celsius

    return ds_air,ds_rhum,ds_slp

def calcul_NCEP_PPOX(dsinwater: xr.Dataset,ds_NCEP_air: xr.Dataset,ds_NCEP_rhum: xr.Dataset,ds_NCEP_slp:xr.Dataset,optode_height:float = 0.2,z0q:float = 1e-4) -> np.ndarray:
    """ Function to calculate NCEP oxygen partial pressure

    Parameters
    -----------
    dsinwater : xr.Dataset
        ARGO InWater Data
    ds_NCEP_air : xr.Dataset
        NCEP air interpolated to ARGO position/time
    ds_NCEP_rhum : xr.Dataset
        NCEP rhum interpolated to ARGO position/time
    ds_NCEP_slp : xr.Dataset
        NCEP slp interpolated to ARGP position/time
    optode_height : float
        Optode Height relative to CTD (default = 20 cm)
    z0q : float
        Constante (default = 1e-4)

    Returns 
    --------
    ncep_Po2 : np.ndarray
        NCEP PPOX on ARGO position/time
    
    """
    bid=watervapor(dsinwater['TEMP'],dsinwater['PSAL'])
    SSph20 = bid * 1013.25 #mbar, seasurface water vapor pressure
    ncep_phum = watervapor(ds_NCEP_air['air'].values,0) * ds_NCEP_rhum['rhum'].values/100*1013.25 #ncep water vapor pressure
    ncep_phum_optode_height = (SSph20.values + (ncep_phum - SSph20.values) * np.log(np.abs(optode_height)/z0q)/np.log(10/z0q))
    ncep_Po2 = (ds_NCEP_slp['slp'].values - ncep_phum_optode_height) * 0.20946   

    return ncep_Po2
