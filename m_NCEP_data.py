import xarray as xr
import glob
import os
from ftplib import FTP
from m_fonctions import watervapor
import numpy as np
# Fonction qui telecharge les donnees NCEP si necessaire
def download_NCEP(time_argo,ftp_server,rep_ftp,rep_local,ncep_variables):
    """
    Cette fonction telecharge les donnees slp/air/rhum NCEP pour les annees où le flotteur est actif.
    Les donnees sont telechargees uniquement si elles n'existent pas deja dans le repertoire local.
    En entree :
        time_argo : Date ARGO
        ftp_server = serveur ftp NCEP
        rep_ftp : repertoire NCEP
        rep_local : repertoire local

    """
    # Creation repertoire de sortie si besoin.
    os.makedirs(rep_local, exist_ok=True)
    
    # Connexion au serveur ftp (en tant qu'anonyme) et on se deplace dans le repertoire NCEP ou sont les fichiers a telecharger.
    ftp = FTP(ftp_server)
    ftp.login()
    ftp.cwd(rep_ftp)
    min_year = int(time_argo[0].dt.year)
    max_year = int(time_argo[-1].dt.year)
    print(f"Telechargement des donnees NCEP si besoin")
    for iyear in range(min_year,max_year+1,1):
        for i_var in range(0,len(ncep_variables),1):
            fic_en_cours = ncep_variables[i_var] + '.' + str(iyear) + '.nc'
            local_file = os.path.join(rep_local,fic_en_cours)   
            if os.path.exists(local_file):
                print(f"File {fic_en_cours} already exists.")
            else:
                with open(local_file,"wb") as f:
                    ftp.retrbinary(f"RETR {fic_en_cours}", f.write) 
                print(f"Fichier {local_file} downloaded")

    ftp.quit()
    
        
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

def calcul_NCEP_PPOX(dsinwater,ds_NCEP_air,ds_NCEP_slp,ds_NCEP_rhum,optode_height,z0q):
    """
    Cette fonction calcule le PPOX NCEP pour les dates/positions ARGO.

    En entree :
        dsinwater : Donnees ARGO InWater
        ds_NCEP_air : donnees NCEP pour la variable air pour les dates/positions ARGO
        ds_NCEP_slp : Idem pour la variable slp
        ds_NCEP_rhum : Idem pour la valeur rhum
        optode_height : hauteur de l'optode
        z0q : Constante

    En sortie :
        ncep_PPOX : PPOX NCEP aus dates/positions ARGO
    """
    bid=watervapor(dsinwater['TEMP'],dsinwater['PSAL'])
    SSph20 = bid * 1013.25 #mbar, seasurface water vapor pressure
    ncep_phum = watervapor(ds_NCEP_air['air'].values,0) * ds_NCEP_rhum['rhum'].values/100*1013.25 #ncep water vapor pressure
    ncep_phum_optode_height = (SSph20.values + (ncep_phum - SSph20.values) * np.log(np.abs(optode_height)*100/z0q)/np.log(10/z0q))
    ncep_Po2 = (ds_NCEP_slp['slp'].values - ncep_phum_optode_height) * 0.20946   

    return ncep_Po2
