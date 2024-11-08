import xarray as xr
import glob

# Fonction de lecture des 3 fichiers NCEP utiles au calcul de PPOX_NCEP
def read_NCEP(rep_data_ncep):
    """ 
    Cette fonction 'lit' les fichiers air/rhum/slp et renvoie les dataset associes

    En entree :
        rep_data_ncep : repertoire NCEP
    En sortie :
        ds_slp, ds_air, ds_rhum
    """
    fic_air = glob.glob(rep_data_ncep + 'air.sig*nc')
    ds_air = xr.open_mfdataset(fic_air)
    fic_rhum = glob.glob(rep_data_ncep + 'rhum.sig*nc')
    ds_rhum = xr.open_mfdataset(fic_rhum)
    fic_slp = glob.glob(rep_data_ncep + 'slp*nc')
    ds_slp = xr.open_mfdataset(fic_slp)

    return ds_air,ds_slp,ds_rhum
