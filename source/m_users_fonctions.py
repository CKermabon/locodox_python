# Definition de fonctions utiles
import numpy as np
import xarray as xr


#########################################################
# Fonctions permettant d'estimer le gain et/ou la derive.
#########################################################
def model_Gain(X,G):
    """
    Modele utilise via curvefit pour estimer un gain.
    """
    return G * X 

def model_Gain_Derive(X,G,D):
    """
    Modele utilise via curvefit pour estimer un gain et une derive
    """
    return (G * (1 + (D * X[1])/(365*100)) * X[0] )

def model_Gain_CarryOver(X,G,C):
    """
    Modele utilise via curvefit pour estimer un gain avec prise en compte CarryOver (utile pour calcul via donnees dans l'air NCEP)
    """
    return G * (X[0] - C * X[1]) / (1 - C) # C : Carry-over

def model_Gain_Derive_CarryOver(X,G,C,D):
    """
    Modele utilise via curvefit pour estimer un gain et une derive avec prise en compte CarryOver (utile pour calcul via donnees dans l'air NCEP)
    """
    return (G / (1-C) * (1 + D / 100 * X[2]/365) * (X[0] - C * X[1]) )

#
# Fonction de calcul de la vapeur d'eau.
#
def watervapor(T : xr.DataArray,S: xr.DataArray)->xr.DataArray:
	""" Function to calculate watervapor from Temperature and Salinity

    Parameters
    ----------
    T : xr.DataArray
        Temperature
    S : xr.DataArray
        Salinity

    Returns
    -------
    pw : xr.DataArray
        WaterVapor
	"""
	pw=(np.exp(24.4543-(67.4509*(100/(T+273.15)))-(4.8489*np.log(((273.15+T)/100)))-0.000544*S))
	return pw	
    
##############################
# Fonctions de conversion O2.
##############################
def umolkg_to_umolL(O2mmolKg: xr.DataArray,units:str,ana_dens: np.array =1000) -> xr.DataArray:
    """ Function to convert Oxygen from umol/Kg to umol/L

        Parameters
        ----------
        O2mmolKg : xr.DataArray
            Oxygen in umol/Kg
        units : str
            Units for O2mmolKg (must be micromole/kg)
        ana_dens : np.array
            potential density at 0

        Returns
        -------
        O2mmolL : np.DataArray
            Oxygen in units umol/L
    """
    if units == 'micromole/kg':
        O2mmolL = O2mmolKg / (1000 / ana_dens) 
        return O2mmolL
    else:
        print(f'Les donnees O2 doivent etre en micromole/kg')
        return None
        

def O2ctoO2p(O2conc : xr.DataArray,T: xr.DataArray,S:xr.DataArray,P: int=0) -> xr.DataArray:
    """ Function to convert oxygen concentration in umol/L into oxygen partial pressure

    Parameters
    -----------
        O2conc : xr.DataArray
            Oxygen concentration in umol/L
        T : xr.DataArray
            Temperature
        S : xr.DataArray
            Salinity
        P : int (default : 0)
            Pressure

    Returns
    --------
        pO2 : xr.DataArray
            Oxygen partial pressure
    """
    xO2     = 0.20946 # mole fraction of O2 in dry air (Glueckauf 1951)
    pH2Osat = 1013.25*(np.exp(24.4543-(67.4509*(100/(T+273.15)))-(4.8489*np.log(((273.15+T)/100)))-0.000544*S)) # saturated water vapor in mbar 
    sca_T   = np.log((298.15-T)/(273.15+T)) # scaled temperature for use in TCorr and SCorr
    TCorr   = 44.6596*np.exp(2.00907+3.22014*sca_T+4.05010*sca_T**2+4.94457*sca_T**3-2.56847e-1*sca_T**4+3.88767*sca_T**5) # temperature correction part from Garcia and Gordon (1992), Benson and Krause (1984) refit mL(STP) L-1; and conversion from mL(STP) L-1 to umol L-1
    Scorr   = np.exp(S*(-6.24523e-3-7.37614e-3*sca_T-1.03410e-2*sca_T**2-8.17083e-3*sca_T**3)-4.88682e-7*S**2) # salinity correction part from Garcia and Gordon (1992), Benson and Krause (1984) refit ml(STP) L-1
    Vm      = 0.317 # molar volume of O2 in m3 mol-1 Pa dbar-1 (Enns et al. 1965)
    R       = 8.314 # universal gas constant in J mol-1 K-1

    pO2=O2conc*(xO2*(1013.25-pH2Osat))/(TCorr*Scorr)*np.exp(Vm*P/(R*(T+273.15)))
    return pO2


def O2stoO2p(O2sat: xr.DataArray,T: xr.DataArray,S:xr.DataArray,P:int =0,P_atm:int =1013.25) -> xr.DataArray:
    """ Function to convert Oxygen saturation % into partial pressure

        Parameters
        -----------
            O2sat : xr.DataArray
                Saturation Oxygen pourcentage
            T : xr.DataArray
                Temperature
            S : xr.DataArray
                Salinity
            P : int
                Pressure (default : 0)
            P_atm : int
                atmospheric pressure (default : 1013.25)
                
        Returns
        --------
            pO2 : xr.DataArray
                Oxygen partial pressure
            
    """
    xO2     = 0.20946 # mole fraction of O2 in dry air (Glueckauf 1951)
    pH2Osat = 1013.25*(np.exp(24.4543-(67.4509*(100/(T+273.15)))-(4.8489*np.log(((273.15+T)/100)))-0.000544*S)); # saturated water vapor in mbar
    Vm      = 0.317 # molar volume of O2 in m3 mol-1 Pa dbar-1 (Enns et al. 1965)
    R       = 8.314 # universal gas constant in J mol-1 K-1

    pO2=O2sat/100*(xO2*(P_atm-pH2Osat))

    return pO2

def O2ctoO2s(O2conc:xr.DataArray,T: xr.DataArray,S: xr.DataArray,P: int =0,P_atm: int=1013.25)-> xr.DataArray:
    """ Function to convert oxygen concentration (in umol/L) into oxygen saturation pourcentage

        Parameters
        ----------
            O2conc : xr.DataArray
                Oxygen concentration iin umol/L
            T : xr.DataArray
                Temperature
            S : xr.DataArray
                Salinity
            P : int
                Pressure (default : 0)
            P_atm : int
                atmospheric pressure (default : 1013.25)   
                
        Returns
        -------
            O2sat : xr.DataArray
                Oxygen saturation pourcentage
    
    """
    pH2Osat = 1013.25*(np.exp(24.4543-(67.4509*(100/(T+273.15)))-(4.8489*np.log(((273.15+T)/100)))-0.000544*S)) # saturated water vapor in mbar
    sca_T   = np.log((298.15-T)/(273.15+T)) # scaled temperature for use in TCorr and SCorr
    TCorr   = 44.6596*np.exp(2.00907+3.22014*sca_T+4.05010*sca_T**2+4.94457*sca_T**3-2.56847e-1*sca_T**4+3.88767*sca_T**5)# temperature correction part from Garcia and Gordon (1992), Benson and Krause (1984) refit mL(STP) L-1; and conversion from mL(STP) L-1 to umol L-1
    Scorr   = np.exp(S*(-6.24523e-3-7.37614e-3*sca_T-1.03410e-2*sca_T**2-8.17083e-3*sca_T**3)-4.88682e-7*S**2)# salinity correction part from Garcia and Gordon (1992), Benson and Krause (1984) refit ml(STP) L-1
    Vm      = 0.317 # molar volume of O2 in m3 mol-1 Pa dbar-1 (Enns et al. 1965)
    R       = 8.314 # universal gas constant in J mol-1 K-1

    O2sat=O2conc*100/(TCorr*Scorr)/(P_atm-pH2Osat)*(1013.25-pH2Osat)*np.exp(Vm*P/(R*(T+273.15)))  

    return O2sat

def O2ptoO2c(pO2,T,S,P=0):
    """ Function to convert oxygen paertial pressure into oxygen concentration in umol/L

    Parameters
    ----------
        pO2 : xr.DataArray
            oxygen partial pressure
        T : xr.DataArray
                Temperature
        S : xr.DataArray
                Salinity
        P : int
                Pressure (default : 0)
                
    Returns
    -------
        O2conc : xr.DataArray
            Oxygen concentration in umol/L
    
    """
    xO2     = 0.20946 # mole fraction of O2 in dry air (Glueckauf 1951)
    pH2Osat = 1013.25*(np.exp(24.4543-(67.4509*(100/(T+273.15)))-(4.8489*np.log(((273.15+T)/100)))-0.000544*S)) #saturated water vapor in mbar
    sca_T   = np.log((298.15-T)/(273.15+T)) #scaled temperature for use in TCorr and SCorr
    TCorr   = 44.6596*np.exp(2.00907+3.22014*sca_T+4.05010*sca_T**2+4.94457*sca_T**3-2.56847e-1*sca_T**4+3.88767*sca_T**5)# temperature correction part from Garcia and Gordon (1992), Benson and Krause (1984) refit mL(STP) L-1; and conversion from mL(STP) L-1 to umol L-1
    Scorr   = np.exp(S*(-6.24523e-3-7.37614e-3*sca_T-1.03410e-2*sca_T**2-8.17083e-3*sca_T**3)-4.88682e-7*S**2) #salinity correction part from Garcia and Gordon (1992), Benson and Krause (1984) refit ml(STP) L-1
    Vm      = 0.317 # molar volume of O2 in m3 mol-1 Pa dbar-1 (Enns et al. 1965)
    R       = 8.314 # universal gas constant in J mol-1 K-1

    O2conc=pO2/(xO2*(1013.25-pH2Osat))*(TCorr*Scorr)/np.exp(Vm*P/(R*(T+273.15)))
    return O2conc
     
###################################
# Fonction d'interpolation sur une grille reguliere en pression.
###################################
def interp_pres_grid(min_pres : int,max_pres : int,var_to_interpol : list, ds : xr.Dataset,var_name_pres : str, var_dim_depth : str) -> xr.Dataset:
    """ Function to interpolate data on a regular pressure

    Parameters
    ----------
        min_pres/Max_pres : int
            Minimum and amximum pressure
        var_to_interpol : list
            List of variables from the dataset to interpolate
        ds : xr.Dataset
            Dataset
        var_name_pres : str
            The name of the pressure variable in the dataset

    Returns
        ds_interpol_var : xr.Dataset
            Xarray dataset with data interpolated on a regular pressure grid.

    """

    nb_profil = len(ds['N_PROF'])
    interpol_var = {}
    new_pres = np.arange(min_pres,max_pres+1,1)
    for var in var_to_interpol:
        print(f'Interpolation variable {var} on a regular pressure grid')
        interpol_data = np.zeros(shape=(nb_profil,max_pres-min_pres+1))
        for i_prof in (range(0,nb_profil)):
            data_en_cours = ds[var].isel(N_PROF=i_prof)
            isok = np.isfinite(data_en_cours)
            nb_indices = np.count_nonzero(isok)
            if nb_indices>0:
                #print(data_en_cours[isok])
                #print(ds[var_name_pres][i_cycle,isok].values)
                interpol_data[i_prof,:] = np.interp(new_pres,ds[var_name_pres].isel(N_PROF=i_prof,N_LEVELS=isok).values,data_en_cours[isok])
            else:
                interpol_data[i_prof,:] = np.nan
        interpol_var[var] = interpol_data  
        dims = ('N_PROF','N_LEVELS')
        ds_interpol_var = xr.Dataset({var: (dims, data) for var, data in interpol_var.items()})
        
    return ds_interpol_var

def diff_time_in_days(juld_data : np.ndarray, launch_date : np.datetime64) -> np.ndarray:
    """ Function to calculate a difference tile in days

    Parameters
    ----------
    juld_data : np.ndarray
        JULD values
    launch_date : np.datetime64
        Date from which we calculate the number of days

    Returns
    -------
    delta_T : np.ndarray
        Difference (juld_data - launch_date) in days

    """
    #delta_T = (juld_data - launch_date)
    #delta_T = delta_T.astype(float)
    #delta_T = delta_T/1e9/86400 # Difference en jour
    delta_T = (juld_data - launch_date) / np.timedelta64(1,'D')
    return delta_T

def corr_data(ds_argo_Sprof : xr.Dataset,corr_final : np.ndarray,launch_date : np.datetime64)->xr.Dataset:
    """ Function to apply correction on DOXY

    Parameters
    ----------
    ds_argo_Sprof : xr.Dataset
     Contains DOXY/PRES/DOXY_ADJUSTED Variables
    corr_final : np.ndarray
     Contains Gain and Drift to apply

     Returns
     -------
      ds_argo_Sprof : xr.Dataset
       Dataset with DOXY_ADJUSTED
    """
    delta_T_Sprof = diff_time_in_days(ds_argo_Sprof['JULD'].values,launch_date)
    for i_prof in range(1,len(ds_argo_Sprof['CYCLE_NUMBER'])):
        tab_delta_T= np.repeat(delta_T_Sprof[i_prof],len(ds_argo_Sprof['N_LEVELS']))
        #tab_delta_T= np.tile(delta_T_Sprof[i_prof],(1,len(ds_argo_Sprof['N_LEVELS'])))
        #print(tab_delta_T.shape)
        new_values = (corr_final[0] * (1+corr_final[1]/100 * tab_delta_T/365))* ds_argo_Sprof['DOXY'].isel(N_PROF=i_prof) 
        ds_argo_Sprof['DOXY_ADJUSTED'].loc[dict(N_PROF=i_prof)] = new_values

    return ds_argo_Sprof
    