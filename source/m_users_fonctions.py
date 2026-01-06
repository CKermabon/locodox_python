# Definition de fonctions utiles
import numpy as np
import xarray as xr



def interp_climatology_float(grid_sal, grid_theta, grid_pres, float_sal, float_theta, float_pres):
    """
    interpolate historical salinity and pressure data on the float theta
    :param grid_sal: historical salinity
    :param grid_theta: historical potential temperature
    :param grid_pres: historical pressure
    :param float_sal: current float salinity
    :param float_theta: current float potential temperature
    :param float_pres: current float pressure
    :return: matrices [number of floats x number of historical profiles] of
    interpolated salinity and interpolated pressure on float theta surface

    Comes from pyowc (1st version)
    """

    # get the shape of the data inputs (float and climatology)
    grid_level = grid_sal.shape[0]
    grid_stations = grid_sal.shape[1]
    float_len = float_sal.shape[0]

    # initialise variables to hold the interpolated values
    interp_sal_final = np.full((float_len, grid_stations), np.nan, dtype=np.float64)
    interp_pres_final = np.full((float_len, grid_stations), np.nan, dtype=np.float64)

    # check that the climatology data has no infinite (bad) values in the middle
    # of the profiles.
    max_level = 0
    for i in range(0, grid_stations):

        # find where data is not missing
        grid_good_sal = np.isfinite(grid_sal[:, i])
        grid_good_theta = np.isfinite(grid_theta[:, i])
        grid_good_pres = np.isfinite(grid_pres[:, i])

        # find indices of good data
        grid_good_data_index = []
        for j in range(0, grid_level):
            if grid_good_sal[j] and grid_good_theta[j] and grid_good_pres[j]:
                grid_good_data_index.append(j)

        # now find the max level
        grid_good_data_len = grid_good_data_index.__len__()
        if grid_good_data_len != 0:
            for j in range(0, grid_good_data_len):
                grid_sal[j, i] = grid_sal[grid_good_data_index[j], i]
                grid_theta[j, i] = grid_theta[grid_good_data_index[j], i]
                grid_pres[j, i] = grid_pres[grid_good_data_index[j], i]
                max_level = np.maximum(max_level, grid_good_data_len)

    # Truncate the number of levels to the maximum level that has
    # available data
    if max_level > 0:
        grid_sal = grid_sal[:max_level, :]
        grid_theta = grid_theta[:max_level, :]
        grid_pres = grid_pres[:max_level, :]
    else:
        raise ValueError("No good climatological data has been found")

    # find where data isn't missing in the float
    float_good_sal = np.isfinite(float_sal)
    float_good_theta = np.isfinite(float_theta)
    float_good_pres = np.isfinite(float_pres)

    # get the indices of the good float data
    float_good_data_index = []
    for i in range(0, float_len):
        if float_good_sal[i] and float_good_theta[i] and float_good_pres[i]:
            float_good_data_index.append(i)

    # get the number of good float data observations
    float_data_len = float_good_data_index.__len__()

    # compare good float data to the closest climatological data
    for i in range(0, float_data_len):

        # get index of good float data
        index = float_good_data_index[i]
        # Find the difference between the float data and the climatological data
        delta_sal = np.array(grid_sal - float_sal[index])
        delta_pres = np.array(grid_pres - float_pres[index])
        delta_theta = np.array(grid_theta - float_theta[index])

        # Find the indices of the closest pressure value each climatological station has to
        # the float pressures
        abs_delta_pres = np.abs(delta_pres)
        delta_pres_min_index = np.nanargmin(abs_delta_pres, axis=0)
        # go through all the climatological stations
        for j in range(0, grid_stations):

            # find if delta_theta is different to delta_theta for closest pressure
            # (equals -1 if different)
            tst = np.sign(delta_theta[:, j]) * np.sign(delta_theta[delta_pres_min_index[j], j])
            # look for a theta match below the float pressure
            grid_theta_below_pres = np.argwhere(tst[delta_pres_min_index[j]:grid_level] < 0)
            # look for a theta match above the float pressure
            grid_theta_above_pres = np.argwhere(tst[0:delta_pres_min_index[j]] < 0)
            # initialise arrays to hold interpolated pressure and salinity
            interp_pres = []
            interp_sal = []

            # there is a theta value at a deeper level
            if grid_theta_below_pres.__len__() > 0:
                min_grid_theta_index = np.min(grid_theta_below_pres)
                i_1 = min_grid_theta_index + delta_pres_min_index[j]
                w_t = delta_theta[i_1, j] / (delta_theta[i_1, j] - delta_theta[i_1 - 1, j])
                interp_pres.append(w_t * delta_pres[i_1 - 1, j] + (1 - w_t) * delta_pres[i_1, j])
                interp_sal.append(w_t * delta_sal[i_1 - 1, j] + (1 - w_t) * delta_sal[i_1, j])

            # there is a theta value at a shallower level
            if grid_theta_above_pres.__len__() > 0:
                i_2 = np.max(grid_theta_above_pres)
                w_t = delta_theta[i_2, j] / (delta_theta[i_2, j] - delta_theta[i_2 + 1, j])
                interp_pres.append(w_t * delta_pres[i_2 + 1, j] + (1 - w_t) * delta_pres[i_2, j])
                interp_sal.append(w_t * delta_sal[i_2 + 1, j] + (1 - w_t) * delta_sal[i_2, j])

            if interp_pres.__len__() > 0:
                # if there are two nearby theta values, choose the closest one
                abs_interp_pres = np.abs(interp_pres)
                k = np.argmin(abs_interp_pres)
                interp_sal_final[index, j] = interp_sal[k] + float_sal[index]
                interp_pres_final[index, j] = interp_pres[k] + float_pres[index]

    return interp_sal_final, interp_pres_final


def write_param_results(dict_corr : dict,num_float : str,*args) :
    """ Function to write the corrections estimated on screen on in an ASCII file
    
    Parameters
    -----------
    dict_corr : dict
        dict of correction (label : value/error)
    num_float : str
        Float number
    *args : optionnal
        Name of the ASCII file
        Info_CTD/ctd_number/cycle_number/CTD_file)
    """
    line_tot = []
    if args:
        fic = args[0]
        info_ctd = args[1]
        liste_ctd = args[2]
        liste_cycle = args[3]
        fic_ctd = args[4]
        pres_threshold = args[5]
    else:
        fic = None
        info_ctd = 0

    line_tot.append(num_float)
    
    if info_ctd==1:
        line = f"Ctd/Cycle comparison : "
        line_tot.append(line)
        line = f"File  {fic_ctd} / CTD {liste_ctd} / Cycle  {liste_cycle} / Pressure_threshold {pres_threshold}"
        line_tot.append(line)
    
    for index, (key, value) in enumerate(dict_corr.items()):
        param =  value
        nb_dim = param.ndim
        if nb_dim==2:
            nb_segment = 1
        elif nb_dim == 3:
            val_bid1,nb_segment,val_bid2 = param.shape

        line_tot.append(key)

        if (nb_segment == 1):
            if len(param[0]) == 1:
                gain = param[0,0]
                drift = 0
                egain = param[1,0]
                edrift = 0
                coef_pres = 0
                error_pres = 0
            else:
                gain = param[0,0]
                drift = param[0,1]
                egain = param[1,0]
                edrift = param[1,1]
                if len(param[0])==3:
                    coef_pres = param[0,2]
                    error_pres = param[1,2]
                else:
                    coef_pres = 0
                    error_pres = 0

            line = f"Gain/Drift/Pressure {gain:.4f}/{drift:.4f}/{coef_pres:.4f} with error {egain:.4f}/{edrift:.4f}/{error_pres:.4f}"
            line_tot.append(line)

        else:                      
            for i in range(nb_segment):
                gain, drift,*coef_pres = param[0,i]
                egain,edrift,*error_pres = param[1,i]
                if len(coef_pres)==0:
                    coef_pres = 0
                    error_pres = 0
                else:
                    coef_pres = coef_pres[0]
                    error_pres = error_pres[0]
                line = f"Piece {i+1}  : "f"Gain/Drift/Pressure {gain:.4f}/{drift:.4f}/{coef_pres:.4f} with error {egain:.4f}/{edrift:.4f}/{error_pres:.4f}"
                line_tot.append(line)
        
    if fic is None:
        for line in line_tot:
            print(line)
    else:
        print('ecriture fichier')
        with open(fic, 'w') as f:
            for line in line_tot:
                f.write(line + '\n')
    return None

#
# Function to write the differents corrections estimated in an ASCII File.
#
def write_ASCII_file(fic : str, num_float : str, list_comment : list, list_corr : list, list_error : list) : 
    """ Function to write the coorections estimated in an ASCII file
    
    Parameters
    -----------
    fic : str
        Name of the result file
    num_float : str
        WMO float
    list_comment : list
        Comment for all the corrections
    list_corr : list
        Values of the corrections (slope/drift/pressure effect)
    alignt_number : int
        number of character after the comment to write the values (to align slope/drift/pressure effect in the file)
    """
    max_comment_len = max(len(var) for var in list_comment)
    max_corr_str_len = max(len("/".join(f"{val:.4f}" for val in var)) for var in list_corr)

    with open(fic, "w") as f:
        line = "WMO Float : " + num_float
        f.write(line+ "\n")
        for comment_en_cours,corr_en_cours, error_en_cours in zip(list_comment,list_corr,list_error) :
            comment_str = comment_en_cours.ljust(max_comment_len)
            corr_str =  "/".join(f"{val:.4f}" for val in corr_en_cours) 
            corr_str = corr_str.ljust(max_corr_str_len)
            line = f"{comment_str} {corr_str}"
            if len(error_en_cours) > 0 :
                error_str = "/".join(f"{val:.4f}" for val in error_en_cours)
                line = line + f"   (error : {error_str})"
            f.write(line + "\n")  # une chaîne par ligne
    return None
 
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
                interpol_data[i_prof,:] = np.interp(new_pres,ds[var_name_pres].isel(N_PROF=i_prof,N_LEVELS=isok).values,data_en_cours[isok],right=np.nan,left=np.nan)
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

def corr_data(ds_argo_Sprof : xr.Dataset,corr_final : np.ndarray,launch_date : np.datetime64,coef2 : float, coef3: float)->xr.Dataset:
    """ Function to apply correction on DOXY

    Parameters
    ----------
    ds_argo_Sprof : xr.Dataset
     Contains DOXY/PRES/DOXY_ADJUSTED Variables
    corr_final : np.ndarray
     Contains Gain/Drift/Pressure effect to apply
     launch_date : datetime64
     coef2, coef3 : float
         coefficient used in constructor pressure effect : (1 + (coef2 * Temp + coef3)*Pres/1000)     

     Returns
     -------
      ds_argo_Sprof : xr.Dataset
       Dataset with DOXY_ADJUSTED
    """
    delta_T_Sprof = diff_time_in_days(ds_argo_Sprof['JULD'].values,launch_date)
    for i_prof in range(0,len(ds_argo_Sprof['CYCLE_NUMBER'])):
        tab_delta_T= np.repeat(delta_T_Sprof[i_prof],len(ds_argo_Sprof['N_LEVELS']))
        #tab_delta_T= np.tile(delta_T_Sprof[i_prof],(1,len(ds_argo_Sprof['N_LEVELS'])))
        #print(tab_delta_T.shape)
        new_values = (corr_final[0] * (1+corr_final[1]/100 * tab_delta_T/365))* ds_argo_Sprof['DOXY'].isel(N_PROF=i_prof)
        
        if len(corr_final)==3: 
            if corr_final[2] == 0:
                new_values = new_values
            else:
                new_values = new_values/(1 + (coef2*ds_argo_Sprof['TEMP'].isel(N_PROF=i_prof) + coef3) *ds_argo_Sprof['PRES'].isel(N_PROF=i_prof)/1000)  
                new_values=  (1 + (coef2*ds_argo_Sprof['TEMP'].isel(N_PROF=i_prof) + corr_final[2]) *ds_argo_Sprof['PRES'].isel(N_PROF=i_prof)/1000) * new_values
            
            
        ds_argo_Sprof['DOXY_ADJUSTED'].loc[dict(N_PROF=i_prof)] = new_values

    return ds_argo_Sprof

def copy_attr(ds1 : xr.Dataset, ds2 : xr.Dataset) -> xr.Dataset:
    """ Function to copy attribut for each variable in ds1 from attribut form ds2

    Parameters
    ----------
    ds1 : xr.Dataset
    ds2 : xr.Dataset

    Returns
    --------
    ds2 : xr.Dataset
    For each variable in ds2, the attributs are copied from the same variable in ds1
    """
    for var in ds2.data_vars :
        ds2[var].attrs = ds1[var].attrs    

    return ds2
    