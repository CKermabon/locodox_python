import netCDF4
import numpy as np
import xarray as xr
from m_users_fonctions import diff_time_in_days
from datetime import datetime

def read_netcdf_all(file_name : str, var_in : list=None, verbose : bool=False)->dict:
    """ Function to read a NetCDF file

    Parameters
    ------------
    file_name : str
        NetCDF file to read
    var_in : (list, optional : Default : None)
        Varraible Name to read.
        if None, all variables are read
    verbose : (bool, optional : Default : False) 
    If True, all informations are written
 
    Returns
    --------
    dict : dictionnary
    Contains all the variables, attributs, of the Netcdf File.
    """
    # Open NetCDF file
    dataset = netCDF4.Dataset(file_name, mode='r')

    # Global attributs
    glob_att = {attr: getattr(dataset, attr) for attr in dataset.ncattrs()}
    if verbose:
        print("Attributs globaux:", glob_att)

    # Dimensions
    dimensions = {dim: len(dataset.dimensions[dim]) for dim in dataset.dimensions}
    if verbose:
        print("Dimensions:", dimensions)

    # Variables
    variables = {}
    for var_name, variable in dataset.variables.items():
        if var_in is None or var_name in var_in:
            variables[var_name] = {
                "data": variable[:],  # Data
                "dimensions": variable.dimensions,  # Dimensions
                "attributes": {attr: getattr(variable, attr) for attr in variable.ncattrs()}  # Attributs
            }
            if verbose:
                print(f"Variable: {var_name}, Dimensions: {variable.dimensions}, Attributs: {list(variable.ncattrs())}")

    # Close file
    dataset.close()

    return {
        "global_attributes": glob_att,
        "dimensions": dimensions,
        "variables": variables
    }

def write_netcdf(file_name : str, data : dict, dim_unlim : str,verbose : bool=False)->None:
    """ Function to write a NetCDF file

    Parameters
    ---------
    file_name :  str
     Name file to create
    data : dict
        Data to write into the Netcdf file
    dim_unlim : str
     Name of the unlimited dimension
    verbose : bool (optional : Default : False)
    If True, all informations are written

    Returns 
    -------
    None
    The file_name is created with the data.
    """
    # Netcdf file Creation
    dataset = netCDF4.Dataset(file_name, mode='w', format='NETCDF4')
    for dim_name, dim_size in data["dimensions"].items():
        if dim_name == dim_unlim:
            dataset.createDimension(dim_name, None)
        else:
            dataset.createDimension(dim_name, dim_size)

        if verbose:
            print(f"Dimension écrite: {dim_name} (taille: {dim_size})")

    # Variables
    for var_name, var_data in data["variables"].items():
        fill_value = var_data["attributes"].get("_FillValue", None)
        var = dataset.createVariable(var_name, var_data["data"].dtype,var_data["dimensions"], fill_value=fill_value)
        var[:] = var_data["data"]

        # Variables attributs
        for attr_name, attr_value in var_data["attributes"].items():
            if attr_name != "_FillValue":
                setattr(var, attr_name, attr_value)
        if verbose:
            print(f"Variable écrite: {var_name}, Dimensions: {var_data['dimensions']}")

    # Global Attributs
    for attr_name, attr_value in data["global_attributes"].items():
        setattr(dataset, attr_name, attr_value)
    if verbose:
        print("Attributs globaux écrits:", data["global_attributes"])

    # close the file
    dataset.close()
    if verbose:
        print(f"Fichier NetCDF écrit: {file_name}")


def xarray_to_dict(ds:xr.Dataset)->dict:
    """ Function to converta xarray dataset to a dictionnary OK for write_netcdf.

    Parameters
    -----------
    ds : xr.Dataset
        Dataset to convert

    Returns
    --------
    dict : dictionnary
    """
    # Dimensions
    #dimensions = {dim: size for dim, size in ds.dims.items()}
    dimensions = {dim: size for dim, size in ds.sizes.items()}

    # Variables
    variables = {}
    for var_name, da in ds.data_vars.items():
        variables[var_name] = {
            "data": da.values,  # Données de la variable
            "dimensions": da.dims,  # Noms des dimensions
            "attributes": da.attrs  # Attributs
        }

    # global attributs
    global_attributes = ds.attrs

    return {
        "dimensions": dimensions,
        "variables": variables,
        "global_attributes": global_attributes
    }

def dict_to_xarray(data : dict)->xr.Dataset:
    """ Function to convert a dinctionnary to a xarray dataset

    Parameters
    -----------
    data : dictionnary

    Returns
    -------
        ds : xr.Dataset
    """
    # Dictionnary creation
    data_vars = {}

    # Variables
    for var_name, var_data in data["variables"].items():
        data_vars[var_name] = xr.DataArray(
            data=var_data["data"],  # Données de la variable
            dims=var_data["dimensions"],  # Noms des dimensions
            attrs=var_data["attributes"]  # Attributs de la variable
        )

    # Dataset xarray creation
    ds = xr.Dataset(
        data_vars=data_vars,
        attrs=data["global_attributes"]  # Attributs globaux
    )

    return ds

def corr_file(fic_en_cours : str,fic_res : str,launch_date : np.datetime64,comment_corr :str,coef_corr : str,eq_corr : str,gain_final : np.float64 = 1,drift_final : np.float64=0,coef_pres:np.float64 = 0,percent_relative_error: np.float64=3) -> None : 
    """ Function to update the B_monoprofile with the DOXY_ADJUSTED

    Parameters
    ----------
    fic_en_cours : str
        Original file
    fic_res : str
        Result file
    launch_date : np.datetime64
        Launch Date
    comment_corr : str
        Information about DOXY Correction
    coef_corr : str
        Correction Coefficient
    eq_corr : str
        Correction Equation
    gain_final : np.float
        Slope to apply (default = 1)
    drift_final : np.float
        Drift to apply (default = 0)
    coef_pres : np.float
        Correction for pressure effect (default = 0)

    Returns
    -------
    None
    The file fic_res is created, containing the DOXY_ADJUSTED
    
    

    Returns
    -------
    None
    The result file is created
    """
    dsargo_oxy = xr.open_dataset(fic_en_cours,engine='argo')
    delta_T = diff_time_in_days(dsargo_oxy['JULD'],launch_date)
    nb_depth = len(dsargo_oxy['N_LEVELS'])
    nb_profil = len(dsargo_oxy['N_PROF'])
    delta_T = delta_T.to_numpy()
    delta_T = delta_T.reshape(nb_profil,1)
    delta_T = np.tile(delta_T,nb_depth)
    doxy_index = np.where(dsargo_oxy['PARAMETER'].str.strip() =='DOXY')
    dsargo_oxy.close()
    
    dsargo_oxy = xr.open_dataset(fic_en_cours,decode_cf = False)
#    for vname in dsargo_oxy:
#        if "_FillValue" in dsargo_oxy[vname].attrs and isinstance(dsargo_oxy[vname]._FillValue, bytes):
#            dsargo_oxy[vname].attrs["_FillValue"] = dsargo_oxy[vname].attrs["_FillValue"].decode("utf8")

    for i_prof in range(nb_profil):
        O2_ARGO_corr = (gain_final * (1+drift_final/100 * delta_T[i_prof,:]/365))* dsargo_oxy['DOXY'].isel(N_PROF=i_prof)
        O2_ARGO_corr = O2_ARGO_corr * (1 + coef_pres * dsargo_oxy['PRES'].isel(N_PROF=i_prof)/1000)
        dsargo_oxy['DOXY_ADJUSTED'].loc[dict(N_PROF=i_prof)] =  O2_ARGO_corr 
        dsargo_oxy['DOXY_ADJUSTED_ERROR'].loc[dict(N_PROF=i_prof)] =  O2_ARGO_corr * percent_relative_error /100



    # FillValue where no DOXY DATA
    dsargo_oxy['DOXY_ADJUSTED'] = dsargo_oxy['DOXY_ADJUSTED'].where(dsargo_oxy['DOXY']!=dsargo_oxy['DOXY'].attrs['_FillValue'],dsargo_oxy['DOXY_ADJUSTED'].attrs['_FillValue'])
    #dsargo_oxy['DOXY_ADJUSTED'] = dsargo_oxy['DOXY_ADJUSTED'].where(((dsargo_oxy['DOXY_QC']==1) | (dsargo_oxy['DOXY_QC']==2) | (dsargo_oxy['DOXY_QC']==3)),dsargo_oxy['DOXY_ADJUSTED'].attrs['_FillValue'])
    dsargo_oxy['DOXY_ADJUSTED_QC'] = dsargo_oxy['DOXY_QC']
    mask = dsargo_oxy['DOXY_QC'].isin([b'1', b'2', b'3'])  # Flag 1/2/3 ==> flag 1 because they are corrected
    dsargo_oxy['DOXY_ADJUSTED_QC'] = dsargo_oxy['DOXY_ADJUSTED_QC'].where(~mask, np.array(b'1', dtype='S1'))  

    mask = dsargo_oxy['DOXY_ADJUSTED_QC'].isin([b'4', b'9'])  # Flag 4/9 ==> FillValue
    dsargo_oxy['DOXY_ADJUSTED'] = dsargo_oxy['DOXY_ADJUSTED'].where(~mask, dsargo_oxy['DOXY_ADJUSTED'].attrs['_FillValue'])  
    dsargo_oxy['DOXY_ADJUSTED_ERROR'] = dsargo_oxy['DOXY_ADJUSTED_ERROR'].where(dsargo_oxy['DOXY_ADJUSTED']!=dsargo_oxy['DOXY_ADJUSTED'].attrs['_FillValue'],dsargo_oxy['DOXY_ADJUSTED_ERROR'].attrs['_FillValue'])
    #dsargo_oxy['DOXY_ADJUSTED'] = dsargo_oxy['DOXY_ADJUSTED'].where(((dsargo_oxy['DOXY_QC']==b'1') | (dsargo_oxy['DOXY_QC']==b'2') | (dsargo_oxy['DOXY_QC']==b'3')),dsargo_oxy['DOXY_ADJUSTED'].attrs['_FillValue'])



    for n_prof, n_calib, n_param in zip(*doxy_index):
        dsargo_oxy['PARAMETER_DATA_MODE'].loc[dict(N_PROF=n_prof,N_PARAM=n_param)] = 'D'
        dsargo_oxy['DATA_MODE'].loc[dict(N_PROF=n_prof)] = 'D'

    # Creation of dataset with variables with N_CALIB dimension
    var_n_calib = [var for var in dsargo_oxy.data_vars if "N_CALIB" in dsargo_oxy[var].dims]
    ds2 = dsargo_oxy[var_n_calib]
    ds2 = xr.concat([ds2,ds2.isel(N_CALIB=-1)],dim='N_CALIB')
    nb_calib = len(ds2['N_CALIB'])
    #doxy_index = np.where(ds2['PARAMETER'].isel(N_CALIB=nb_calib-1).str.strip() =='DOXY') 
    #doxy_index = np.where(ds2['PARAMETER'].isel(N_CALIB=nb_calib-1).str.strip().values =='DOXY') 

    for n_prof, n_calib_bid, n_param in zip(*doxy_index):
        ds2['SCIENTIFIC_CALIB_COMMENT'].loc[dict(N_PROF=n_prof,N_PARAM=n_param,N_CALIB=n_calib)] = list(('{: <256}'.format(comment_corr))) 
        ds2['SCIENTIFIC_CALIB_DATE'].loc[dict(N_PROF=n_prof,N_PARAM=n_param,N_CALIB=n_calib)] = list(datetime.now().strftime("%Y%m%d%H%M%S"))
        ds2['SCIENTIFIC_CALIB_COEFFICIENT'].loc[dict(N_PROF=n_prof,N_PARAM=n_param,N_CALIB=n_calib)] = list(('{: <256}'.format(coef_corr)))
        ds2['SCIENTIFIC_CALIB_EQUATION'].loc[dict(N_PROF=n_prof,N_PARAM=n_param,N_CALIB=n_calib)] = list(('{: <256}'.format(eq_corr)))

    
    
    ds3 = dsargo_oxy.drop_dims('N_CALIB')
    ds3['DATE_UPDATE'][:] = list(datetime.now().strftime("%Y%m%d%H%M%S"))
    ds3 = ds3.merge(ds2)
    #ds3.to_netcdf(fic_res)

    # PROFILE_DOXY_QC
    good_flags = [b'1', b'2', b'5', b'8']
    bad_flags = [b'3', b'4']

    for i in range(ds3.sizes['N_PROF']):
        doxy_qc_en_cours = ds3['DOXY_ADJUSTED_QC'].isel(N_PROF=i)
        good_count = np.isin(doxy_qc_en_cours, good_flags).sum()
        bad_count = np.isin(doxy_qc_en_cours, bad_flags).sum()
        total = good_count + bad_count # flag 0 and 9 are ignored

        if total!=0:
            if good_count == total:  # Tous les DOXY_QC sont bons
                ds3['PROFILE_DOXY_QC'].loc[dict(N_PROF=i)] = b'A'
            elif good_count / total >= 0.75:  # 75% des DOXY_QC sont bons
                ds3['PROFILE_DOXY_QC'].loc[dict(N_PROF=i)] = b'B'
            elif good_count / total >= 0.5 :  # 75% des DOXY_QC sont bons
                ds3['PROFILE_DOXY_QC'].loc[dict(N_PROF=i)] = b'C'
            elif good_count / total >= 0.25 :  # 75% des DOXY_QC sont bons
                ds3['PROFILE_DOXY_QC'].loc[dict(N_PROF=i)] = b'D'
            elif good_count / total > 0 :  # 75% des DOXY_QC sont bons
                ds3['PROFILE_DOXY_QC'].loc[dict(N_PROF=i)] = b'D'
            else:
                ds3['PROFILE_DOXY_QC'].loc[dict(N_PROF=i)] = b' '
    
    dict_res = xarray_to_dict(ds3)
    write_netcdf(fic_res,dict_res,'N_HISTORY')
    return None
