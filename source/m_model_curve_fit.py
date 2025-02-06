#########################################################
# Model Function for curve_fit.
#########################################################
def model_Gain(X,G):
    """ Function to estimate, with curve_fit, a correction with a Gain 

    Parameters
    ----------
    X : Oxygen Values

    Returns
    --------
    G * X
    """
    return G * X 

def model_Gain_Derive(X,G,D):
    """ Function to estimate, with curve_fit, a correction with a Time Drift and a Gain 
    
    Parameters
    ----------
    X : Oxygen Values (X[0]) and delta_T from launch_date (X[1])

    Returns
    --------
    G * (1 + (D * X[1])/(365*100)) * X[0]
    """
    return (G * (1 + (D * X[1])/(365*100)) * X[0] )

def model_Gain_CarryOver(X,G,C):
    """ Function to estimate, with curve_fit, a correction with a Gain using CarryOver (for NCEP correction only)

    Parameters 
    -----------
    X contains PPOX in air (X[0]) and PPOX in Water (X[1])

    Returns
    --------
    G * (X[0] - C * X[1]) / (1 - C)
    """
    return G * (X[0] - C * X[1]) / (1 - C) # C : Carry-over

def model_Gain_Derive_CarryOver(X,G,C,D):
    """ Function to estimate, with curve_fit, a correction with a Time Drift and a Gain using CarryOver (for NCEP correction only)

    Parameters
    ----------
    X contains PPOX in air (X[0]), PPOX in water (X[1]) and delta_T from launch_date (X[2])

    Returns
    G / (1-C) * (1 + D / 100 * X[2]/365) * (X[0] - C * X[1])
    """
    return (G / (1-C) * (1 + D / 100 * X[2]/365) * (X[0] - C * X[1]) )

def model_corr_pres(X,Gp):
    """ Function to estimate a pressure effect correction (with curve_fit)

    Parameters
    ----------
    X: contains Oxygen Values (X[0] and Pressure X[1])

    Returns
    -------
    (1 + Gp * X[1]/1000) * X[0])
    """
    return ((1 + Gp * X[1]/1000) * X[0]) 