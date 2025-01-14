#########################################################
# Model Function for curve_fit.
#########################################################
def model_Gain(X,G):
    """ Function to estimate a correction with a Gain
    """
    return G * X 

def model_Gain_Derive(X,G,D):
    """ Function to estimate a correction with a Time Drift and a Gain
    """
    return (G * (1 + (D * X[1])/(365*100)) * X[0] )

def model_Gain_CarryOver(X,G,C):
    """ Function to estimate a correction with a Gain using CarryOver (for NCEP correction only)
    """
    return G * (X[0] - C * X[1]) / (1 - C) # C : Carry-over

def model_Gain_Derive_CarryOver(X,G,C,D):
    """ Function to estimate a correction with a Time Drift and a Gain using CarryOver (for NCEP correction only)
    """
    return (G / (1-C) * (1 + D / 100 * X[2]/365) * (X[0] - C * X[1]) )
