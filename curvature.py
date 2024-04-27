'''
Compute curvature of unevenly spaced points along curve
Useful for Tikhonov curve (L-curve)

Author: R Nate Crummett
Date: 04.26.24
'''
import numpy as np
from scipy.interpolate import Akima1DInterpolator

########################################################

def curvature(beta, data_misfit, model_norm):
    ''' Compute curvature of Tikhonov curve '''
    # Derivative of model_norm at beta spacing
    model_norm_deriv = Akima1DInterpolator(beta, model_norm, method = "makima").derivative()(beta)

    # Curvature of parametric curve
    k = model_norm*data_misfit * \
        (model_norm*data_misfit + beta*model_norm_deriv*data_misfit + beta**2*model_norm_deriv*model_norm) / \
        (np.abs(model_norm_deriv) * np.power(data_misfit**2 + beta**2*model_norm**2, 3/2))
    
    return k

