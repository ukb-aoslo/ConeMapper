import numpy as np

def polar_to_cartesian(polar, origin):
    """
    Convert polar to cartesian coordinates
    and shift to origin
    """
    x = polar[:,0] * np.cos(polar[:,1])
    y = polar[:,0] * np.sin(polar[:,1])
    return np.stack([y,x], axis=1) + origin
    

def cartesian_to_polar(cartesian, origin):
    """
    Shift cartesian coordinates by origin
    and convert to polar coordinates
    """
    cartesian = cartesian - origin
    r = np.sqrt(np.sum(cartesian ** 2, axis=1))
    theta = np.arctan2(cartesian[:,0], cartesian[:,1])
    return np.stack([r,theta], axis=1)