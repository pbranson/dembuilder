import numpy as np

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return rho, phi

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y

def cart2pol2(x, y):
    rho, phi = cart2pol(y, x)
    deg = phi * 180. / np.pi
    inds = deg < 0
    deg[inds] = 360. + deg[inds]
    return rho, deg

def pol2cart2(rho, deg):
    x, y = pol2cart(rho, float(deg)/180.*np.pi)
    return y, x