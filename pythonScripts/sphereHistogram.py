# necessary imports
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
# function to convert to spherical coords 
def appendSpherical_np(xyz):
    ptsnew = np.hstack((xyz, np.zeros(xyz.shape)))
    xy = xyz[:,0]**2 + xyz[:,1]**2
    ptsnew[:,3] = np.sqrt(xy + xyz[:,2]**2)
    ptsnew[:,4] = np.arctan2(np.sqrt(xy), xyz[:,2]) # for elevation angle defined from Z-axis down #theta
    #ptsnew[:,4] = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
    ptsnew[:,5] = np.arctan2(xyz[:,1], xyz[:,0]) #phi
    return ptsnew

def makeSphericalHist(syn_pos, soma_pos,ntheta,nphi,sphereR): 
    soma_pos0 = [soma_pos[0]*4/1000,soma_pos[1]*4/1000, soma_pos[2]*40/1000] # soma pos in microns
    syn_pos_wrtsoma = syn_pos_nm - soma_pos0
    # convert to radial positions
    syn_pos_all = appendSpherical_np(syn_pos_wrtsoma)
    
    #make bins 
    syn_pos_radial = syn_pos_all[:,3:]
    # volume element = sin(theta) d(theta) d(phi)
    bins_phi = np.linspace(0,2*np.pi, nphi)
    bins_sintheta = np.linspace(-1,1,ntheta)
    bins_theta = np.arcsin(bins_sintheta)

    # meshgrid
    phimesh, thetamesh = np.meshgrid(bins_phi, bins_theta)

    hist, _, _ = np.histogram2d(syn_pos_radial[:,2],syn_pos_radial[:,1], bins=(bins_phi, bins_theta))
    phimesh, thetamesh = np.meshgrid(bins_phi, bins_theta)
    
    # surface plot representation
    #import matplotlib.colors
    # arrows indicating synaptic number for angular discretized space
    R = hist.T 
    X = R* np.sin(phimesh[:-1,:-1]) * np.cos(thetamesh[:-1,:-1])
    Y = R * np.sin(phimesh[:-1,:-1]) * np.sin(thetamesh[:-1,:-1])
    Z = R * np.cos(phimesh[:-1,:-1])
    
    # sphere as a reference for span of 
    r = sphereR
    x = r* np.sin(phimesh) * np.cos(thetamesh)
    y = r * np.sin(phimesh) * np.sin(thetamesh)
    z = r * np.cos(phimesh)

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1, projection='3d')
    plot = ax.plot_surface(
        X, Y, Z, rstride=1, cstride=1, linewidth=0, antialiased=False, alpha=0.7)
    plot2 = ax.plot_surface(
        x, y, z, rstride=1, cstride=1, linewidth=0, antialiased=False, alpha=0.3)

    ax.set_xlim([-sphereR,sphereR])
    ax.set_ylim([-sphereR,sphereR])
    ax.set_zlim([-sphereR,sphereR])