# # necessary imports
import numpy as np
import matplotlib.pyplot as plt

# function to convert xz space coordinates to radial coordinates

# function to convert xz space coordinates to radial coordinates
def convertToRadial(xz):
    # xz coordinates : 2D projection coordinates in x-z plane
    r = np.sqrt(xz[:,0]**2 + xz[:,1]**2)
    theta = np.arctan2(xz[:,1], xz[:,0])
    return np.column_stack((r,theta))

def circular_hist(ax, x, weights, bins=16, density=True, offset=0, gaps=True):
    """
    Produce a circular histogram of angles on ax.

    Parameters
    ----------
    ax : matplotlib.axes._subplots.PolarAxesSubplot
        axis instance created with subplot_kw=dict(projection='polar').

    x : array
        Angles to plot, expected in units of radians.

    bins : int, optional
        Defines the number of equal-width bins in the range. The default is 16.

    density : bool, optional
        If True plot frequency proportional to area. If False plot frequency
        proportional to radius. The default is True.

    offset : float, optional
        Sets the offset for the location of the 0 direction in units of
        radians. The default is 0.
        
    gaps : bool, optional
        Whether to allow gaps between bins. When gaps = False the bins are
        forced to partition the entire [-pi, pi] range. The default is True.

    Returns
    -------
    n : array or list of arrays
        The number of values in each bin.

    bins : array
        The edges of the bins.

    patches : `.BarContainer` or list of a single `.Polygon`
        Container of individual artists used to create the histogram
        or list of such containers if there are multiple input datasets.
    """
        # Wrap angles to [-pi, pi)
    x = (x+np.pi) % (2*np.pi) - np.pi

    # Force bins to partition entire circle
    if not gaps:
        bins = np.linspace(-np.pi, np.pi, num=bins+1)

    # Bin data and record counts
    n, bins = np.histogram(x, bins=bins, weights = weights)

    # Compute width of each bin
    widths = np.diff(bins)

    # By default plot frequency proportional to area
    if density:
        # Area to assign each bin
        area = n / x.size
        # Calculate corresponding bin radius
        radius = (area/np.pi) ** .5
    # Otherwise plot frequency proportional to radius
    else:
        radius = n

    # Plot data on ax
    patches = ax.bar(bins[:-1], radius, zorder=1, align='edge', width=widths,
                     edgecolor='C0', fill=False, linewidth=1)

    # Set the direction of the zero angle
    ax.set_theta_offset(offset)

    # Remove ylabels for area plots (they are mostly obstructive)
    if density:
        ax.set_yticks([])

    return n, bins, patches

def getRadialCoord(skel_pos, soma_pos): 
    skel_pos_wrtsoma = skel_pos - soma_pos
    skel_pos_wrtsoma_xz = np.column_stack((skel_pos_wrtsoma[:,0],skel_pos_wrtsoma[:,2]))
    # convert to radial positions - only extract x-z plane positions
    skel_pos_radial = convertToRadial(skel_pos_wrtsoma_xz)
    
    # get major axis coordinates
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    X = skel_pos_wrtsoma_xz
    X = (X-np.mean(X,axis = 0))
    pca.fit(X);
    majorAxis = pca.components_[0]
    
    majorAxisAngle = np.arctan2(majorAxis[1],majorAxis[0])
    # get radial coordinates with shifted axis wrt majorAxis determined
    theta = majorAxisAngle
    rotMat = np.array([[np.cos(theta), -np.sin(theta)],
               [np.sin(theta), np.cos(theta)]])
    skel_pos_transformed = rotMat @ skel_pos_wrtsoma_xz.T
    skel_pos_transformed = (skel_pos_transformed).T

    skel_pos_radial_transformed = convertToRadial(skel_pos_transformed)
    
    return skel_pos_radial, majorAxis, skel_pos_transformed, skel_pos_radial_transformed

def getAngularDensity(nrn,skel_pos_radial_transformed,df,field_to_sum='len',angle=np.pi/4):
    # get density of cable (weights_to_sum) etc +/- angle from major axis
    # effectively - we want points with theta values either +angle/2, -angle/2, pi-angle/2, -pi+angle/2
    skel_in_range_mask = (((skel_pos_radial_transformed[:,1] < angle/2) & (skel_pos_radial_transformed[:,1] > -angle/2)) | ((skel_pos_radial_transformed[:,1] > np.pi-(angle/2)) & (skel_pos_radial_transformed[:,1] < np.pi)) | ((skel_pos_radial_transformed[:,1] < -np.pi+(angle/2)) & (skel_pos_radial_transformed[:,1] > -np.pi)))
    skel_in_range = skel_pos_radial_transformed[skel_in_range_mask]  
    skel_out_range = skel_pos_radial_transformed[~skel_in_range_mask]
    
    # convert to mesh inds to pull out req weights
    meshmask_within = nrn.skeleton.SkeletonIndex(np.where(skel_in_range_mask)).to_mesh_mask
    meshmask_out = nrn.skeleton.SkeletonIndex(np.arange(len(skel_pos_radial_transformed))).to_mesh_mask
    
    # pull dataframe values
    withinAngle = np.sum(nrn.anno.segment_properties.filter_query(meshmask_within).df[field_to_sum].values)
    allAngle = np.sum(nrn.anno.segment_properties.filter_query(meshmask_out).df[field_to_sum].values)
    
    angularRatio = withinAngle/allAngle
    
    return angularRatio, withinAngle, skel_in_range_mask
    
    
    
    
    
    
    
    
    
    
    
    
