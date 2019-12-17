import os

import trackpy as tp
import scipy.optimize as spo
import numpy as np
import matplotlib as mpl
import matplotlib.animation as anm
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

idx = pd.IndexSlice

def dense2multiindex(trj_dat):
    no_of_col = trj_dat.shape[1]
    no_of_part = round((no_of_col-1)/2)
    
    times = trj_dat[0].values
    indexes = np.arange(0,no_of_part)
    
    index = pd.MultiIndex.from_product([times,indexes],names=('time','id'))
    
    trj = pd.DataFrame(
        {"x":trj_dat.iloc[:,1::2].values.flatten(),
         "y":trj_dat.iloc[:,2::2].values.flatten()},index=index)
    return trj
# load .dat and convert
def load_dat(name):
    date_parser = lambda time : pd.to_datetime(
        float(time)+2*3600, unit="s", origin=pd.Timestamp('1904-01-01'))
    
    trj_dat = pd.read_csv(name+'.dat',sep="\t",
                header=None, parse_dates=[0], date_parser=date_parser)
    
    trj = dense2multiindex(trj_dat)

    trj["time"] = trj.index.get_level_values("time")
    trj["name"] = os.path.split(name)[-1]
    trj["id"] = trj.index.get_level_values("id")
    
    trj = trj.set_index(["name","time","id"])
    
    return trj

def get_names(directory,ext='.dat'):

    base_names = [os.path.join(root,os.path.splitext(file)[0]) 
              for root, dirs, files in os.walk(directory) 
              for file in files 
              if file.endswith(ext)]
    base_names = list(np.sort(np.array(base_names)))
    
    return base_names 

def trackingLoading(filePath):
    
    t = pd.read_csv(filePath ,sep = "\t", index_col=0)
    t = t.filter(["frame", "particle", "x", "y"])
    t = t.set_index(["frame", "particle"])
    
    return t

def from_px_to_um(trj,px_size):
    
    trj.x = trj.x*px_size # microns per pixel
    trj.y = trj.y*px_size # microns per pixel
        
    return trj

def find_timestamp_file(test_num, tmstmp_list):
    
    tmstmp_file = np.NaN

    for f in tmstmp_list:

        if test_num in f:
            tmstmp_file = f
            break

    return tmstmp_file

def tmstmpLoading(path):
    
    t = pd.read_csv(path, sep='\t', header=None)
    time = pd.DataFrame(data=t[0].values, columns=['time'])
    
    #Calculation of timestamps in nanoseconds
    time["datetime"] =  pd.to_datetime(time.time+2*3600, unit="s", origin=pd.Timestamp('1904-01-01'))
    time["ellapsed_time"] = (time.datetime-time.datetime[0])
    time["time"] = time.ellapsed_time.dt.total_seconds()

    return time

def trjLoading(filepath):

    trj = pd.read_csv(filepath, sep="\t", index_col = [0,1])
    
    return trj

def add_time_to_trj(trj, timestamp):
    
    trj["time"] = np.NaN
    
    for p, t in trj.groupby("particle"):
        
        trj.loc[idx[:,p], "time"] = timestamp.time.values[:len(trj)]
        
    return trj

def get_center(trj):
    
    def calc_R(xc, yc):
        """ calculate the distance of each 2D points from the center (xc, yc) """
        return np.sqrt((trj.x.values-xc)**2 + (trj.y.values-yc)**2)

    def f_2(c):
        """ calculate the algebraic distance between the data points and the mean circle centered at c=(xc, yc) """
        Ri = calc_R(*c)
        return Ri - Ri.mean()    

    center_estimate = 0, 0
    center, ir = spo.leastsq(f_2, center_estimate)
    return center

def changeToPolarCoordinates(t):
    
    #Obtention of the circle centre defined by the trajectory for each particle 
    center = get_center(t)
    t.x = t.x-center[0]
    t.y = t.y-center[1]

    #New calculation of the ro and theta coordinates with the coordinates origin correction 
    t["r"] = np.sqrt(t.x**2 + t.y**2)
    t["theta"] = np.arctan2(t.y,t.x)

    return t

def movementAnalyser(trj, omega):
    
    trj["theta_unwrap"] = np.NaN
    trj["theta_dot"] = np.NaN
    trj["theta_prime"] = np.NaN
    trj["theta_prime_unwrap"] = np.NaN    
    trj["theta_dot_prime"] = np.NaN

    for p, trj_sub in trj.groupby("particle"):
        
        trj.loc[idx[:, p], "theta_unwrap"] = np.unwrap(trj_sub.theta.values)
        trj.loc[idx[:, p], "theta_dot"] = trj.loc[idx[:, p], "theta_unwrap"].diff().values/trj_sub.time.diff().values

        trj.loc[idx[:, p], "theta_prime"] = np.mod(trj.loc[idx[:, p], "theta_unwrap"].values-omega*trj_sub.time.values, 2*np.pi)
        trj.loc[idx[:, p], "theta_prime_unwrap"] = np.unwrap(trj.loc[idx[:, p], "theta_prime"].values)    
        trj.loc[idx[:, p], "theta_dot_prime"] = trj.loc[idx[:, p], "theta_prime_unwrap"].diff().values/trj_sub.time.diff().values
    
    return trj





























def animation_trapRF(video,omega,ax = None, circle = None, pad = 23, frames = None):

    if ax is None:
        fig, ax = plt.subplots(1,1,figsize=(4,4))
        
    if circle is None:
        center, radius = find_image_center(frame)
        
    else: 
        center, radius = circle
    
    def box(center, radius, pad = 0):
        return [int(np.round(center[0]-radius-pad)),
                int(np.round(center[1]-radius-pad)),
                int(np.round(center[0]+radius+pad)),
                int(np.round(center[1]+radius+pad))]
                
    def animate(frame):
        
        th = frame/framerate*omega/np.pi*180
        im = Image.fromarray(video[frame])
        im_rot = im.rotate(th,center = tuple(center)).crop(box = box(center, radius, pad = 4/0.1805))
        
        ax.imshow(im_rot)
        ax.axis('off')
        return ax
    
    framerate = video.frame_rate
    period = 1/(omega/(np.pi*2))
    if frames is None:
        frames = (np.round(period*framerate).values[0]).astype("int16")
        
    anim = anm.FuncAnimation(fig, animate, frames=frames,interval=1000/framerate, blit=False)
    plt.close(anim._fig)
    return anim

def find_image_center(frame, part_radius = None, minmass = 2000, vis = True, ax = None, **kargs):

    if part_radius is None:
        part_radius = np.round(2/0.1805)*2-1

    def as_gray(frame):
        red = frame[:, :, 0]
        green = frame[:, :, 1]
        blue = frame[:, :, 2]
        return 0.2125 * red + 0.7154 * green + 0.0721 * blue
    
    pos = tp.locate(as_gray(frame), part_radius, minmass, **kargs)
    center = get_center(pos)
    radius = np.sqrt((pos["x"]-center[0])**2+(pos["y"]-center[1])**2).mean()
    
    if vis and ax is None:
        fig, ax = plt.subplots(1,1,figsize=(5,5))
    
    if vis:
        ax.imshow(frame)
        ax.plot(pos.x,pos.y, 'o', fillstyle='none', color = "red")
        ax.plot(center[0],center[1],'.',color="blue")
        p = mpl.patches.Circle(center,radius, fill = False, color = "blue")
        ax.add_artist(p)
        ax.set_xlim(center[0]-radius-2*part_radius,center[0]+radius+2*part_radius)
        ax.set_ylim(center[1]-radius-2*part_radius,center[1]+radius+2*part_radius)
        
    return center, radius