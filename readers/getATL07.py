import warnings
import h5py
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import h5py
import scipy
# from astropy.time import Time
import math

def height_sampling(x, height, L = 5000):
    # Find minimum of surface height for a particular distance
    # L: segment length (default: 10 km  = 10,000 m)
    
    min_x = math.floor(np.min(x)/L)*L
    max_x = math.ceil(np.max(x)/L)*L
    
    x_start = np.arange(min_x, max_x, L)
    x_end = np.arange(min_x + L, max_x+L, L)
    
    h_min = np.zeros(len(x))*np.nan    
    h_sum = np.zeros(len(x))
    h_cnt = np.zeros(len(x))
    
    for i in range(0, len(x_start)):
        x1 = x_start[i] - L/2
        x2 = x_end[i] + L/2
        L_idx = np.where((x >= x1) & (x <= x2))[0]
        if len(L_idx) > 0:
            h_sum[L_idx] += height[L_idx].min()
            h_cnt[L_idx] += 1
       
    h_min = smooth_line(h_sum/h_cnt, w = 1000)
    
    return h_min

def smooth_line(data, w = 2):
    # Smooth the surface with the defined window size
    output = np.zeros(len(data))
    for n in range(0, len(data)):
        output[n] = np.mean(data[max(0, n-w):min(len(data), n+w+1)])
    return output

def convert_time(delta_time):
    times = []
    years = []
    months = []
    days = []
    hours = []
    minutes = []
    seconds =[]
    
    for i in range(0, len(delta_time)):
        times.append(dt.datetime(1980, 1, 6) + dt.timedelta(seconds = delta_time[i]))
        years.append(times[i].year)
        months.append(times[i].month)
        days.append(times[i].day)
        hours.append(times[i].hour)
        minutes.append(times[i].minute)
        seconds.append(times[i].second)
    
    temp = pd.DataFrame({'time':times, 'year': years, 'month': months, 'day': days,
                         'hour': hours, 'minute': minutes, 'second': seconds
                        })
    return temp

# Function to read ATL03 data (.h5 format)
def get_ATL07(fname, beam_number, bbox, maxh = 1000):
    # 0, 2, 4 = Strong beam; 1, 3, 5 = weak beam
    # sfype: surface type (0=land, 1=ocean , 2=sea ice, 3=land ice, 4=inland water)
    
    f = h5py.File(fname, 'r')
    
    orient = f['orbit_info']['sc_orient'][:]  # orientation - 0: backward, 1: forward, 2: transition
    
    if len(orient) > 1:
        print('Transitioning, do not use for science!')
        return [[] for i in beamlist]
    elif (orient == 0):
        beams=['gt1l', 'gt1r', 'gt2l', 'gt2r', 'gt3l', 'gt3r']                
    elif (orient == 1):
        beams=['gt3r', 'gt3l', 'gt2r', 'gt2l', 'gt1r', 'gt1l']
    # (strong, weak, strong, weak, strong, weak)
    # (beam1, beam2, beam3, beam4, beam5, beam6)

    beam = beams[beam_number]  
    
    flag=f[beam]['sea_ice_segments']['heights']['height_segment_fit_quality_flag'][:]
    idx = (flag != -1)
    
    # height of each received photon, relative to the WGS-84 ellipsoid
    # (with some, not all corrections applied, see background info above)
    heights=f[beam]['sea_ice_segments']['heights']['height_segment_height'][idx]
    # latitude (decimal degrees) of each received photon
    lats=f[beam]['sea_ice_segments']['latitude'][idx]
    # longitude (decimal degrees) of each received photon
    lons=f[beam]['sea_ice_segments']['longitude'][idx]
    # longitude (decimal degrees) of each received photon
    x_atc=f[beam]['sea_ice_segments']['seg_dist_x'][idx]
    # seconds from ATLAS Standard Data Product Epoch. use the epoch parameter to convert to gps time
    deltatime=f[beam]['sea_ice_segments']['delta_time'][idx]
    # Height segment id
    sid=f[beam]['sea_ice_segments']['height_segment_id'][idx]
    
    if bbox != None:
        if bbox[0] < bbox[2]:
            valid = (lats >= bbox[1]) & (lats <= bbox[3]) & (lons >= bbox[0]) & (lons <= bbox[2])
        else:
            valid = (lats >= bbox[1]) & (lats <= bbox[3]) & ((lons >= bbox[0]) | (lons <= bbox[2]))
    
    if len(heights[valid]) == 0:
        return pd.DataFrame({})
    
    else:    
        # width of best fit gaussian
        width=f[beam]['sea_ice_segments']['heights']['height_segment_w_gaussian'][idx]
        # RMS difference between sea ice modeled and observed photon height distribution
        rms=f[beam]['sea_ice_segments']['heights']['height_segment_w_gaussian'][idx]
        # number of laser pulses
        n_pulse=f[beam]['sea_ice_segments']['heights']['height_segment_n_pulse_seg'][idx]
        
        # Length of each height segment
        seg_len=f[beam]['sea_ice_segments']['heights']['height_segment_length_seg'][idx]      
        # Beam azimuth
        b_azi = f[beam]['sea_ice_segments/geolocation/beam_azimuth'][idx]
        # Beam elevation
        b_ele = f[beam]['sea_ice_segments/geolocation/beam_coelev'][idx]
        # Solar azimuth
        s_azi = f[beam]['sea_ice_segments/geolocation/solar_azimuth'][idx]
        # Solar elevation
        s_ele = f[beam]['sea_ice_segments/geolocation/solar_elevation'][idx]
        
        # Default surface type of ATL07 product
        stype=f[beam]['sea_ice_segments']['heights']['height_segment_type'][idx]

        # Calculated background count rate based on sun angle, surface slope, unit reflectance
        bck_cal=f[beam]['sea_ice_segments']['stats']['backgr_calc'][idx]
        # Background count rate, averaged over the segment based on 200 hz atmosphere
        bck_r200=f[beam]['sea_ice_segments']['stats']['backgr_r_200'][idx]
        # Background count rate, averaged over the segment based on 25 hz atmosphere
        bck_r25=f[beam]['sea_ice_segments']['stats']['backgr_r_25'][idx]
        # Background count rate, averaged over the segment based on 25 hz atmosphere
        asr_25=f[beam]['sea_ice_segments']['stats']['asr_25'][idx]

        # Background rate normalized to a fixed solar elevation angle
        bck_norm=f[beam]['sea_ice_segments']['stats']['background_r_norm'][idx]

        # Segment histogram width estimate
        hist_w=f[beam]['sea_ice_segments']['stats']['hist_w'][idx]
        # photon count rate, averaged over segment
        photon_rate=f[beam]['sea_ice_segments']['stats']['photon_rate'][idx]
        
        # mean height of histogram
        h_mean=f[beam]['sea_ice_segments']['stats']['hist_mean_h'][idx]
        # median height of histogram
        h_median=f[beam]['sea_ice_segments']['stats']['hist_median_h'][idx]
        # Difference between mean and median
        h_diff=h_mean - h_median
        
        # Normalized height (for every 10 km)
        h_norm = heights - height_sampling(x_atc, heights)
        
        # Sea ice concentration
        sic=f[beam]['sea_ice_segments']['stats']['ice_conc'][idx]

        # Delta time to gps seconds
        atlas_epoch=f[beam]['/ancillary_data/atlas_sdp_gps_epoch'][0]
        temp = convert_time(deltatime + atlas_epoch)    

        df07=pd.DataFrame({'seg_id': sid,
                           'beam': beam, 'lat':lats, 'lon':lons, 'x': x_atc, 'deltatime':deltatime,
                           'height':heights, 'h_mean': h_mean, 'h_median': h_median, 'h_diff': h_diff, 'h_norm': h_norm,
                           'width': width, 'rms': rms, 'n_pulse': n_pulse,
                           'bck_cal': bck_cal, 'bck_r200': bck_r200, 'bck_r25': bck_r25, 'asr': asr_25,
                           'bck_norm': bck_norm, 'hist_w': hist_w, 'ph_rate': photon_rate,
                           'sic': sic, 'stype': stype, 'seg_len': seg_len,
                           'b_azi': b_azi, 'b_ele': b_ele, 's_azi': s_azi, 's_ele': s_ele
                          })

        # Concatenate ATL03 dataframe and time dataframe
        df07 = pd.concat([df07, temp], axis=1).reset_index(drop = True)
        
        df07 = df07[df07['height'] <= maxh]

        if bbox != None:
            df07 = df07[df07['lat'] >= bbox[1]].reset_index(drop = True)
            df07 = df07[df07['lat'] <= bbox[3]].reset_index(drop = True)
            if bbox[0] < bbox[2]:
                df07 = df07[df07['lon'] >= bbox[0]].reset_index(drop = True)
                df07 = df07[df07['lon'] <= bbox[2]].reset_index(drop = True)
            else:
                df07 = df07[(df07['lon'] >= bbox[0]) | (df07['lon'] <= bbox[2])].reset_index(drop = True)

        return df07
    
    