import numpy as np
import pandas as pd
import h5py
import datetime as dt

# Adapted from a notebook by Tyler Sutterly 6/14/2910

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

## Read h5 ATL10 files ========================================================
def get_ATL10data(fileT, maxFreeboard, bbox, beamlist=None):

    f1 = h5py.File(fileT, 'r')

    orient = f1['orbit_info']['sc_orient'][:]  # orientation - 0: backward, 1: forward, 2: transition
    
    if len(orient) > 1:
        print('Transitioning, do not use for science!')
        return [[] for i in beamlist]
    elif (orient == 0):
        beams=['gt1l', 'gt1r', 'gt2l', 'gt2r', 'gt3l', 'gt3r']                
    elif (orient == 1):
        beams=['gt3r', 'gt3l', 'gt2r', 'gt2l', 'gt1r', 'gt1l']
    # (strong, weak, strong, weak, strong, weak)
    # (beam1, beam2, beam3, beam4, beam5, beam6)

    if beamlist == None:
        beams = [ beams[i] for i in [0, 2, 4]]
    else:
        beams = [ beams[i] for i in beamlist ]
    # use only strong beams

    dL = []

    for beam in beams:
        if beam in list(f1.keys()):
            freeboard=f1[beam]['freeboard_beam_segment']['beam_freeboard']['beam_fb_height'][:]

            freeboard_confidence=f1[beam]['freeboard_beam_segment']['beam_freeboard']['beam_fb_confidence'][:]
            freeboard_quality=f1[beam]['freeboard_beam_segment']['beam_freeboard']['beam_fb_quality_flag'][:]
            atlas_epoch=f1['/ancillary_data/atlas_sdp_gps_epoch'][0]
            
            seg_x = f1[beam]['freeboard_beam_segment']['beam_freeboard']['seg_dist_x'][:]

            # Delta time in gps seconds
            delta_time = f1[beam]['freeboard_beam_segment']['beam_freeboard']['delta_time'][:]
            temp = convert_time(delta_time + atlas_epoch)

            # Height segment ID (10 km segments)
            height_segment_id=f1[beam]['freeboard_beam_segment']['beam_freeboard']['height_segment_id'][:]
            height = f1[beam]['freeboard_beam_segment']['height_segments']['height_segment_height'][:]
            mss = f1[beam]['freeboard_beam_segment']['geophysical']['height_segment_mss'][:]
            stype = f1[beam]['freeboard_beam_segment']['height_segments']['height_segment_type'][:]
            slead = np.zeros(len(stype))
            slead[(stype >= 2) & (stype <= 5)] = 1 # Specular leads
            dlead = np.zeros(len(stype))
            dlead[(stype >= 6) & (stype <= 9)] = 1 # Dark leads

            lons=f1[beam]['freeboard_beam_segment']['beam_freeboard']['longitude'][:]
            lats=f1[beam]['freeboard_beam_segment']['beam_freeboard']['latitude'][:]
            
            refsur_ndx=f1[beam]['freeboard_beam_segment']['beam_freeboard']['beam_refsurf_ndx'][:]-1
            refsur = f1[beam]['freeboard_beam_segment']['beam_refsurf_height'][:]
            refsur_height = np.zeros(len(refsur_ndx))
            for i in range(0, len(refsur_ndx)):
                if refsur[refsur_ndx[i]] < 100:
                    refsur_height[i] = refsur[refsur_ndx[i]]
                else:
                    refsur_height[i] = np.nan
            
            lead_ndx = f1[beam]['leads']['ssh_ndx'][:]
            lead_n = f1[beam]['leads']['ssh_n'][:]
            lead = np.zeros(len(refsur_ndx))
            
            for k in range(0, len(lead_ndx)):
                first_ndx = lead_ndx[k]
                lead[first_ndx:first_ndx+lead_n[k]] = 1

            dF = pd.DataFrame({'beam':beam,
                               'lon':lons, 'lat':lats, 'x': seg_x, 'delta_time':delta_time, 
                               'seg_id':height_segment_id, 'height': height, 'freeboard':freeboard,
                               'mss': mss, 'h_ref': refsur_height, 'lead': lead, 'stype': stype,
                               'slead': slead, 'dlead': dlead 
                              })
            dF = pd.concat([dF, temp], axis=1)
            
        else:
            dF = pd.DataFrame(columns=['beam','lon','lat','delta_time',
                            'height_segment_id', 'height', 'freeboard', 'lead',
                            'time', 'year', 'month', 'day', 'hour', 'minute', 'second'])    

        if len(dF) > 0:
            
            dF = dF[(dF['freeboard']>=0)]
            dF = dF[(dF['freeboard']<maxFreeboard)]
            
            if bbox != None:
                dF = dF[dF['lat'] >= bbox[1]].reset_index(drop = True)
                dF = dF[dF['lat'] <= bbox[3]].reset_index(drop = True)

                if bbox[0] < bbox[2]:
                    dF = dF[dF['lon'] >= bbox[0]].reset_index(drop = True)
                    dF = dF[dF['lon'] <= bbox[2]].reset_index(drop = True)
                else:
                    dF = dF[(dF['lon'] >= bbox[0]) | (dF['lon'] <= bbox[2])].reset_index(drop = True)

            # Reset row indexing
            dF=dF.reset_index(drop=True)

            dL.append(dF)
        else:
            dL.append([])              
        
    return dL

def get_ATL10lead(fileT, maxFreeboard, bbox, beamlist=None):
    # Pandas/numpy ATL10 reader
        
    f1 = h5py.File(fileT, 'r')

    orient = f1['orbit_info']['sc_orient']  # orientation - 0: backward, 1: forward, 2: transition

    if orient == 1: # forward
        beams = ['gt3r', 'gt3l', 'gt2r', 'gt2l', 'gt1r', 'gt1l']
    else: # backward
        beams = ['gt1l', 'gt1r', 'gt2l', 'gt2r', 'gt3l', 'gt3r']
    # (strong, weak, strong, weak, strong, weak)
    # (beam1, beam2, beam3, beam4, beam5, beam6)

    if beamlist == None:
        beams = [ beams[i] for i in [0, 2, 4]]
    else:
        beams = [ beams[i] for i in beamlist ]
    # use only strong beams

    dL = []

    for beam in beams:
        if beam in list(f1.keys()):
            lead_height=f1[beam]['leads']['lead_height'][:]
            lead_length=f1[beam]['leads']['lead_length'][:]
            lead_sigma=f1[beam]['leads']['lead_sigma'][:]

            atlas_epoch=f1['/ancillary_data/atlas_sdp_gps_epoch'][0]

            # Delta time in gps seconds
            delta_time = f1[beam]['leads']['delta_time'][:]
            temp = convert_time(delta_time + atlas_epoch)

            # Height segment ID (10 km segments)
            ssh_n=f1[beam]['leads']['ssh_n'][:]
            ssh_ndx=f1[beam]['leads']['ssh_ndx'][:]

            lons=f1[beam]['leads']['longitude'][:]
            lats=f1[beam]['leads']['latitude'][:]

            dF = pd.DataFrame({'beam':beam, 'height':lead_height, 'length': lead_length, 'sigma': lead_sigma,
                               'lon':lons, 'lat':lats, 'delta_time':delta_time, 
                               'ssh_n':ssh_n, 'ssh_ndx': ssh_ndx
                              })
            dF = pd.concat([dF, temp], axis=1)
            
        else:
            dF = pd.DataFrame(columns=['beam', 'height','length', 'sigma', 'delta_time',
                            'ssh_n', 'ssh_ndx', 'year', 'month', 'day',
                            'hour', 'minute', 'second'])

        if len(dF) > 0:

            if bbox != None:
                dF = dF[dF['lat'] >= bbox[1]].reset_index(drop = True)
                dF = dF[dF['lat'] <= bbox[3]].reset_index(drop = True)

                if bbox[0] < bbox[2]:
                    dF = dF[dF['lon'] >= bbox[0]].reset_index(drop = True)
                    dF = dF[dF['lon'] <= bbox[2]].reset_index(drop = True)
                else:
                    dF = dF[(dF['lon'] >= bbox[0]) | (dF['lon'] <= bbox[2])].reset_index(drop = True)

            # Reset row indexing
            dF=dF.reset_index(drop=True)

            dL.append(dF)
        else:
            dL.append([])              
        
    return dL
