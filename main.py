import pyart
import numpy as np
import nexradaws
import tempfile
import xarray as xr
import datetime
import matplotlib.pyplot as plt


def main():
    
    # Request files by station ID, start date/time, and end date/time (UTC)
    radar_id = 'KVNX'
    start = datetime.datetime(2019, 8, 22, 6, 0) 
    end = datetime.datetime(2019, 8, 22, 12, 0)
    
    # Central lat/lon of ARM SGP site 
    site_latlon = [36.607322, -97.487643]
    
    radarfiles, count_scans = get_radar_from_aws(radar_id, start, end)
    
    ds, scan_times = get_ref_cols(radarfiles, site_latlon, count_scans)
    print(scan_times)



def get_radar_from_aws(radar_id, start, end):
    
    # Create temp file, connect to NEXRAD AWS, download tempfiles locally
    templocation = tempfile.mkdtemp()
    conn = nexradaws.NexradAwsInterface()

    scans = conn.get_avail_scans_in_range(start, end, radar_id)
    print("There are {} scans available between {} and {}\n".format(len(scans), start, end)) 
    
    # Get all the scans that don't end in MDM; count them
    count_scans = 0
    good_scans = []
    for i in range(len(scans)):
        this_str = str(scans[i])
        if this_str[-8::] != 'V06_MDM>':
            good_scans.append(scans[i])
            count_scans +=1
        else:
            i+=1
    
    # Download selected volume scans
    localfiles = conn.download(good_scans, templocation)
    
    return localfiles, count_scans



def get_ref_cols(radarfiles, site_latlon, count_scans):
    
    # Get vertical columns of reflectivity over the ARM SGP site
    
    # Create numpy arrays for sweeps, latitude and longitude of each gate,
    # height at each of the gates, and reflectivity (Z) at each of the gates
    sweep = np.arange(0, 18, 1)
    gate_lat_site = np.nan*np.ma.ones((count_scans,18))
    gate_lon_site = np.nan*np.ma.ones((count_scans,18))
    gate_hgt_site = np.nan*np.ma.ones((count_scans,18))
    Z_site = np.nan*np.ma.ones((count_scans,18))
    scan_times = []
    tol = 0.25
    j = 0 

    # Loop through the volume scans
    for scan in radarfiles.iter_success():
        print('Reading in: '+scan.filename)

        # Read in the volume scans, get the times of the scans
        radar = pyart.io.read(radarfiles.success[j].filepath)
        volume_scan_t = scan.scan_time
        scan_times.append(volume_scan_t)

        # Get rid of duplicate sweeps; only interested in the first sweep at a given height
        # Find the median elevation in each sweep to ID and select only the unique sweeps
        vcp = np.asarray([np.median(el_this_sweep) for el_this_sweep in radar.iter_elevation()], dtype=radar.elevation['data'].dtype)
        close_enough = (vcp/tol).astype('int32')
        unq_el, unq_idx = np.unique(close_enough, return_index=True)

        # Create a new radar object with only the unique sweeps
        myradar = radar.extract_sweeps(unq_idx)

        # Loop through all the sweeps in a volume scan
        for k in range(myradar.nsweeps):
            slice_start, slice_end = myradar.get_start_end(k)
            gate_longitude = myradar.gate_longitude['data'][slice_start:slice_end,:]
            gate_latitude = myradar.gate_latitude['data'][slice_start:slice_end,:]
            gate_altitude = myradar.gate_altitude['data'][slice_start:slice_end,:]

            # Find the gate closest to the desired site 
            # Get the lat,lon,alt,reflectivity of the gate closest to the desired site
            dist = np.sqrt(((gate_latitude - site_latlon[0])**2) + ((gate_longitude - site_latlon[1])**2))
            index = np.where(dist == np.min(dist))
            gate_lat_site[j,k] = gate_latitude[index[0][0], index[1][0]]
            gate_lon_site[j,k] = gate_longitude[index[0][0], index[1][0]]
            gate_hgt_site[j,k] = gate_altitude[index[0][0], index[1][0]]
            Z_site[j,k] = myradar.fields['reflectivity']['data'][slice_start+index[0][0], index[1][0]]

        if j < (count_scans-1):
            j += 1
            del radar
        else:
            print('Done')
            
            # Create an xarray dataset with dimensions time and sweep
            ds = xr.Dataset({'Z_site': (['time', 'sweep'], Z_site),
                        'gate_lat_site': (['time', 'sweep'], gate_lat_site),
                        'gate_lon_site': (['time', 'sweep'], gate_lon_site),
                        'gate_hgt_site': (['time', 'sweep'], gate_hgt_site)
                        },
                        coords={'time': scan_times,'sweep': sweep})
    return ds, scan_times



if __name__ == "__main__":
    main()
