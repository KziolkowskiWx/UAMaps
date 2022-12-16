from datetime import datetime, timedelta
from siphon.simplewebservice.wyoming import WyomingUpperAir
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as feat
#Uncomment the two lines below if running in cron 
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from mpl_toolkits.axes_grid1 import make_axes_locatable
from metpy.plots import simple_layout, StationPlot, StationPlotLayout
from metpy.calc import equivalent_potential_temperature as te
from metpy.calc import relative_humidity_from_specific_humidity as calc_rh
from metpy.calc import dewpoint_from_relative_humidity as calc_td
from metpy.units import units
import metpy.calc as mpcalc
import numpy as np
import xarray as xr
import scipy.ndimage as ndimage
from optparse import OptionParser
import time
from pathlib import Path
from os import path
from os import getcwd
from siphon.catalog import TDSCatalog
from io import BytesIO
from PIL import Image


def main():

    usage="usage: %prog \n example usage for 12z maps: python uaplot.py --latest --td --te, or python uaplot.py --date=20220523 --td. Use --levels=850,700 to plot specific levels"
    parser = OptionParser(conflict_handler="resolve", usage=usage, version="%prog 1.0 By Kyle Ziolkowski")
    parser.add_option("-h", "--help", dest="help", help="--am or --pm for 12z or 00z obs")
    parser.add_option("--latest", "--latest", dest="latest",  action="store_true", help="Get 00z obs", default=False)
    parser.add_option("--date", "--date", dest="date",type="str",help="date in format YYYYMMDDHH. Note: HH must be 12 or 00 for 12z or 00z maps", default=False)
    parser.add_option("--td", "--td", dest='td', action="store_true", help="Plot dewpoint instead of dewpoint depression", default=False)
    parser.add_option("--te", "--te",dest='te', action="store_true", help="Plot Theta-e instead of temperatures for 925/850/700 mb", default=False)
    parser.add_option("--levels", "--levels",dest='levels',type="str", help="Levels in which to plot. Use levels=850,500 to plot specific levels. This option is not required to plot all levels (250, 300, 500, 700, 850, 925).", default='All')
    parser.add_option("--compress_output", "--compress_output", dest="compress",action="store_true", help="Compress the output PNGs when saving; will add extra to the runtime.", default=False)
    parser.add_option("--cwd", "--cwd", dest="cwd", action="store_true",
                      help="Flag to use the current working directory instead of ~/UAMaps/.", default=False)
    parser.add_option("--png_colours", "--png_colours", dest="png_colours",
                      type="int", help="Number of colours to use in PNG comprssion.", default=256)
    parser.add_option("--thumbnails", "--thumbnails", dest="thumbnails",
                      action="store_true", help="Enable to generate thumbnail output as well.", default=False)
    parser.add_option("--thumbnail_size", "--thumbnail_size", dest="thumbnail_size",
                      type="int", help="Specify the maximum pixel dimension of a generated thumbnail.", default=640)
    parser.add_option("--long_filenames", "--long_filenames", dest="long_filenames",
                      action="store_true", help="Enable to create files with the full datestring in the output filenames.", default=False)
					  
    (opt, arg) = parser.parse_args()

    if  opt.latest == False and opt.date == False:
        parser.error('No option for plotting selected. Please select --latest for latest UAMaps or --date=YYYYMMDDHH for a specific time.')
    td_option = opt.td
    te_option = opt.te
    if opt.date != False:
        input_date = datetime.strptime(opt.date, '%Y%m%d%H').date()
        hour = datetime.strptime(opt.date, '%Y%m%d%H').hour
    if opt.latest != False:
        input_date = datetime.utcnow().date()
        #Get the right hour
        if datetime.utcnow().hour >= 12 and datetime.utcnow().hour <= 23:
            hour = 12
        else:
            hour = 00
    if opt.levels == 'All':
        levels = [250, 300, 500, 700, 850, 925]
    else:
        levels = [int(i) for i in opt.levels.split(',')]
        

    
    start = time.time()
    if (opt.cwd):
        path_root = getcwd()
    else:
        path_root = Path.home().as_posix() + "/UAMaps/"

    station_file = path_root + '/ua_station_list.csv'
    save_dir = path_root + '/maps/' #Change the string to choose where to save the file. 
    dt = datetime.strptime(input_date.strftime('%Y%m%d') + str(hour), '%Y%m%d%H')
    date = dt - timedelta(hours=6) #Go back 6 hours to for 18z Objective Analysis.
    ds = xr.open_dataset('https://thredds.ucar.edu/thredds/dodsC/grib/NCEP/GFS/Global_0p5deg_ana/GFS_Global_0p5deg_ana_{0:%Y%m%d}_{0:%H}00.grib2'.format(date)).metpy.parse_cf()
    #Added to handle new GFS analysis files. GFS files from THREDDS now contain a bunch of dates
    ds = ds.sel(time=date) #This grabs the proper date for analysis.
    uadata, stations = getData(station_file, dt, hour)
    print('Working on maps:')
    for level in levels:
        print ("    Processing {}...".format(level))
        data = generateData(uadata, stations, level)
        uaPlot(data, level, dt, save_dir, ds, hour, td_option, te_option, opt.date, opt.compress, opt.png_colours, opt.thumbnails, opt.thumbnail_size, opt.long_filenames)
    end = time.time()
    total_time = round(end-start, 2)
    print('Process Complete..... Total time = {}s'.format(total_time))
    


def getData(station_file, date, hh):
    """
    This function will make use of Siphons Wyoming Upperair utility. Will pass
    the function a list of stations and data will be downloaded as pandas a 
    pandas dataframe and the corresponding lats and lons will be placed into
    a dictionary along with the data. 
    """
    
    print ('Getting station data:')
    station_data = pd.read_csv(station_file)
    stations, lats, lons = station_data['station_id'].values, station_data['lat'].values, station_data['lon'].values
    stations = list(stations)    
    # date = datetime.utcnow()
    # date = datetime(date.year, date.month, date.day, hh)
    data = {} #a dictionary for our data
    station_list = []
    total_stations = 0
    data_success = 0
    for station, lat, lon in zip(stations, lats, lons):
        print ('    ' + station, end='')
        total_stations += 1
        try:
            df = WyomingUpperAir.request_data(date, station)
            data[station] = [df, lat, lon]
            station_list += [station]
            data_success += 1
            print('')
        except:
            print ('...failed.')
            pass
    print ('Data retrieved. Fetched ' + str(data_success) + '/' + str(total_stations) + "(" + str(round(100*(data_success/total_stations))) + "%) of requested stations.")
    return data, station_list


def generateData(data, stations, level):
    """
    Test the data and put it into a an array to so it can be passed to dataDict
    """
    
    temp_c = []  
    dpt = []
    u = []
    v = []
    h= []
    p = []   
    lats = []
    lons = []
    
    for station in stations:
        t, td, u_wind, v_wind, height = getLevels(data[station][0], level)
        temp_c += [t]
        dpt += [td]
        u += [u_wind]
        v += [v_wind]
        h += [height]
        p += [level]
        lats += [data[station][1]]
        lons += [data[station][2]]
    temp = np.array(temp_c, dtype=float)    
    dewp = np.array(dpt, dtype=float)
    uwind = np.array(u, dtype=float)
    vwind = np.array(v, dtype=float)     
    data_array = np.array([temp, dewp, uwind, vwind, lats, lons, h, p])
        
    return data_array
    
def getLevels(df, level):
    """
    Get the 925, 850, 700, 500, 300, and 250 mb levels called by generateData
    """   

    level = df.loc[df['pressure'] == level]
        
    t, td, u, v, h, p = level['temperature'].values, level['dewpoint'].values, \
                        level['u_wind'].values, level['v_wind'].values, \
                        level['height'].values, level['pressure']
    #check to see if the data exits and create it. If not label as np.nan
    try:
        temp = t[0]
    except:
        temp = np.nan
    try:
        dwpt = td[0]
    except:
        dwpt = np.nan
    try:
        u_wind = u[0]
    except:
        u_wind = np.nan
    try:
        v_wind = v[0]
    except:
        v_wind = np.nan
    try:
        height = h[0]
    except:
        height = np.nan

    return temp, dwpt, u_wind, v_wind, height
    
def dataDict(data_arr):
    """In order to plot the data using MetPy StationPlot, we need to put
    the data into dictionaries, which will be done here. We will also assign
    the data its units here as well."""  
    
    
    #Container for the data
    data = dict()

    data['longitude'] = data_arr[5]
    data['latitude'] = data_arr[4]
    data['air_temperature'] = data_arr[0] * units.degC
    data['dew_point_temperature'] = data_arr[1] * units.degC
    data['eastward_wind'], data['northward_wind'] = data_arr[2] * units('knots') , data_arr[3] * units('knots')
    data['height'] = data_arr[6] * units('meters')
    data['pressure'] = data_arr[7] * units('hPa')
    data['thetae'] = te(data['pressure'], data['air_temperature'], data['dew_point_temperature']) 
    data['tdd'] = (data_arr[0] - data_arr[1]) * units.degC

    return data    



def mapbackground():

    globe = ccrs.Globe(ellipse='sphere', semimajor_axis=6371200.,
                       semiminor_axis=6371200.)
    proj = ccrs.Stereographic(central_longitude=-105., 
                               central_latitude=90., globe=globe,
                               true_scale_latitude=60)
    # Plot the image
    fig = plt.figure(figsize=(40, 40))
    ax = fig.add_subplot(1, 1, 1, projection=proj)
    state_boundaries = feat.NaturalEarthFeature(category='cultural',
                                            name='admin_1_states_provinces_lines',
                                            scale='10m', facecolor='none')
    coastlines = feat.NaturalEarthFeature('physical', 'coastline', '50m', facecolor='none')
    lakes = feat.NaturalEarthFeature('physical', 'lakes', '50m', facecolor='none')
    countries = feat.NaturalEarthFeature('cultural', 'admin_0_countries', '50m', facecolor='none')
    ax.add_feature(state_boundaries, zorder=2, edgecolor='grey')
    ax.add_feature(lakes, zorder=2, edgecolor='grey')
    ax.add_feature(coastlines, zorder=2, edgecolor='grey')
    ax.add_feature(lakes, zorder=2, edgecolor='grey')
    ax.add_feature(countries, zorder=2, edgecolor='grey') 
    ax.coastlines(resolution='50m', zorder=2, color='grey')    
    ax.set_extent([-132., -70, 26., 80.], ccrs.PlateCarree())

    return ax


def uaPlot(data, level, date, save_dir, ds, hour, td_option, te_option, date_option, image_compress, png_colours, thumbnails, thumbnail_size, long_filenames):

    custom_layout = StationPlotLayout()
    custom_layout.add_barb('eastward_wind', 'northward_wind', units='knots')
    custom_layout.add_value('NW', 'air_temperature', fmt='.0f', units='degC', color='darkred')

    #decide on the height format based on the level
    if level == 250 or level == 300:
        custom_layout.add_value('NE', 'height', fmt=lambda v: format(v, '1')[1:4], units='m', color='black')
        cint = 120
        tint = 5
        lats = ds.lat.sel(lat=slice(85, 15)).values
        lons = ds.lon.sel(lon=slice(360-200, 360-10)).values
        pres = ds['isobaric'].values[:] * units('Pa')
        # tmpk_var = ds.Temperature_isobaric.metpy.sel(lat=slice(85, 15), lon=slice(360-200, 360-10)).squeeze()
        # smooth_tmpc = (mpcalc.smooth_n_point(tmpk_var, 9, 10).to('degC')).squeeze()
        tmpk = ds.Temperature_isobaric.metpy.sel(vertical=level*100, lat=slice(85, 15), lon=slice(360-200, 360-10))*units.degK
        smooth_tmpc = (mpcalc.smooth_n_point(tmpk.data, 9, 10)).to('degC').squeeze()
        # thta = mpcalc.potential_temperature(pres[:, None, None], tmpk_smooth)
        # uwnd_var = ds['u-component_of_wind_isobaric'].metpy.sel(lat=slice(85, 15), lon=slice(360-200, 360-10)).squeeze()
        # vwnd_var = ds['v-component_of_wind_isobaric'].metpy.sel(lat=slice(85, 15), lon=slice(360-200, 360-10)).squeeze()
        # uwnd = mpcalc.smooth_n_point(uwnd_var, 9, 2)
        # vwnd = mpcalc.smooth_n_point(vwnd_var, 9, 2)
        # dx, dy = mpcalc.lat_lon_grid_deltas(lons, lats)
        # # pv = mpcalc.potential_vorticity_baroclinic(thta, pres[:, None, None], uwnd, vwnd,
        # #                                    dx[None, :, :], dy[None, :, :],
        # #                                    lats[None, :, None] * units('degrees'))
        # div = mpcalc.divergence(uwnd, vwnd, dx[None, :, :], dy[None, :, :], dim_order='yx')
        level_idx = list(pres.m).index(((level * units('hPa')).to(pres.units)).m)

    if level == 500:
        custom_layout.add_value('NE', 'height', fmt=lambda v: format(v, '1')[0:3], units='m', color='black')
        cint = 60
        tint =5
        lats = ds.lat.sel(lat=slice(85, 15)).values
        lons = ds.lon.sel(lon=slice(360-200, 360-10)).values
        hght = ds.Geopotential_height_isobaric.metpy.sel(vertical=level * 100,  lat=slice(85, 15), lon=slice(360-200, 360-10))*units.hPa
        smooth_hght = mpcalc.smooth_n_point(hght, 9, 10).squeeze()
        tmpk = ds.Temperature_isobaric.metpy.sel(vertical=level*100, lat=slice(85, 15), lon=slice(360-200, 360-10))*units.degK
        smooth_tmpc = (mpcalc.smooth_n_point(tmpk.data, 9, 10)).to('degC').squeeze()
        abs_vort = ds.Absolute_vorticity_isobaric.metpy.sel(vertical=level*100,lat=slice(85, 15), lon=slice(360-200, 360-10)).squeeze()
        avort = ndimage.gaussian_filter(abs_vort, sigma=3, order=0) * units('1/s')

    if level == 700:
        custom_layout.add_value('NE', 'height', fmt=lambda v: format(v, '1')[1:4], units='m', color='black')
        custom_layout.add_value('SW', 'tdd', units='degC', color='green')
        custom_layout.add_value('SE', 'thetae', units='degK', color='orange')
        temps = 'Tdd, and Temperature'
        cint = 30
        tint=4
    if level == 850:
        custom_layout.add_value('NE', 'height', fmt=lambda v: format(v, '1')[1:4], units='m', color='black')
        if td_option == True:
            custom_layout.add_value('SW', 'dew_point_temperature', units='degC', color='green')
            temps = 'Td'
        if td_option == False:
            custom_layout.add_value('SW', 'tdd', units='degC', color='green')
            temps = 'Tdd'
        # custom_layout.add_value('SW', 'tdd', units='degC', color='green')
        # temps = 'Tdd, and Theta-e'
        custom_layout.add_value('SE', 'thetae', units='degK', color='orange')
        cint = 30
        tint = 4
    if level == 925:
        custom_layout.add_value('NE', 'height', fmt=lambda v: format(v, '1')[1:4], units='m', color='black') 
        if td_option == True:
            custom_layout.add_value('SW', 'dew_point_temperature', units='degC', color='green')
            temps = 'Td, and Temperature'
        if td_option == False:
            custom_layout.add_value('SW', 'tdd', units='degC', color='green')
            temps = 'Tdd, and Temperature'
        custom_layout.add_value('SE', 'thetae', units='degK', color='orange')
        cint = 30
        tint = 4
 
    globe = ccrs.Globe(ellipse='sphere', semimajor_axis=6371200.,
                       semiminor_axis=6371200.)
    proj = ccrs.Stereographic(central_longitude=-105., 
                               central_latitude=90., globe=globe,
                               true_scale_latitude=60)
    # Plot the image
    fig = plt.figure(figsize=(30, 30))
    ax = fig.add_subplot(1, 1, 1, projection=proj)
    state_boundaries = feat.NaturalEarthFeature(category='cultural',
                                            name='admin_1_states_provinces_lines',
                                            scale='10m', facecolor='none')
    coastlines = feat.NaturalEarthFeature('physical', 'coastline', '50m', facecolor='none')
    lakes = feat.NaturalEarthFeature('physical', 'lakes', '50m', facecolor='none')
    countries = feat.NaturalEarthFeature('cultural', 'admin_0_countries', '50m', facecolor='none')
    ax.add_feature(state_boundaries, zorder=2, edgecolor='grey')
    ax.add_feature(lakes, zorder=2, edgecolor='grey')
    ax.add_feature(coastlines, zorder=2, edgecolor='grey')
    ax.add_feature(lakes, zorder=2, edgecolor='grey')
    ax.add_feature(countries, zorder=2, edgecolor='grey') 
    ax.coastlines(resolution='50m', zorder=2, color='grey')    
    ax.set_extent([-132., -70, 26., 80.], ccrs.PlateCarree())
    #ax.set_extent([-122., -60, 34., 60.], ccrs.PlateCarree())

    stationData = dataDict(data) 
    stationplot = StationPlot(ax, stationData['longitude'], stationData['latitude'],
                              transform=ccrs.PlateCarree(), fontsize=26)
    custom_layout.plot(stationplot, stationData)

    # Plot Solid Contours of Geopotential Height
    #Get lat and lon info 
    lons = ds.lon.sel(lon=slice(360-200, 360-10)).values
    lats = ds.lat.sel(lat=slice(85, 15)).values
    hght = ds.Geopotential_height_isobaric.metpy.sel(vertical=level * 100,  lat=slice(85, 15), lon=slice(360-200, 360-10))*units.hPa
    smooth_hght = mpcalc.smooth_n_point(hght, 9, 10).squeeze()
    cs = ax.contour(lons, lats, smooth_hght.values, range(0, 20000, cint), colors='black', transform=ccrs.PlateCarree())
    clabels = plt.clabel(cs, fmt='%d', colors='white', inline_spacing=5, use_clabeltext=True, fontsize=14)

    # Contour labels with black boxes and white text
    for t in clabels:
        t.set_bbox({'facecolor': 'black', 'pad': 4})
        t.set_fontweight('heavy')

    #Check levels for different contours
    if level == 250 or level == 300:
        # Plot Dashed Contours of Temperature
        cs2 = ax.contour(hght.lon, hght.lat, smooth_tmpc.m, range(-60, 51, tint), colors='red', transform=ccrs.PlateCarree())
        clabels = plt.clabel(cs2, fmt='%d', colors='red', inline_spacing=5, use_clabeltext=True, fontsize=22)
        # # Set longer dashes than default
        # for c in cs2.collections:
        #     c.set_dashes([(0, (5.0, 3.0))])
        temps = 'Temperature'
        # clevs_pv = np.arange(0, 25, 1)
        # Plot the colorfill of divergence, scaled 10^5 every 1 s^1
        # clevs_div1 = np.arange(2, 16, 1)
        # clevs_div2 = np.arange(-15, -1, 1)
        # cs1 = ax.contourf(lons, lats, div[level_idx]*1e5, clevs_div, cmap='PuOr', extend='both', transform=ccrs.PlateCarree())
        # cs1 = ax.contour(lons, lats, div[level_idx]*1e5, clevs_div1, linestyle='dashed', colors='purple', extend='both', transform=ccrs.PlateCarree())
        # cs2 = ax.contour(lons, lats, div[level_idx]*1e5, clevs_div2, linestyle='dashed', colors='orange', extend='both', transform=ccrs.PlateCarree())
        # plt.colorbar(cs1, orientation='vertical', pad=0, aspect=50, extendrect=True)


    if level == 500:
        # Plot Dashed Contours of Temperature
        cs2 = ax.contour(lons, lats, smooth_tmpc.m, range(-60, 51, tint), colors='red', transform=ccrs.PlateCarree())
        clabels = plt.clabel(cs2, fmt='%d', colors='red', inline_spacing=5, use_clabeltext=True, fontsize=30)
        # Set longer dashes than default
        for c in cs2.collections:
            c.set_dashes([(0, (5.0, 3.0))])
        temps = 'Temperature, and Absolute Vorticity $*10^5$ ($s^{-1}$)'

        #plot vorticity
        clevvort500 = np.arange(-8, 50, 2)
        cs1 = ax.contour(lons, lats, avort.m*10**5, clevvort500, colors='grey', linestyles='dashed', linewidths=1.5, transform=ccrs.PlateCarree())
        plt.clabel(cs1, fontsize=10, inline=1, inline_spacing=10, fmt='%i', rightside_up=True, use_clabeltext=True)

    if level == 700 or level == 850 or level == 925:
        tmpk = ds.Temperature_isobaric.metpy.sel(vertical=level*100, lat=slice(85, 15), lon=slice(360-200, 360-10))*units.degK
        smooth_tmpc = (mpcalc.smooth_n_point(tmpk.data, 9, 10)).to('degC').squeeze()

        if te_option == True:
        #Calculate Theta-e
            rh = ds.Relative_humidity_isobaric.metpy.sel(vertical=level*100, lat=slice(85, 15), lon=slice(360-200, 360-10))
            td = mpcalc.dewpoint_from_relative_humidity(tmpk, rh)
            te = mpcalc.equivalent_potential_temperature(level*units.hPa, tmpk, td)
            smooth_te = mpcalc.smooth_n_point(te.data, 9,10).squeeze()
            zeroline = ax.contour(lons, lats, smooth_tmpc.m, 0, colors='red', linestyles='solid', linewidths=3, transform=ccrs.PlateCarree())
            zeroline_label = plt.clabel(zeroline, fmt='%d', colors='black', inline_spacing=5, use_clabeltext=True, fontsize=30)
            if level == 700:
                dingle_line = ax.contour(lons, lats, smooth_tmpc.m, [10], colors='brown', linestyles='solid', linewidths=3, transform=ccrs.PlateCarree())
                dingle_line_label = plt.clabel(dingle_line, fmt='%d', colors='black', inline_spacing=5, use_clabeltext=True, fontsize=30)
            cs2 = ax.contour(lons, lats, smooth_te.m, range(210, 360, tint), colors='orange', transform=ccrs.PlateCarree())
            clabels = plt.clabel(cs2, fmt='%d', colors='orange', inline_spacing=5, use_clabeltext=True, fontsize=22)
            temps = temps + ', and Theta-e'
            # Set longer dashes than default
            for c in cs2.collections:
                c.set_dashes([(0, (5.0, 3.0))])   
  
        else:
            cs2 = ax.contour(lons, lats, smooth_tmpc.m, range(
                4, 56, tint), colors='red', transform=ccrs.PlateCarree())
            cs3 = ax.contour(lons, lats, smooth_tmpc.m, range(-50, -4,
                                                              tint), colors='blue', transform=ccrs.PlateCarree())
            zeroline = ax.contour(lons, lats, smooth_tmpc.m, 0, colors='red',
                                  linestyles='solid', linewidths=3, transform=ccrs.PlateCarree())
            if level == 700:
                dingle_line = ax.contour(lons, lats, smooth_tmpc.m, [
                                         10], colors='brown', linestyles='solid', linewidths=3, transform=ccrs.PlateCarree())
                dingle_line_label = plt.clabel(
                    dingle_line, fmt='%d', colors='black', inline_spacing=5, use_clabeltext=True, fontsize=30)
            zeroline_label = plt.clabel(
                zeroline, fmt='%d', colors='black', inline_spacing=5, use_clabeltext=True, fontsize=30)
            clabels2 = plt.clabel(cs2, fmt='%d', colors='black',
                                  inline_spacing=5, use_clabeltext=True, fontsize=30)
            clabels3 = plt.clabel(cs3, fmt='%d', colors='black',
                                  inline_spacing=5, use_clabeltext=True, fontsize=30)
            temps = temps + ', and Temperature'
        # Set longer dashes than default
            for c in cs2.collections:
                c.set_dashes([(0, (5.0, 3.0))])
            for c in cs3.collections:
                c.set_dashes([(0, (5.0, 3.0))])

    dpi = plt.rcParams['savefig.dpi'] = 255
    text = AnchoredText(str(level) + 'mb Wind, Heights, ' + temps +
                        ' Valid: {0:%Y-%m-%d} {0:%H}:00 UTC'.format(date), loc=3, frameon=True, prop=dict(fontsize=30))
    ax.add_artist(text)
    plt.tight_layout()
    if date_option != False:
        save_fname = str(level) + 'mb_{0:%Y%m%d%H%M}z.png'.format(date)
    else:
        if hour == 12:
            save_fname = str(level) + 'mb_' + str(hour) + 'z.png'
            thumb_fname = str(level) + 'mb_' + str(hour) + 'z_thumbnail.png'
        if hour == 0:
            save_fname = str(level) + 'mb_' + str(hour) + '0z.png'
            thumb_fname = str(level) + 'mb_' + str(hour) + '0z_thumbnail.png'
    if image_compress is False:
        plt.savefig(save_dir + "/" + save_fname, dpi=dpi, bbox_inches='tight')
    else:
        temp_img = BytesIO()
        plt.savefig(temp_img, dpi=dpi, bbox_inches='tight')
        temp_img.seek(0)
        im = Image.open(temp_img)
        im2 = im.convert('RGB').convert(
            'P', palette=Image.ADAPTIVE, colors=png_colours)
        im2.save(save_dir + "/" + save_fname, format='PNG', optimize=True)
        if (thumbnails != False):
            im2.thumbnail((thumbnail_size, thumbnail_size), Image.ANTIALIAS)
            im2.save(save_dir + "/" + thumb_fname, format='PNG', optimize=True)
    print('        Image saved.')
    # plt.show()

if __name__ == '__main__':
    main()