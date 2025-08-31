# Spatiotemporal Analysis of the Impacts of Early 2024 Wildfires in North America: A Study of Correlations Between Meteorological Conditions, Air Quality, and Wind Dynamics in Stinnett and Canadian, TX


import xarray as xr

v10m= r"C:\Users\hp\OneDrive\Desktop\10mv.nc"
u10m= r"C:\Users\hp\OneDrive\Desktop\10mu.nc"
AOD500nm = r"C:\Users\hp\OneDrive\Desktop\AOD500nm.nc"
t2m = r"C:\Users\hp\OneDrive\Desktop\t2m.nc"
coplev = r"C:\Users\hp\OneDrive\Desktop\coplev.nc"
comlev = r"C:\Users\hp\OneDrive\Desktop\comlev.nc"

dsv10m = xr.open_dataset(u10m)
dsv10m = xr.open_dataset(v10m)
dsAOD500nm = xr.open_dataset(AOD500nm)
dst2m = xr.open_dataset(t2m)
dscoplev = xr.open_dataset(coplev)
dscomlev = xr.open_dataset(comlev)

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr

dsAOD500nm['valid_time'] = xr.decode_cf(dsAOD500nm).valid_time

start_date = '2024-02-26'
end_date = '2024-03-16'
period_ds = dsAOD500nm.sel(valid_time=slice(start_date, end_date))
mean_period = period_ds['aod550'].mean(dim='valid_time')

fig = plt.figure(figsize=(14,8))
ax = plt.axes(projection=ccrs.PlateCarree())

mean_period.plot(ax=ax, cmap='magma', cbar_kwargs={'label': 'AOD 550 nm'})

ax.add_feature(cfeature.BORDERS, linewidth=1, edgecolor='white')
ax.add_feature(cfeature.COASTLINE, linewidth=1, edgecolor='white')

gl = ax.gridlines(draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {'size': 12}
gl.ylabel_style = {'size': 12}

plt.title(f"Mean AOD550 from {start_date} to {end_date}", fontsize=16)

plt.show()


import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime

ds_co_model = ds_co_model.assign_coords(
    longitude=((ds_co_model.longitude + 180) % 360 - 180)
)

t2m_f = (ds_t2m.t2m - 273.15) * 9/5 + 32  
co_pressure_ppb = ds_co_pressure.co.squeeze() * 1e9  # kg/kg to ppb
co_model_ppb = ds_co_model.co.squeeze() * 1e9  # kg/kg to ppb

daily_aod = ds_aod.aod550.resample(valid_time='D').mean()
daily_t2m = t2m_f.resample(valid_time='D').mean()
daily_co_pressure = co_pressure_ppb.resample(valid_time='D').mean()
daily_co_model = co_model_ppb.resample(valid_time='D').mean()

u10_daily = ds_u10.u10.resample(valid_time='D').mean()
v10_daily = ds_v10.v10.resample(valid_time='D').mean()

plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['figure.titleweight'] = 'bold'

fig, axs = plt.subplots(2, 2, figsize=(18, 14),
                       subplot_kw={'projection': ccrs.PlateCarree()})
ax1, ax2, ax3, ax4 = axs.flatten()


plot_params = {
    'AOD550': {'cmap': 'YlOrRd', 'units': '', 'title': 'Daily Mean Aerosol Optical Depth at 550nm'},
    'Temperature': {'cmap': 'coolwarm', 'units': '°F', 'title': 'Daily Mean Temperature at 2m'},
    'CO Pressure': {'cmap': 'plasma', 'units': 'ppb', 'title': 'Daily Mean Carbon Monoxide at 1000 hPa'},
    'CO Model': {'cmap': 'viridis', 'units': 'ppb', 'title': 'Daily Mean Carbon Monoxide at Model Level 1'}
}


stride = 4  


target_date = daily_aod.valid_time[0].values

# AOD550 
ax = ax1
data_to_plot = daily_aod.sel(valid_time=target_date)
p = data_to_plot.plot(
    ax=ax,
    cmap=plot_params['AOD550']['cmap'],
    add_colorbar=False,
    transform=ccrs.PlateCarree()
)
u_data = u10_daily.sel(valid_time=target_date)
v_data = v10_daily.sel(valid_time=target_date)
u_sub = u_data.isel(latitude=slice(None, None, stride), longitude=slice(None, None, stride))
v_sub = v_data.isel(latitude=slice(None, None, stride), longitude=slice(None, None, stride))
quiver = ax.quiver(u_sub.longitude, u_sub.latitude, u_sub.values, v_sub.values,
                   color='black', scale=200, transform=ccrs.PlateCarree(), width=0.003)
qk = ax.quiverkey(quiver, 0.85, 0.95, 5, '5 m/s', labelpos='E',
                  coordinates='axes', color='black', fontproperties={'size': 10, 'weight': 'bold'})
ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=2, edgecolor='white')
ax.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=2, edgecolor='white')
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {'size': 8, 'weight': 'bold'}
gl.ylabel_style = {'size': 8, 'weight': 'bold'}
ax.set_title(plot_params['AOD550']['title'], fontsize=14, fontweight='bold')
cbar_var = plt.colorbar(p, ax=ax, orientation='vertical', pad=0.05, shrink=0.8)
cbar_var.set_label(f'{plot_params["AOD550"]["units"]}', fontsize=12, fontweight='bold')
for label in cbar_var.ax.get_yticklabels():
    label.set_fontweight('bold')

# Temperature 
ax = ax2
data_to_plot = daily_t2m.sel(valid_time=target_date)
p = data_to_plot.plot(
    ax=ax,
    cmap=plot_params['Temperature']['cmap'],
    add_colorbar=False,
    transform=ccrs.PlateCarree()
)
u_data = u10_daily.sel(valid_time=target_date)
v_data = v10_daily.sel(valid_time=target_date)
u_sub = u_data.isel(latitude=slice(None, None, stride), longitude=slice(None, None, stride))
v_sub = v_data.isel(latitude=slice(None, None, stride), longitude=slice(None, None, stride))
quiver = ax.quiver(u_sub.longitude, u_sub.latitude, u_sub.values, v_sub.values,
                   color='black', scale=200, transform=ccrs.PlateCarree(), width=0.003)
ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=2, edgecolor='white')
ax.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=2, edgecolor='white')
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {'size': 8, 'weight': 'bold'}
gl.ylabel_style = {'size': 8, 'weight': 'bold'}
ax.set_title(plot_params['Temperature']['title'], fontsize=14, fontweight='bold')
cbar_var = plt.colorbar(p, ax=ax, orientation='vertical', pad=0.05, shrink=0.8)
cbar_var.set_label(f'{plot_params["Temperature"]["units"]}', fontsize=12, fontweight='bold')
for label in cbar_var.ax.get_yticklabels():
    label.set_fontweight('bold')

# CO Pressure 
ax = ax3
data_to_plot = daily_co_pressure.sel(valid_time=target_date)
p = data_to_plot.plot(
    ax=ax,
    cmap=plot_params['CO Pressure']['cmap'],
    add_colorbar=False,
    transform=ccrs.PlateCarree()
)
u_data = u10_daily.sel(valid_time=target_date)
v_data = v10_daily.sel(valid_time=target_date)
u_sub = u_data.isel(latitude=slice(None, None, stride), longitude=slice(None, None, stride))
v_sub = v_data.isel(latitude=slice(None, None, stride), longitude=slice(None, None, stride))
quiver = ax.quiver(u_sub.longitude, u_sub.latitude, u_sub.values, v_sub.values,
                   color='black', scale=200, transform=ccrs.PlateCarree(), width=0.003)
ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=2, edgecolor='white')
ax.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=2, edgecolor='white')
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {'size': 8, 'weight': 'bold'}
gl.ylabel_style = {'size': 8, 'weight': 'bold'}
ax.set_title(plot_params['CO Pressure']['title'], fontsize=14, fontweight='bold')
cbar_var = plt.colorbar(p, ax=ax, orientation='vertical', pad=0.05, shrink=0.8)
cbar_var.set_label(f'{plot_params["CO Pressure"]["units"]}', fontsize=12, fontweight='bold')
for label in cbar_var.ax.get_yticklabels():
    label.set_fontweight('bold')

#CO Model 
ax = ax4
data_to_plot = daily_co_model.sel(valid_time=target_date)
p = data_to_plot.plot(
    ax=ax,
    cmap=plot_params['CO Model']['cmap'],
    add_colorbar=False,
    transform=ccrs.PlateCarree()
)
u_data = u10_daily.sel(valid_time=target_date)
v_data = v10_daily.sel(valid_time=target_date)
u_sub = u_data.isel(latitude=slice(None, None, stride), longitude=slice(None, None, stride))
v_sub = v_data.isel(latitude=slice(None, None, stride), longitude=slice(None, None, stride))
quiver = ax.quiver(u_sub.longitude, u_sub.latitude, u_sub.values, v_sub.values,
                   color='black', scale=200, transform=ccrs.PlateCarree(), width=0.003)
ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=2, edgecolor='white')
ax.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=2, edgecolor='white')
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {'size': 8, 'weight': 'bold'}
gl.ylabel_style = {'size': 8, 'weight': 'bold'}
ax.set_title(plot_params['CO Model']['title'], fontsize=14, fontweight='bold')
cbar_var = plt.colorbar(p, ax=ax, orientation='vertical', pad=0.05, shrink=0.8)
cbar_var.set_label(f'{plot_params["CO Model"]["units"]}', fontsize=12, fontweight='bold')
for label in cbar_var.ax.get_yticklabels():
    label.set_fontweight('bold')


date_str = datetime.utcfromtimestamp(target_date.astype('O')/1e9).strftime('%Y-%m-%d')
plt.suptitle(f'Daily Mean Values for {date_str} with Wind Direction', fontsize=16, y=0.98, fontweight='bold')

plt.show()




plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['figure.titleweight'] = 'bold'

ds_co_model = ds_co_model.assign_coords(
    longitude=((ds_co_model.longitude + 180) % 360 - 180)
)

t2m_f = (ds_t2m.t2m - 273.15) * 9/5 + 32  # Kelvin to Fahrenheit
co_pressure_ppb = ds_co_pressure.co.squeeze() * 1e9  # kg/kg to ppb
co_model_ppb = ds_co_model.co.squeeze() * 1e9  # kg/kg to ppb

daily_aod = ds_aod.aod550.resample(valid_time='D').mean()
daily_t2m = t2m_f.resample(valid_time='D').mean()
daily_co_pressure = co_pressure_ppb.resample(valid_time='D').mean()
daily_co_model = co_model_ppb.resample(valid_time='D').mean()

locations = {
    'Stinnett': {'lat': 35.823, 'lon': -101.444},
    'Canadian, TX': {'lat': 35.913, 'lon': -100.383}
}

colors = {
    'Stinnett': 'red',
    'Canadian, TX': 'blue'
}


fig, axs = plt.subplots(4, 1, figsize=(14, 16))
ax1, ax2, ax3, ax4 = axs.flatten()

#AOD550 
ax = ax1


lat_idx = np.abs(daily_aod.latitude - locations['Stinnett']['lat']).argmin()
lon_idx = np.abs(daily_aod.longitude - locations['Stinnett']['lon']).argmin()
exact_lat = daily_aod.latitude.values[lat_idx]
exact_lon = daily_aod.longitude.values[lon_idx]
ts = daily_aod.isel(latitude=lat_idx, longitude=lon_idx)
ax.plot(ts.valid_time, ts.values, 
        color=colors['Stinnett'], 
        linewidth=2, 
        marker='o',
        markersize=3,
        label=f"Stinnett ({exact_lat:.2f}°N, {exact_lon:.2f}°W)")


lat_idx = np.abs(daily_aod.latitude - locations['Canadian, TX']['lat']).argmin()
lon_idx = np.abs(daily_aod.longitude - locations['Canadian, TX']['lon']).argmin()
exact_lat = daily_aod.latitude.values[lat_idx]
exact_lon = daily_aod.longitude.values[lon_idx]
ts = daily_aod.isel(latitude=lat_idx, longitude=lon_idx)
ax.plot(ts.valid_time, ts.values, 
        color=colors['Canadian, TX'], 
        linewidth=2, 
        marker='o',
        markersize=3,
        label=f"Canadian, TX ({exact_lat:.2f}°N, {exact_lon:.2f}°W)")


ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
ax.grid(True, linestyle='--', alpha=0.7)
ax.legend(loc='best', fontsize=10)
ax.set_title('Aerosol Optical Depth at 550nm', fontsize=14, fontweight='bold')
ax.set_xlabel('Date', fontsize=12, fontweight='bold')
ax.set_ylabel('AOD', fontsize=12, fontweight='bold')

#  Temperature 
ax = ax2


lat_idx = np.abs(daily_t2m.latitude - locations['Stinnett']['lat']).argmin()
lon_idx = np.abs(daily_t2m.longitude - locations['Stinnett']['lon']).argmin()
exact_lat = daily_t2m.latitude.values[lat_idx]
exact_lon = daily_t2m.longitude.values[lon_idx]
ts = daily_t2m.isel(latitude=lat_idx, longitude=lon_idx)
ax.plot(ts.valid_time, ts.values, 
        color=colors['Stinnett'], 
        linewidth=2, 
        marker='o',
        markersize=3,
        label=f"Stinnett ({exact_lat:.2f}°N, {exact_lon:.2f}°W)")


lat_idx = np.abs(daily_t2m.latitude - locations['Canadian, TX']['lat']).argmin()
lon_idx = np.abs(daily_t2m.longitude - locations['Canadian, TX']['lon']).argmin()
exact_lat = daily_t2m.latitude.values[lat_idx]
exact_lon = daily_t2m.longitude.values[lon_idx]
ts = daily_t2m.isel(latitude=lat_idx, longitude=lon_idx)
ax.plot(ts.valid_time, ts.values, 
        color=colors['Canadian, TX'], 
        linewidth=2, 
        marker='o',
        markersize=3,
        label=f"Canadian, TX ({exact_lat:.2f}°N, {exact_lon:.2f}°W)")


ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
ax.grid(True, linestyle='--', alpha=0.7)
ax.legend(loc='best', fontsize=10)
ax.set_title('Temperature at 2m (°F)', fontsize=14, fontweight='bold')
ax.set_xlabel('Date', fontsize=12, fontweight='bold')
ax.set_ylabel('Temperature (°F)', fontsize=12, fontweight='bold')

# -CO Pressure 
ax = ax3

lat_idx = np.abs(daily_co_pressure.latitude - locations['Stinnett']['lat']).argmin()
lon_idx = np.abs(daily_co_pressure.longitude - locations['Stinnett']['lon']).argmin()
exact_lat = daily_co_pressure.latitude.values[lat_idx]
exact_lon = daily_co_pressure.longitude.values[lon_idx]
ts = daily_co_pressure.isel(latitude=lat_idx, longitude=lon_idx)
ax.plot(ts.valid_time, ts.values, 
        color=colors['Stinnett'], 
        linewidth=2, 
        marker='o',
        markersize=3,
        label=f"Stinnett ({exact_lat:.2f}°N, {exact_lon:.2f}°W)")

lat_idx = np.abs(daily_co_pressure.latitude - locations['Canadian, TX']['lat']).argmin()
lon_idx = np.abs(daily_co_pressure.longitude - locations['Canadian, TX']['lon']).argmin()
exact_lat = daily_co_pressure.latitude.values[lat_idx]
exact_lon = daily_co_pressure.longitude.values[lon_idx]
ts = daily_co_pressure.isel(latitude=lat_idx, longitude=lon_idx)
ax.plot(ts.valid_time, ts.values, 
        color=colors['Canadian, TX'], 
        linewidth=2, 
        marker='o',
        markersize=3,
        label=f"Canadian, TX ({exact_lat:.2f}°N, {exact_lon:.2f}°W)")

ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
ax.grid(True, linestyle='--', alpha=0.7)
ax.legend(loc='best', fontsize=10)
ax.set_title('Carbon Monoxide at 1000 hPa (ppb)', fontsize=14, fontweight='bold')
ax.set_xlabel('Date', fontsize=12, fontweight='bold')
ax.set_ylabel('CO Concentration (ppb)', fontsize=12, fontweight='bold')

ax = ax4

lat_idx = np.abs(daily_co_model.latitude - locations['Stinnett']['lat']).argmin()
lon_idx = np.abs(daily_co_model.longitude - locations['Stinnett']['lon']).argmin()
exact_lat = daily_co_model.latitude.values[lat_idx]
exact_lon = daily_co_model.longitude.values[lon_idx]
ts = daily_co_model.isel(latitude=lat_idx, longitude=lon_idx)
ax.plot(ts.valid_time, ts.values, 
        color=colors['Stinnett'], 
        linewidth=2, 
        marker='o',
        markersize=3,
        label=f"Stinnett ({exact_lat:.2f}°N, {exact_lon:.2f}°W)")

lat_idx = np.abs(daily_co_model.latitude - locations['Canadian, TX']['lat']).argmin()
lon_idx = np.abs(daily_co_model.longitude - locations['Canadian, TX']['lon']).argmin()
exact_lat = daily_co_model.latitude.values[lat_idx]
exact_lon = daily_co_model.longitude.values[lon_idx]
ts = daily_co_model.isel(latitude=lat_idx, longitude=lon_idx)
ax.plot(ts.valid_time, ts.values, 
        color=colors['Canadian, TX'], 
        linewidth=2, 
        marker='o',
        markersize=3,
        label=f"Canadian, TX ({exact_lat:.2f}°N, {exact_lon:.2f}°W)")

ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
ax.grid(True, linestyle='--', alpha=0.7)
ax.legend(loc='best', fontsize=10)
ax.set_title('Carbon Monoxide at Model Level 1 (ppb)', fontsize=14, fontweight='bold')
ax.set_xlabel('Date', fontsize=12, fontweight='bold')
ax.set_ylabel('CO Concentration (ppb)', fontsize=12, fontweight='bold')

plt.tight_layout()

plt.suptitle('Daily Time Series of Meteorological Variables at Two Locations', fontsize=16, y=0.98)

plt.show()







plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['figure.titleweight'] = 'bold'

aod_path = r"C:\Users\hp\OneDrive\Desktop\AOD500nm.nc"
ds_aod = xr.open_dataset(aod_path)

locations = {
    'Stinnett': {'lat': 35.823, 'lon': -101.444},
    'Canadian, TX': {'lat': 35.913, 'lon': -100.383}
}

lat_idx_stinnett = np.abs(ds_aod.latitude - locations['Stinnett']['lat']).argmin()
lon_idx_stinnett = np.abs(ds_aod.longitude - locations['Stinnett']['lon']).argmin()

exact_lat_stinnett = ds_aod.latitude.values[lat_idx_stinnett]
exact_lon_stinnett = ds_aod.longitude.values[lon_idx_stinnett]

ts_stinnett = ds_aod.aod550.isel(latitude=lat_idx_stinnett, longitude=lon_idx_stinnett)

df_stinnett = ts_stinnett.to_dataframe().reset_index()

df_stinnett['hour'] = df_stinnett['valid_time'].dt.hour
df_stinnett['date'] = df_stinnett['valid_time'].dt.date

pivot_df_stinnett = df_stinnett.pivot(index='date', columns='hour', values='aod550')

lat_idx_canadian = np.abs(ds_aod.latitude - locations['Canadian, TX']['lat']).argmin()
lon_idx_canadian = np.abs(ds_aod.longitude - locations['Canadian, TX']['lon']).argmin()

exact_lat_canadian = ds_aod.latitude.values[lat_idx_canadian]
exact_lon_canadian = ds_aod.longitude.values[lon_idx_canadian]

ts_canadian = ds_aod.aod550.isel(latitude=lat_idx_canadian, longitude=lon_idx_canadian)

df_canadian = ts_canadian.to_dataframe().reset_index()

df_canadian['hour'] = df_canadian['valid_time'].dt.hour
df_canadian['date'] = df_canadian['valid_time'].dt.date

pivot_df_canadian = df_canadian.pivot(index='date', columns='hour', values='aod550')

plt.figure(figsize=(14, 8))
sns.heatmap(pivot_df_stinnett, 
            cmap='YlOrRd',
            cbar_kws={'label': 'AOD'},
            robust=True)

plt.title(f'Hourly AOD at Stinnett ({exact_lat_stinnett:.2f}°N, {exact_lon_stinnett:.2f}°W)', 
          fontsize=16, fontweight='bold')
plt.xlabel('Hour of Day', fontsize=14, fontweight='bold')
plt.ylabel('Date', fontsize=14, fontweight='bold')

hours = pivot_df_stinnett.columns.tolist()
plt.xticks(np.arange(len(hours)) + 0.5, hours, rotation=0)

dates = pivot_df_stinnett.index.strftime('%m-%d').tolist()
plt.yticks(np.arange(0, len(dates), 2) + 0.5, dates[::2], rotation=0)

plt.tight_layout()
plt.show()



plt.figure(figsize=(14, 8))
sns.heatmap(pivot_df_canadian, 
            cmap='YlOrRd',
            cbar_kws={'label': 'AOD'},
            robust=True)

plt.title(f'Hourly AOD at Canadian, TX ({exact_lat_canadian:.2f}°N, {exact_lon_canadian:.2f}°W)', 
          fontsize=16, fontweight='bold')
plt.xlabel('Hour of Day', fontsize=14, fontweight='bold')
plt.ylabel('Date', fontsize=14, fontweight='bold')

hours = pivot_df_canadian.columns.tolist()
plt.xticks(np.arange(len(hours)) + 0.5, hours, rotation=0)

dates = pivot_df_canadian.index.strftime('%m-%d').tolist()
plt.yticks(np.arange(0, len(dates), 2) + 0.5, dates[::2], rotation=0)

plt.tight_layout()
plt.show()





plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['figure.titleweight'] = 'bold'


locations = {
    'Stinnett': {'lat': 35.823, 'lon': -101.444},
    'Canadian, TX': {'lat': 35.913, 'lon': -100.383}
}

def create_wind_rose(direction, speed, title):
    sectors = 16
    sector_angle = 360 / sectors
    direction_bins = np.linspace(0, 360, sectors + 1)
    
    speed_bins = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    
    hist, dir_edges, speed_edges = np.histogram2d(
        direction, speed, bins=[direction_bins, speed_bins]
    )
    
    hist_norm = hist / hist.sum(axis=1)[:, np.newaxis]
    
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': 'polar'})
    
    theta = np.deg2rad(dir_edges[:-1] + sector_angle / 2)
    
    width = np.deg2rad(sector_angle)
    
    colors = plt.cm.viridis(np.linspace conseguenze: 0, 1, len(speed_bins) - 1))
    
    bottom = np.zeros(sectors)
    for i in range(len(speed_bins) - 1):
        ax.bar(theta, hist_norm[:, i], width=width, bottom=bottom, 
               color=colors[i], edgecolor='white', alpha=0.8)
        bottom += hist_norm[:, i]
    
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(90)
    
    legend_labels = [f'{speed_bins[i]}-{speed_bins[i+1]} m/s' for i in range(len(speed_bins) - 1)]
    ax.legend(legend_labels, loc='center left', bbox_to_anchor=(1.1, 0.5), title='Wind Speed')
    
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    
    return fig, ax

lat_idx_stinnett = np.abs(ds_u10.latitude - locations['Stinnett']['lat']).argmin()
lon_idx_stinnett = np.abs(ds_u10.longitude - locations['Stinnett']['lon']).argmin()

exact_lat_stinnett = ds_u10.latitude.values[lat_idx_stinnett]
exact_lon_stinnett = ds_u10.longitude.values[lon_idx_stinnett]

u_ts_stinnett = ds_u10.u10.isel(latitude=lat_idx_stinnett, longitude=lon_idx_stinnett)
v_ts_stinnett = ds_v10.v10.isel(latitude=lat_idx_stinnett, longitude=lon_idx_stinnett)

wind_direction_stinnett = (270 - np.degrees(np.arctan2(v_ts_stinnett, u_ts_stinnett))) % 360

wind_speed_stinnett = np.sqrt(u_ts_stinnett**2 + v_ts_stinnett**2)

lat_idx_canadian = np.abs(ds_u10.latitude - locations['Canadian, TX']['lat']).argmin()
lon_idx_canadian = np.abs(ds_u10.longitude - locations['Canadian, TX']['lon']).argmin()

exact_lat_canadian = ds_u10.latitude.values[lat_idx_canadian]
exact_lon_canadian = ds_u10.longitude.values[lon_idx_canadian]

u_ts_canadian = ds_u10.u10.isel(latitude=lat_idx_canadian, longitude=lon_idx_canadian)
v_ts_canadian = ds_v10.v10.isel(latitude=lat_idx_canadian, longitude=lon_idx_canadian)

wind_direction_canadian = (270 - np.degrees(np.arctan2(v_ts_canadian, u_ts_canadian))) % 360

wind_speed_canadian = np.sqrt(u_ts_canadian**2 + v_ts_canadian**2)

title_stinnett = f'Wind Rose at Stinnett ({exact_lat_stinnett:.2f}°N, {exact_lon_stinnett:.2f}°W)'
fig1, ax1 = create_wind_rose(wind_direction_stinnett.values, wind_speed_stinnett.values, title_stinnett)
plt.tight_layout()
plt.show()

title_canadian = f'Wind Rose at Canadian, TX ({exact_lat_canadian:.2f}°N, {exact_lon_canadian:.2f}°W)'
fig2, ax2 = create_wind_rose(wind_direction_canadian.values, wind_speed_canadian.values, title_canadian)
plt.tight_layout()
plt.show()


ds_aod = xr.open_dataset(aod_path)
ds_co_model = xr.open_dataset(co_model_path)

ds_co_model = ds_co_model.assign_coords(
    longitude=((ds_co_model.longitude + 180) % 360 - 180)
)

co_model_ppb = ds_co_model.co.squeeze() * 1e9  # kg/kg to ppb

aod_daily = ds_aod.aod550.resample(valid_time='D').mean()
co_daily = co_model_ppb.resample(valid_time='D').mean()

print("Dimensions AOD:", aod_daily.dims)
print("Dimensions CO:", co_daily.dims)
print("Coordonnées temporelles compatibles:", np.array_equal(aod_daily.valid_time.values, co_daily.valid_time.values))

correlation_map = np.full(aod_daily.shape[1:], np.nan)  
p_value_map = np.full(aod_daily.shape[1:], np.nan)      

for i in range(aod_daily.shape[1]):  
    for j in range(aod_daily.shape[2]):   
      
        aod_ts = aod_daily.isel(latitude=i, longitude=j).values
        co_ts = co_daily.isel(latitude=i, longitude=j).values
        
        mask = ~(np.isnan(aod_ts) | np.isnan(co_ts))
        aod_clean = aod_ts[mask]
        co_clean = co_ts[mask]
        
        if len(aod_clean) > 10:  
            corr, p_value = pearsonr(aod_clean, co_clean)
            correlation_map[i, j] = corr
            p_value_map[i, j] = p_value

fig = plt.figure(figsize=(14, 10))
ax = plt.axes(projection=ccrs.PlateCarree())

im = ax.pcolormesh(aod_daily.longitude, aod_daily.latitude, correlation_map,
                   cmap='RdBu_r', vmin=-1, vmax=1, transform=ccrs.PlateCarree())

ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=2, edgecolor='white')
ax.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=2, edgecolor='white')
ax.add_feature(cfeature.STATES.with_scale('50m'), linewidth=1, edgecolor='white', alpha=0.5)

cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, shrink=0.8)
cbar.set_label('Coefficient de corrélation de Pearson', fontsize=12, fontweight='bold')

plt.title('Corrélation entre AOD et CO (model level) par pixel', fontsize=16, fontweight='bold', pad=20)

plt.tight_layout()

