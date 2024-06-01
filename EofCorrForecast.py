import xarray as xr
import numpy as np
from sklearn.decomposition import PCA
import cartopy.crs as ccrs
import cartopy.feature as cf
import matplotlib.pyplot as plt
import CorrelationCalculator

forecast_months = range(5, 6)
forecast_years = range(1970, 2024)
eof_month = 9

monthsDict = {1: "January", 2: "February", 3: "March", 4: "April", 5: "May", 6: "June", 7: "July", 8: "August",
              9: "September", 10: "October", 11: "November", 12: "December"}


def detrendData(data):
    """
    Detrends variable data to remove the skewing that gobal warming causes
    :param data: the variable DataArray to be detrended
    :return: the detrended variable DataArray
    """
    # calculate line of best fit for each month
    data = data.fillna(0)
    month_groups = data.groupby('time.month')
    regress_lines = month_groups.apply(lambda x: x.polyfit(dim='time', deg=1))
    regress_lines = regress_lines.polyfit_coefficients

    # solve appropriate line of best fit for each time to get least squares regression
    month_lsrs = []
    for mon in set(data['time.month'].to_numpy()):
        mask = data['time.month'] == mon
        month_lsr = xr.polyval(data.sel(time=mask).time, regress_lines.sel(month=mon))
        month_lsrs.append(month_lsr.drop_vars('month'))

    # subtract least squares regressions from data and return
    lsr = xr.concat(month_lsrs, dim='time')
    detrended_data = data - lsr
    return detrended_data


def zscoreThing(data):
    """
    Normalizes variable data to account for different amounts of variability in different regions
    :param data: the variable DataArray to be normalized
    :return: the normalized variable DataArray
    """
    # calculate mean/standard deviation maps for each month
    month_groups = data.groupby('time.month')
    allMeans = month_groups.mean(dim='time')
    allStds = month_groups.std(dim='time')

    # use appropriate mean/std maps to calculate z-score map for each time
    month_zscores = []
    for mon in set(data['time.month'].to_numpy()):
        mask = data['time.month'] == mon
        zscoreMap = (data.sel(time=mask) - allMeans.sel(month=mon)) / allStds.sel(month=mon)
        month_zscores.append(zscoreMap.drop_vars('month'))

    # concat z-score data for each month and return
    zscore_data = xr.concat(month_zscores, dim='time')
    return zscore_data


# open variable data
varPath = f'C:/Nikhil Stuff/Coding Stuff/variablefiles/sstEraModified.nc'
varDataset = xr.open_dataset(varPath)
varData = varDataset['sst']
datesMask = np.isin(varData['time.month'], forecast_months) & np.isin(varData['time.year'], forecast_years)
varData = varData.sel(time=datesMask)

zscoreData = zscoreThing(detrendData(varData))
zscoreData = zscoreData.sel(latitude=slice(70, -30), longitude=slice(150, 360))
zscoreArray = np.nan_to_num(zscoreData)
print(f"Initial shape: {zscoreArray.shape}")

# Flatten the 3D array to a 2D matrix (time, space)
time_steps, lat_size, lon_size = zscoreArray.shape
sst_reshaped = zscoreArray.reshape(time_steps, lat_size * lon_size)
print(f"Flattened shape: {sst_reshaped.shape}")

# Perform PCA to get the most prevalent patterns (EOFs)
n_components = 4  # You can adjust this value based on your requirement
pca = PCA(n_components=n_components)
PCs = pca.fit_transform(sst_reshaped)
print(f"PC matrix shape: {PCs.shape}")

explained_variance = pca.explained_variance_ratio_
print(f"Explained variance: {explained_variance}")

for i in range(1, 2):
    reshaped_array = PCs.T[i].reshape(len(forecast_months), -1)
    contributions = np.mean(reshaped_array, axis=0)
    contributions = np.array([forecast_years, contributions]).T
    contributions = contributions[contributions[:, 1].argsort()]
    contributions[:, 1] *= -1

    sorted_values = contributions[contributions[:, 0].argsort()]
    corr_map = CorrelationCalculator.main(sorted_values[:, 1], eof_month)
    corr_map = corr_map.sel(latitude=slice(70, -30), longitude=slice(150, 360))

    # plot cartopy map and various features
    plt.figure(figsize=(12, 6))
    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
    ax.add_feature(cf.LAND)
    ax.add_feature(cf.STATES, linewidth=0.2, edgecolor="gray")
    ax.add_feature(cf.BORDERS, linewidth=0.3)
    ax.coastlines(linewidth=0.5, resolution='50m')

    # plot gridlines
    gl = ax.gridlines(crs=ccrs.PlateCarree(central_longitude=0), draw_labels=True, linewidth=1, color='gray',
                      alpha=0.5, linestyle='--')
    gl.top_labels = gl.right_labels = False
    gl.xlabel_style = {'size': 6, 'weight': 'bold', 'color': 'gray'}
    gl.ylabel_style = {'size': 6, 'weight': 'bold', 'color': 'gray'}

    # add data and colormap
    plt.contourf(corr_map.longitude, corr_map.latitude, corr_map, np.arange(-0.8, 0.8, 0.02), extend='both',
                 transform=ccrs.PlateCarree(), cmap='RdBu_r')
    cbar = plt.colorbar(pad=0.015, aspect=27, shrink=0.83)
    cbar.set_ticks(np.arange(-0.8, 1, 0.2))
    cbar.ax.tick_params(labelsize=7)

    # add titling
    mainTitle = f"ERA5 {monthsDict[eof_month]} Detrended SST Correlated to " \
                f"{monthsDict[forecast_months[0]]} EOF{i + 1} in Box"
    plt.title(mainTitle, fontsize=9, weight='bold', loc='left')
    plt.title("DCAreaWx", fontsize=9, weight='bold', loc='right', color='gray')
    ax.text(154, -27, 'Only significant correlations are plotted', fontsize=9, weight='bold',
            transform=ccrs.PlateCarree())

    # save and display map
    plt.savefig(r"C:/Nikhil Stuff/Coding Stuff/EofCorrMap.png", dpi=300, bbox_inches='tight')
    plt.show()
