import xarray as xr
import numpy as np
from scipy.stats import pearsonr

variable = "sst"
aceMonths = range(1, 13)  # ACE range to correlate for
varMonths = range(1, 12)  # variable months to correlate for
corrYears = range(1970, 2024)
basin = "NA"

# variable path and dictionary for conversions
varPath = f'C:/Nikhil Stuff/Coding Stuff/variablefiles/{variable}EraModified.nc'
varDict = {"sst": "sst", "mslp": "msl", "hgtmid": "z", "hgtup": "z", "uwndlow": "u", "uwndup": "u", "shummid": "q", "stab": "ss", "vp": "vp"}


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


def getCorrelation(varData, aceVals, varMonth):
    """
    Calculates a global correlation map between a given variable and list of ACE values for a given month. E.g. if the
    variable is SST data, the aceVals are for September only, and the month is June, a correlation map between June
    SST's and September ACE will be calculated
    :param varData: xarray dataset for a variable
    :param aceVals: List of ACE values for a given month and time period
    :param varMonth: month that the correlation map is calculated for
    :return: a global correlation map
    """
    # each element in switchedData is a list data from each year for a given pixel
    varData = varData.sel(time=(varData['time.month'] == varMonth))
    allFlattenedData = np.reshape(varData.values.flatten(), (len(corrYears), -1))
    switchedData = np.nan_to_num(allFlattenedData.T)

    # each element is a correlation for a given pixel
    corrList = np.array([pearsonr(pixelData, aceVals) for pixelData in switchedData])
    corrList = np.where(corrList[:, 1] <= 0.05, corrList[:, 0], 0)

    # list of correlations is reshaped to map of correlations
    corrList = np.reshape(corrList, varData.shape[1:])
    corrList = np.nan_to_num(corrList)

    return corrList


def main(time_series_data, month):
    print(f"Generating correlation map")

    # create dataset for given variable
    varDataset = xr.open_dataset(varPath)
    varData = varDataset[varDict[variable]]
    varData = zscoreThing(detrendData(varData))
    varData = varData.sel(time=(varData['time.year'] >= corrYears[0]) & (varData['time.year'] <= corrYears[-1]))

    # correlate variable data to ACE for each month of variable data
    corrMonths = getCorrelation(varData, time_series_data, month)

    # convert back to DataArray
    corrMonths = xr.DataArray(
        data=corrMonths,
        dims=["latitude", "longitude"],
        coords=dict(
            latitude=(["latitude"], varData.latitude.values),
            longitude=(["longitude"], varData.longitude.values)
        )
    )

    return corrMonths
