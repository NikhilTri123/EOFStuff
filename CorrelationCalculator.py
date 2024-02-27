import xarray as xr
import numpy as np
from scipy.stats import pearsonr
import AceCalculator
import PacificAceCalculator
import time as t
from scipy.signal import detrend

variable = "hgtmid"
aceMonths = [1]  # ACE range to correlate for
varMonths = [12, 1, 2]   # variable months to correlate for
corrYears = range(1970, 2023)
basin = "NA"

# variable path and dictionary for conversions
varPath = f'C:/Nikhil Stuff/Coding Stuff/variablefiles/{variable}EraModified.nc'
varDict = {"sst": "sst", "mslp": "msl", "hgtmid": "z", "hgtup": "z", "uwndlow": "u", "uwndup": "u", "shummid": "q"}


def createDataset(path):
    """
    Creates a xarray DataArray for the given variable
    :param path: the file path for the netcdf variable file
    :return: the xarray DataArray for the variable
    """
    dataset = xr.open_dataset(path)
    data = dataset[varDict[variable]]
    return data


def detrendData(data):
    """
    Detrends variable data to remove the skewing that gobal warming causes
    :param data: the variable DataArray to be detrended
    :return: the detrended variable DataArray
    """
    data = data.fillna(0)
    reshaped_data = data.stack(combined=('latitude', 'longitude'))
    detrended_data = detrend(reshaped_data, axis=0)

    detrended_map_data = detrended_data.reshape(data.shape)
    detrended_map_data = xr.DataArray(
        data=detrended_map_data,
        dims=('time', 'latitude', 'longitude'),
        coords=dict(
            time=(["time"], data.time.values),
            latitude=(["latitude"], data.latitude.values),
            longitude=(["longitude"], data.longitude.values)
        ))
    return detrended_map_data


def climoSet(data, period, month):
    """
    gets the variable data for the given month in the given period
    :param data: the variable data to be used
    :param period: the period of time that data is retrieved for
    :param month: the month that data is retrieved for
    :return: a list with each element being a year of data in the period
    """
    month = format(month, '02d')
    years = []
    for year in period:
        date = np.datetime64(f'{year}-{month}', 'D')
        years.append(data.sel(time=date))
    return years


def zscoreThing(data, years, month, calcYear):
    """
    Calculates a z-score map for the given month and year
    :param data: the variable data to be used
    :param years: list of all year maps to be used for z-scores
    :param month: the month that data is calculated for
    :param calcYear: the year that data is calculated for
    :return: a z-score map for the given month and year
    """
    allMeans = np.mean(np.array(years), axis=0)
    allStds = np.std(np.array(years), axis=0)

    month = format(month, '02d')
    date = np.datetime64(str(calcYear) + '-' + str(month), 'D')
    calcYearData = data.sel(time=date)

    zscoreMap = (calcYearData - allMeans) / allStds

    return zscoreMap


def getCorrelation(varDataset, aceVals, varMonth):
    """
    Calculates a global correlation map between a given variable and list of ACE values for a given month. E.g. if the
    variable is SST data, the aceVals are for September only, and the month is June, a correlation map between June
    SST's and September ACE will be calculated
    :param varDataset: xarray dataset for a variable
    :param aceVals: List of ACE values for a given month and time period
    :param varMonth: month that the correlation map is calculated for
    :return: a global correlation map
    """
    # each element in switchedData is a list data from each year for a given pixel
    allYears = climoSet(varDataset, corrYears, varMonth)
    allFlattenedData = np.array([zscoreThing(varDataset, allYears, varMonth, year).values.flatten() for year in
                                 corrYears])
    switchedData = np.nan_to_num(allFlattenedData.T)

    # each element is a correlation for a given pixel
    corrList = np.array([pearsonr(pixelData, aceVals) for pixelData in switchedData])
    corrList = np.where(corrList[:, 1] <= 0.05, corrList[:, 0], 0)

    # list of correlations is reshaped to map of correlations
    corrList = np.reshape(corrList, varDataset.shape[1:])
    corrList = np.nan_to_num(corrList)

    return corrList


def main(aceMonth):
    print(f"Generating correlation map for month {aceMonth}")

    # get 1970-present ACE for given month
    if basin == 'NA':
        yearsAce = np.array(AceCalculator.getMonthAce(aceMonth))
    else:
        yearsAce = np.array(PacificAceCalculator.getMonthAce(aceMonth, basin))
    aceData = yearsAce[:, 1]

    pmmData = [-2.21, -0.3, -0.02, -3.12, -0.45, -3.46, 0.32, -1.7, -0.7, 0.8, 3.91, 1.25, 1.06, -0.85, -0.28, -0.11, -1.55, -0.96, 4.26, 4.3, 3.62, -1.73, 0.8, -3.56, -4.95, -1.19, -3.21, -4.45, -2.09, 1.59, 0.11, 2.64, -0.53, 1.54, -6.05, -0.8, 0.14, 4.01, 1.84, -0.24, 1.45, 1.93, 3.72, -0.35, 5.01, 3.04, -0.83, 3.06, 3.2, -6.98, -3.69, -1.52, -1.2, 1.99, -2.01, 0.9, 1.12, 0.97, -1.09, 1.26, -2.62, 1.33, -0.82, -2.32, -0.68, 0.74, 2.53, 0.31, 1.67, 5.6, 2.55, 5.53, 4.1, 0.8, 2.44]
    pmmData = np.array(pmmData[-len(aceData):])

    # create dataset for given variable
    varDataset = createDataset(varPath)
    varDataset = detrendData(varDataset)

    # correlate variable data to ACE for each month of variable data
    corrMonths = np.array([getCorrelation(varDataset, aceData, varMonth) for varMonth in varMonths])
    return corrMonths


if __name__ == '__main__':
    # calculate all correlations in parallel and store them in correlations
    start = t.time()
    correlations = main(aceMonths)
    print(f"Finished in {round(t.time() - start)} seconds")

    # create and save correlation dataset
    latLonValues = createDataset(varPath)
    first = format(varMonths[0], '02d')
    last = format(varMonths[-1] + 1, '02d')
    time = np.arange(f'1969-01-01', f'1969-04-01', dtype='datetime64[M]')
    ds = xr.Dataset(
        data_vars=dict(
            decPmm=(["time", "latitude", "longitude"], correlations)
        ),
        coords=dict(
            time=(["time"], time),
            latitude=(["latitude"], latLonValues.latitude.values),
            longitude=(["longitude"], latLonValues.longitude.values)
        )
    )
    print(ds)
    ds.to_netcdf(f'C:/Nikhil Stuff/Coding Stuff/variablefiles/{basin}corrs/{variable}EraCorrWINTER.nc')
