import xarray as xr
import numpy as np
from scipy.signal import detrend
from sklearn.decomposition import PCA
import AceCalculator
from scipy.stats import norm, pearsonr

forecast_month = 7


def detrend_data(data):
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


def get_zscores(data, month):
    month = format(month, '02d')
    years = []
    for year in range(1970, 2024):
        date = np.datetime64(f'{year}-{month}', 'D')
        years.append(data.sel(time=date))

    all_means = np.mean(np.array(years), axis=0)
    all_stds = np.std(np.array(years), axis=0)

    all_zscores = []
    for calcYear in range(1970, 2024):
        date = np.datetime64(f'{calcYear}-{month}', 'D')
        calc_year_data = data.sel(time=date)
        zscore_map = (calc_year_data - all_means) / all_stds
        all_zscores.append(zscore_map)
    all_zscores = np.array(all_zscores)

    all_zscores = xr.DataArray(
        data=all_zscores,
        dims=('time', 'latitude', 'longitude'),
        coords=dict(
            time=(["time"], data.time.values),
            latitude=(["latitude"], data.latitude.values),
            longitude=(["longitude"], data.longitude.values)
        ))
    return all_zscores


# open variable data
varPath = f'C:/Nikhil Stuff/Coding Stuff/variablefiles/sstEraModified.nc'
varDataset = xr.open_dataset(varPath)
varData = varDataset['sst']
detrendedData = detrend_data(varData)

# get anomaly data for given variable
months = slice(forecast_month - 1, None, 12)
monthData = detrendedData.isel(time=months)
zscoreData = get_zscores(monthData, forecast_month)
zscoreData = zscoreData.sel(latitude=slice(60, -60))
zscores = np.nan_to_num(zscoreData.to_numpy())
print(f"Initial shape: {zscores.shape}")

# Step 1: Flatten the 3D array to a 2D matrix (time, space)
time_steps, lat_size, lon_size = zscores.shape
sst_reshaped = zscores.reshape(time_steps, lat_size * lon_size)
print(f"Flattened shape: {sst_reshaped.shape}")

# Step 2: Perform PCA to get the most prevalent patterns (EOFs)
n_components = 10  # You can adjust this value based on your requirement
pca = PCA(n_components=n_components)
PCs = pca.fit_transform(sst_reshaped)
print(f"PC matrix shape: {PCs.shape}")

# Step 3: Reshape the PCA results (EOFs) back to 3D (x, latitude, longitude)
EOFs = np.zeros((n_components, lat_size, lon_size))
for i in range(n_components):
    EOFs[i, :, :] = pca.inverse_transform(np.eye(n_components)[i]).reshape(lat_size, lon_size)
print(f"EOF matrix shape: {EOFs.shape}")

explained_variance = pca.explained_variance_ratio_
print(f"Explained variance: {explained_variance}")

curr_zscores = get_zscores(monthData, forecast_month)
scores = []
for pred_year in range(1970, 2024):
    form_month = format(forecast_month, '02d')
    year_zscores = curr_zscores.sel(time=f'{pred_year}-{form_month}-01', latitude=slice(60, -60))

    corrs = []
    for i in range(0, 10):
        PC = PCs.T[i]
        EOF = EOFs[i]
        # if pred_year != 2023:
        #     PC = np.delete(PC, pred_year - 1970)

        ace = np.array(AceCalculator.getMonthAce(range(1, 13)))[:, 1]
        # if pred_year != 2023:
        #     ace = np.delete(ace, pred_year - 1970)
        corr = np.round(pearsonr(PC[:-1], ace)[0], 2)

        year_zscores = np.nan_to_num(year_zscores).flatten()
        EOF = EOF.flatten()
        corrToCurr = np.round(pearsonr(EOF, year_zscores)[0], 2)

        corrs.append([corr, corrToCurr])

    corrs = np.array(corrs)
    score = np.sum(np.prod(corrs, axis=1))
    print([pred_year, score])
    scores.append([pred_year, score])

scores.append([-9999, -0.4])
scores.append([9999, 0.4])
scores = np.array(scores)
scoresAsc = scores[scores[:, 1].argsort()]
print(scores.tolist())

# get the actual ACE data for the given month
actualAce = np.array(AceCalculator.getMonthAce(range(1, 13)))
actualAceAsc = actualAce[actualAce[:, 1].argsort()]

# calculate the worst and best possible ACE values and use them to get the range of possible ACE
worstScore = 0
bestScore = scoresAsc[-1][1] / scoresAsc[-2][1] * actualAceAsc[-1][1]
aceRange = np.round(np.arange(worstScore, bestScore, 0.01), 2)

# create the cdf equations which ACE predictions are based off of
actualCdf = norm.cdf(aceRange, np.mean(actualAceAsc[:, 1]), np.std(actualAceAsc[:, 1]))
predictedCdf = norm.cdf(scoresAsc[:, 1], np.mean(scoresAsc[:, 1]), np.std(scoresAsc[:, 1]))

# calculate ACE predictions and sort them by year
predAce = []
for a in range(len(predictedCdf)):
    num = 0
    while num < len(actualCdf) - 1 and predictedCdf[a] > actualCdf[num]:
        num += 1
    predAce.append([scoresAsc[a][0], aceRange[num]])
predAce = np.array(predAce)
predAce = predAce[predAce[:, 0].argsort()]
print(predAce.tolist())

actual = np.array(AceCalculator.getMonthAce(range(1, 13)))[:, 1]
predicted = predAce[:, 1][1:-2]
MSE = np.square(np.subtract(actual, predicted)).mean()
RMSE = np.round(np.math.sqrt(MSE), 1)
print("\nRMSE: " + str(RMSE) + " ACE")
corr = np.round(pearsonr(actual, predicted)[0], 2)
print("Correlation coefficent: " + str(corr))
