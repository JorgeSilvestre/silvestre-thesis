import json
import concurrent

import numpy as np
import pandas as pd
import pyarrow.parquet
from tqdm import tqdm

import utils
from paths import *
import params
from params import (AIRPORT_AREA, 
                    THRESHOLD_DISTANCE_TO_AIRPORT, 
                    THRESHOLD_GAP_TIME, 
                    THRESHOLD_CONTINUITY)

# np.seterr(invalid='raise')

airports = pd.read_parquet(AIRPORTS_PATH)

### PARAMETERS

### L0/L1 -----------------------------------------------------------------------------------------

def calculate_metrics_fplan(date: str, data: pd.DataFrame, state: str = 'clean') -> None:
    # data = pd.read_parquet(NM_PARQUET_FPLAN_PATH / f'nm.fplan.{date}.parquet')

    results = {}
    results['date'] = date
    results['state'] = state
    results['level'] = 'L0' if state=='raw' else 'L1'

    results['num_messages'] = len(data)
    results['num_flights'] = data.ifplId.nunique()
    completitude = data.notnull().sum()
    results['completitude'] = {col:val/len(data) for col, val in completitude.items()}
    uniqueness = data.drop(['timestamp','estimatedOffBlockTime','totalEstimatedElapsedTime'], axis=1).nunique()
    dups_columns = data.columns.difference(['uuid'])
    results['duplicate_records'] = data.shape[0]-data.drop_duplicates(subset=dups_columns).shape[0]
    results['uniqueness'] = {col:val for col, val in uniqueness.items()}
    results['ranges'] = {
        'timestamp_min':data.timestamp.min(),
        'timestamp_max':data.timestamp.max(),
        'offblockTime_min':data.estimatedOffBlockTime.min(),
        'offblockTime_max':data.estimatedOffBlockTime.max(),
    }

    results['avg_messages_per_flight'] = data.groupby('ifplId').count().timestamp.mean()

    results['flights_airport_dep'] = {col:val for col,val
        in data.drop_duplicates(subset=['ifplId']).groupby('aerodromeOfDeparture').count().ifplId.items()}
    results['flights_airport_dest'] = {col:val for col,val
        in data.drop_duplicates(subset=['ifplId']).groupby('aerodromeOfDestination').count().ifplId.items()}
    results['flights_airport_route'] = {'-'.join(col):val for col,val
        in data.drop_duplicates(subset=['ifplId']).groupby(['aerodromeOfDeparture', 'aerodromeOfDestination']).count().ifplId.items()}

    if state == 'clean':
        filepath = NM_FPLAN_METRICS_L1_PATH / f'fPlan.L1.{date}.json'
    elif state == 'raw':
        filepath = NM_FPLAN_METRICS_L0_PATH / f'fPlan.L0.{date}.json'
    if not filepath.parent.exists():
        filepath.parent.mkdir(parents=True)
    with open(filepath, 'w+', encoding='utf8') as file:
        json.dump(results, file, indent=2, default=utils.custom_json_encoder)

def calculate_metrics_fdata(date: str, data: pd.DataFrame, state: str = 'clean') -> None:
    # data = pd.read_parquet(NM_PARQUET_FDATA_PATH / f'nm.fData.{date}.parquet')

    results = {}
    results['date'] = date
    results['state'] = state
    results['level'] = 'L0' if state=='raw' else 'L1'
    results['num_messages'] = len(data)
    results['num_flights'] = data.ifplId.nunique()
    completitude = data.notnull().sum()
    results['completitude'] = {col:val/len(data) for col, val in completitude.items()}
    uniqueness = data.drop(['actualTakeOffTime','actualTimeOfArrival','estimatedOffBlockTime',
                            'estimatedTakeOffTime','estimatedTimeOfArrival','flightDataVersionNr',
                            'routeLength'], axis=1).nunique()
    dups_columns = data.columns.difference(['uuid'])
    results['duplicate_records'] = data.shape[0]-data.drop_duplicates(subset=dups_columns).shape[0]
    results['uniqueness'] = {col:val for col, val in uniqueness.items()}

    results['ranges'] = {
        'offblockTime_min':data.estimatedOffBlockTime.min(),
        'offblockTime_max':data.estimatedOffBlockTime.max(),
        'actualTakeOffTime_min':data.actualTakeOffTime.min(numeric_only=True),
        'actualTakeOffTime_max':data.actualTakeOffTime.max(),
        'actualTimeOfArrival_min':data.actualTimeOfArrival.min(),
        'actualTimeOfArrival_max':data.actualTimeOfArrival.max(),
        'estimatedTakeOffTime_min':data.estimatedTakeOffTime.min(),
        'estimatedTakeOffTime_max':data.estimatedTakeOffTime.max(),
        'estimatedTimeOfArrival_min':data.estimatedTimeOfArrival.min(),
        'estimatedTimeOfArrival_max':data.estimatedTimeOfArrival.max(),
    }

    results['avg_messages_per_flight'] = data.groupby('ifplId').count().estimatedOffBlockTime.mean()
    
    if state == 'clean':
        filepath = NM_FDATA_METRICS_L1_PATH / f'fData.L1.{date}.json'
    elif state == 'raw':
        filepath = NM_FDATA_METRICS_L0_PATH / f'fData.L0.{date}.json'
    if not filepath.parent.exists():
        filepath.parent.mkdir(parents=True)
    with open(filepath, 'w+', encoding='utf8') as file:
        json.dump(results, file, indent=2, default=utils.custom_json_encoder)

def calculate_metrics_openskyVectors(date: str, state: str = 'clean') -> None:
    if state == 'raw':
        data_path = OPENSKY_RAW_VECTORS_PATH
        output_path = OPENSKY_VECTORS_METRICS_L0_PATH / f'vectors.L0.{date}.json'
    elif state == 'clean':
        data_path = OPENSKY_PARQUET_VECTORS_PATH
        output_path = OPENSKY_VECTORS_METRICS_L1_PATH / f'vectors.L1.{date}.json'
    
    file_list = list((data_path / f'flightDate={date}').glob('*.parquet'))
    if not file_list:
        return None

    res = []
    for path in tqdm(file_list, desc=f'{date} VECTORS | Metrics', ncols=125):
        data = pd.read_parquet(path, engine='pyarrow', dtype_backend='pyarrow')
        if state == 'raw':
            data = data.drop(['sensors', 'spi', 'position_source'], axis=1)
            data = data.rename(columns=params.mapping_opensky)
        completitude_fields = data.columns

        partial_results = {}
        partial_results['num_vectors'] = len(data)
        completitude = data.notnull().sum()
        partial_results['completitude'] = {col:val/len(data) for col, val in completitude.items()}
        partial_results['duplicate_records'] = data.shape[0] - data.drop_duplicates().shape[0]
        partial_results['reused_position'] = data.shape[0] - data.drop_duplicates(subset=['icao24','time_position','latitude','longitude']).shape[0]
        partial_results['nulls'] = {
            'latitude': int(data.latitude.isna().sum()),
            'longitude': int(data.longitude.isna().sum()),
            'latlon': len(data) - len(data[['latitude','longitude']].dropna(how='all'))
        }
        res.append(partial_results)

    results = {}
    results['state'] = state
    results['level'] = 'L0' if state=='raw' else 'L1'
    results['date'] = date
    results['num_vectors'] = sum([r['num_vectors'] for r in res])
    results['reused_position'] = sum([r['reused_position'] for r in res])
    results['duplicate_records'] = sum([r['duplicate_records'] for r in res])
    results['completitude'] = {}
    for attr in completitude_fields:
        results['completitude'][attr] = sum([r['completitude'][attr]*r['num_vectors'] for r in res])/results['num_vectors']
    results['nulls'] = {}
    for attr in ['latitude','longitude','latlon']:
        results['nulls'][attr] = sum([x['nulls'][attr] for x in res])
    
    if state == 'raw':
        data = pd.read_parquet(file_list, columns=['hexid', 'callsign'], 
                               engine='pyarrow', dtype_backend='pyarrow')
        data = data.rename(columns=params.mapping_opensky)
    elif state == 'clean':
        data = pd.read_parquet(file_list, columns=['icao24', 'callsign'], 
                               engine='pyarrow', dtype_backend='pyarrow')
    uniqueness = data[['icao24','callsign']].nunique()
    results['uniqueness'] = {col:val for col, val in uniqueness.items()}

    with open(output_path, 'w+', encoding='utf8') as file:
        json.dump(results, file, indent=2, default=utils.custom_json_encoder)

def calculate_metrics_taf(month: str, data: pd.DataFrame, state: str = 'clean') -> None:
    results = {}
    results['month'] = month
    results['state'] = state
    results['level'] = 'L0' if state=='raw' else 'L1'

    results['num_reports'] = len(data)
    results['num_stations'] = data.station_id.nunique()
    completitude = data.notnull().sum()
    results['completitude'] = {col:val/len(data) for col, val in completitude.items()}
    uniqueness = data.drop(['sky_condition','turbulence_condition','icing_condition','temperature'], axis=1).nunique()
    results['uniqueness'] = {col:val for col, val in uniqueness.items()}
    results['ranges'] = {
        'min_temp':data.min_temp.min(),
        'max_temp':data.max_temp.max(),
    }
    results['reports_per_type'] = {col:val for col,val
        in data.groupby('change_indicator').count().station_id.items()}
    
    if state == 'clean':
        filepath = TAF_METRICS_L1_PATH / f'taf.L1.{month}.json'
    elif state == 'raw':
        filepath = TAF_METRICS_L0_PATH / f'taf.L0.{month}.json'
    if not filepath.parent.exists():
        filepath.parent.mkdir(parents=True)
    with open(filepath, 'w+', encoding='utf8') as file:
        json.dump(results, file, indent=2, default=utils.custom_json_encoder)

### L2/L3 -----------------------------------------------------------------------------------------

def calculate_metrics_trajectories(date: str, trayType: str = 'raw'):
    if trayType == 'raw':
        folder = NM_TRAJECTORIES_RAW_PATH
    elif trayType == 'clean':
        folder = NM_TRAJECTORIES_PATH
    ifplIds = pd.read_parquet(folder / f'tray.{date}.parquet', columns=['ifplId'],
                              engine='pyarrow', dtype_backend='pyarrow', ).ifplId.drop_duplicates()
    
    for t_id in ifplIds.values:
        calculate_metrics_trajectory(date, t_id, trayType)

    # args = [(date, t_id, trayType) for t_id in ifplIds.values]
    # with concurrent.futures.ThreadPoolExecutor(max_workers=7) as executor:
    #     executor.map(lambda x: calculate_metrics_trajectory(*x), args, chunksize=100)

def calculate_metrics_trajectory(date: str, trajectoryId: str, trayType: str = 'raw'):
    if trayType == 'raw':
        folder = NM_TRAJECTORIES_RAW_PATH
    elif trayType == 'clean':
        folder = NM_TRAJECTORIES_PATH
    data = pd.read_parquet(folder / f'tray.{date}.parquet', filters=[('ifplId', '==', trajectoryId)],
                           engine='pyarrow', dtype_backend='pyarrow', )
    with open(folder / f'flightDate={date}' / f'tray.{trajectoryId}.json', 'r', encoding='utf8') as file:
        metadata = json.load(file)
        
    results = {}
    results['ifplId'] = trajectoryId
    ## Generic
    completitude = data[['timestamp', 'latitude', 'longitude', 'baro_altitude', 'geo_altitude', 
                         'callsign', 'vertical_rate', 'velocity', 'altitude', 'true_track']].notnull().sum()
    results['completitude'] = {col:val/len(data) for col, val in completitude.items()}
    results['num_vectors'] = len(data)
    results['duration'] = data.timestamp.max() - data.timestamp.min()
    results['distance'] = calculate_traveled_distance(data)

    ## Semantic
    results['distance_to_origin'] = calculate_distance_to_airport(data, metadata['aerodromeOfDeparture'], where='origin')
    results['distance_to_destination'] = calculate_distance_to_airport(data, metadata['aerodromeOfDestination'], where='destination')
    results['missing_start'] = bool(results['distance_to_origin'] > THRESHOLD_DISTANCE_TO_AIRPORT)
    results['missing_end'] = bool(results['distance_to_destination'] > THRESHOLD_DISTANCE_TO_AIRPORT)
    results['airports_distance'] = calculate_distance_airports(metadata['aerodromeOfDeparture'], metadata['aerodromeOfDeparture'])
    results['effective_flight_time'] = metadata['actualTimeOfArrival']-metadata['actualTakeOffTime']
    # results['missing_taxi_start'] = bool(data[(data['distance_to_origin'] < AIRPORT_AREA) & data.on_ground])
    # results['missing_taxi_end'] = bool(data[(data['distance_to_destination'] < AIRPORT_AREA) & data.on_ground])
    results['last_altitude_before_ground'] = data.loc[data[~data.on_ground].timestamp.idxmax()].altitude

    ## Coverage and density
    results['density'] = results['num_vectors']/results['distance']
    results['mean_granularity'] = calculate_mean_granularity(data)
    results['std_granularity'] = calculate_std_granularity(data)
    results['mean_granularity_distance'] = calculate_mean_granularity_distance(data)
    results['std_granularity_distance'] = calculate_std_granularity_distance(data)
    results['gaps'] = identify_gaps(data)
    results['num_gaps'] = len(results['gaps'])
    results['segments'] = []
    latest = 0
    for g in results['gaps']:
        results['segments'].append(dict(start=latest, end=g['index']))
        latest = g['index']+1
    else:
        results['segments'].append(dict(start=latest, end=len(data)-1))
    results['num_segments'] = len(results['segments'])
    results['gap_time'] = calculate_gap_time(data)
    results['gap_ratio'] = results['gap_time']/results['duration'] if results['duration'] else 0
    results['continuity_time'] = calculate_continuity_time(data)
    results['continuity_ratio'] = results['continuity_time']/results['duration'] if results['duration'] else 0
    results['discontinuity_time'] = calculate_discontinuity_time(data)
    results['discontinuity_ratio'] = results['discontinuity_time']/results['duration'] if results['duration'] else 0
    results['thresholds'] = dict(
        THRESHOLD_DISTANCE_TO_AIRPORT = THRESHOLD_DISTANCE_TO_AIRPORT,
        THRESHOLD_GAP_TIME = THRESHOLD_GAP_TIME,
        THRESHOLD_CONTINUITY = THRESHOLD_CONTINUITY,
        AIRPORT_AREA=AIRPORT_AREA,
    )
    
    if pd.isna(results['last_altitude_before_ground']):
        results['last_altitude_before_ground'] = None

    if trayType == 'raw':
        with open(NM_TRAYS_METRICS_L2_PATH / f'tray.{date}.{trajectoryId}.json', 'w+', encoding='utf8') as file:
            json.dump(results, file, indent=2, default=utils.custom_json_encoder)
            # try:
            # except TypeError:
            #     print(results)
            #     exit()
    elif trayType == 'clean':
        results['sorted_vectors'] = get_resorted_vectors(data)
        results['timestamp_variation'] = get_timestamp_variation(data)
        with open(NM_TRAYS_METRICS_L3_PATH / f'tray.{date}.{trajectoryId}.json', 'w+', encoding='utf8') as file:
            json.dump(results, file, indent=2, default=utils.custom_json_encoder)    

    return results

def calculate_traveled_distance(data):
    return sum(utils.haversine_np(data.latitude.iloc[1:].values, data.longitude.iloc[1:].values, 
                                   data.latitude.iloc[:-1].values, data.longitude.iloc[:-1].values))

def calculate_distance_to_airport(data, airport: str, where: str = 'origin'):
    ap_location = pd.read_parquet(AIRPORTS_PATH, engine='pyarrow', filters=[('icao', '==', airport)])
    if where == 'origin':
        vector = data.iloc[0]
    elif where == 'destination':
        vector = data.iloc[-1]

    return utils.haversine_np(vector.latitude, vector.longitude, 
                               ap_location.latitude, ap_location.longitude)[0]

def calculate_distance_airports(airport_dep: str, airport_dest: str):
    origin_airport = airports[airports.icao == airport_dep].iloc[0]
    destination_airport = airports[airports.icao == airport_dest].iloc[0]
    distance = utils.haversine_np(origin_airport.latitude, 
                                  origin_airport.longitude, 
                                  destination_airport.latitude, 
                                  destination_airport.longitude)
    return distance

def calculate_mean_granularity(data):
    return np.mean(data.timestamp.iloc[1:].values - data.timestamp.iloc[:-1].values)

def calculate_std_granularity(data):
    return np.std(data.timestamp.iloc[1:].values - data.timestamp.iloc[:-1].values)

def calculate_mean_granularity_distance(data):
    return float(np.mean(utils.haversine_np(data.latitude[1:].values, data.longitude[1:].values, 
                                            data.latitude[:-1].values, data.longitude[:-1].values)))

def calculate_std_granularity_distance(data):
    return float(np.std(utils.haversine_np(data.latitude[1:].values, data.longitude[1:].values, 
                                      data.latitude[:-1].values, data.longitude[:-1].values)))

def identify_gaps(data):
    diffs = data.timestamp.iloc[1:].values - data.timestamp.iloc[:-1].values
    gaps = [dict(index=i, size=d) for i, d in enumerate(diffs) if d>THRESHOLD_GAP_TIME]
    
    return gaps

def calculate_continuity_time(data):
    diffs = data.timestamp.iloc[1:].values - data.timestamp.iloc[:-1].values
    return sum(diffs[diffs<=THRESHOLD_CONTINUITY])

def calculate_discontinuity_time(data):
    diffs = data.timestamp.iloc[1:].values - data.timestamp.iloc[:-1].values
    return sum(diffs[(THRESHOLD_CONTINUITY<diffs)&(diffs<=THRESHOLD_GAP_TIME)])

def calculate_gap_time(data):
    diffs = data.timestamp.iloc[1:].values - data.timestamp.iloc[:-1].values
    return sum(diffs[diffs>THRESHOLD_GAP_TIME])

def calculate_distance_ratio(data):
    # TODO: Ratio de distancia cubierta en línea recta respecto a la distancia en línea recta que
    # separa los aeropuertos de origen y destino
    
    return 


def get_outliers_position():
    # WIP
    pass

def get_outliers_altitude():
    # WIP
    pass

def get_resorted_vectors(data):
    return data.reordenado.sum()

def get_timestamp_variation(data):
    tmp = data[data.reordenado]
    if len(tmp)>0:
        return (tmp.timestamp - tmp.original_timestamp).mean()
    else:
        return 0

# if __name__ == '__main__':
#     results = calculate_metrics_trajectory('2022-07-11', 'AT03123212')