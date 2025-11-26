import datetime
import json

import pandas as pd
from tqdm import tqdm

import metrics
import params
import utils
from paths import *

### PARAMETERS
TIMEZONE_DISPL = 0

### UTILS -----------------------------------------------------------------------------------------

def flatten_dict(data: dict, paths: list) -> list:
    """ Extract elements from a nested dictionary

    The nested structure is

    Args:
        data: Nested dictionary
        paths: A list with keys representing the attribute paths to be extracted with dot notation

    Returns:
        A list of extracted elements

    Example:
        data = [
            {"a": {"b": {"c": 1, "e": 2}}, "d": 4},
            {"a": {"b": {"c": 1}}, "d": 4, "f": 3}
        ]
        paths = ["a.b.c", "d", "f"]
        result = flatten_dict(data, paths)
        # result will be [[1, 2, None], [1, 4, 3]]
    """
    paths = [x.split('.') for x in paths]
    def extract_attributes(record):
        def extract_attribute_rec(path: list[str], elem: dict = record, pos: int = 0):
            elem = elem.get(path[pos], None)
            if isinstance(elem, dict):
                return extract_attribute_rec(path, elem, pos+1)
            else:
                return elem
        return list(map(extract_attribute_rec, paths))
    return list(map(extract_attributes, data))

def convert_time_column(column):
    column = pd.to_datetime(column, format='ISO8601').astype('int64[pyarrow]')//10**9
    # TODO: Si no se hace esto, se generan fechas extrañas (año 1600). Comprobar
    # column = column.apply(lambda x: x if x>0 else pd.NA)
    column = column-TIMEZONE_DISPL

    return column


### FLIGHT PLANS ----------------------------------------------------------------------------------

def nm_fplan_json_to_parquet(date: str) -> None:
    """Parse NM flight plan data from a JSON file and write into a parquet file

    Args:
        date: String with a date in format 'YYYY-MM-DD'
    """

    ## Data load --------------------------------------------------------------
    folder = NM_JSON_FPLAN_PATH / f'flightDate={date}'
    # with open(folder / f'nm.fplan.{date}.json', 'r', encoding='utf8') as file:
        # data = json.load(file) # , separators=['\n',':']
    with open(folder / f'flightDate={date}.json', 'r', encoding='utf8') as file:
        data = [json.loads(x) for x in file]

    data = flatten_dict(data, params.mapping_flightPlan.values())
    column_names = params.mapping_flightPlan.keys()
    fplan = pd.DataFrame(data, columns=column_names)
    del data

    ## Data cleaning ----------------------------------------------------------
    # Data type
    fplan['timestamp'] = convert_time_column(fplan.timestamp)
    fplan['estimatedOffBlockTime'] = convert_time_column(fplan.estimatedOffBlockTime)
    string_columns = [
        'ifplId', 'icao24', 'callsign', 'operator', 'operatingOperator',
        'aerodromeOfDeparture', 'aerodromeOfDestination', 'flightType',
        'wakeTurbulenceCategory', 'uuid', 'registrationMark']
    for c in string_columns:
        fplan[c] = fplan[c].astype('string[pyarrow]')

    # Calculate raw data metrics (L0)
    metrics.calculate_metrics_fplan(date, fplan, 'raw')

    # Clean duplicates
    dups_columns = fplan.columns.difference(['uuid'])
    fplan = fplan.drop_duplicates(subset=dups_columns)

    # Sort messages
    fplan = fplan.sort_values(by=['ifplId', 'timestamp']).reset_index(drop=True)

    # Data format
    fplan['icao24'] = fplan.icao24.str.upper().str.strip()
    fplan['callsign'] = fplan.callsign.str.upper().str.strip()
    fplan['aerodromeOfDeparture'] = fplan.aerodromeOfDeparture.str.strip()
    fplan['aerodromeOfDestination'] = fplan.aerodromeOfDestination.str.strip()
    fplan['totalEstimatedElapsedTime'] = fplan.totalEstimatedElapsedTime.apply(
        lambda x: (int(x[:2])*60+int(x[2:])) if x else x).astype('int32[pyarrow]')

    # Fill missing attributes in the last fplan message
    fplan['ifplId_group'] = fplan.ifplId.copy()
    propagate_columns = [
        'icao24', 'registrationMark', 'ssr', 'flightType',
        'totalEstimatedElapsedTime', 'wakeTurbulenceCategory']
    for pc in propagate_columns:
        fplan[pc] = fplan.groupby('ifplId_group')[pc].ffill()
    fplan = fplan.drop('ifplId_group', axis=1)

    fplan = fplan.drop_duplicates(subset=dups_columns, keep='last').reset_index(drop=True)

    # Calculate clean data metrics (L1)
    metrics.calculate_metrics_fplan(date, fplan, 'clean')

    ## Save data --------------------------------------------------------------
    folder = NM_PARQUET_FPLAN_PATH
    if not folder.exists():
        folder.mkdir(parents=True)
    path = folder / f'nm.fplan.{date}.parquet'
    fplan.to_parquet(path, index=False)

def nm_fdata_json_to_parquet(date: str) -> None:
    """Parse NM flight data from a JSON file and write into a parquet file

    Args:
        date: String with a date in format 'YYYY-MM-DD'
    """

    ## Data load --------------------------------------------------------------
    data = []
    file_list = list((NM_JSON_FDATA_PATH / f'flightDate={date}').glob('*.json'))
    for file_path in tqdm(file_list, desc=f'{date} FDATA   | Clean  ', ncols=125):
        with open(file_path, 'r', encoding='utf8') as file:
            chunk = [json.loads(x) for x in file]
            # chunk = json.load(file) # , separators=['\n',':']
        chunk = flatten_dict(chunk, params.mapping_flightData.values())
        column_names = params.mapping_flightData.keys()
        chunk = pd.DataFrame(chunk, columns = column_names)
        data.append(chunk)
    fdata = pd.concat(data)
    del data

    ## Data cleaning ----------------------------------------------------------
    # Data type
    fdata['routeLength'] = fdata.routeLength.astype('Int32[pyarrow]')
    fdata['flightDataVersionNr'] = fdata.flightDataVersionNr.astype('Int32[pyarrow]')
    time_columns = [
        'timestamp',
        'estimatedOffBlockTime', 'actualOffBlockTime',
        'estimatedTakeOffTime', 'actualTakeOffTime',
        'estimatedTimeOfArrival', 'actualTimeOfArrival',
        'calculatedTakeOffTime', 'calculatedTimeOfArrival' ]
    for c in time_columns:
        fdata[c] = convert_time_column(fdata[c])
    string_columns = [
        'ifplId', 'icao24', 'callsign', 'aerodromeOfDeparture', 'aerodromeOfDestination',
        'operator', 'operatingOperator', 'flightState', 'aircraftType', 'uuid']
    for c in string_columns:
        fdata[c] = fdata[c].astype('string[pyarrow]')

    # Calculate raw data metrics (L0)
    metrics.calculate_metrics_fdata(date, fdata, 'raw')

    # Clean duplicates
    dups_columns = fdata.columns.difference(['uuid'])
    fdata = fdata.drop_duplicates(subset=dups_columns)
    fdata = fdata.sort_values(by=['ifplId', 'flightDataVersionNr']).reset_index(drop=True)

    # Data format
    fdata['icao24'] = fdata.icao24.str.upper().str.strip()
    fdata['callsign'] = fdata.callsign.str.strip()
    fdata['aerodromeOfDeparture'] = fdata.aerodromeOfDeparture.str.strip()
    fdata['aerodromeOfDestination'] = fdata.aerodromeOfDestination.str.strip()

    # Fill ICAO24 of the last version of flight data
    fdata['ifplId_group'] = fdata.ifplId.copy()
    propagate_columns = ['icao24', 'actualTakeOffTime', 'actualTimeOfArrival']
    for pc in propagate_columns:
        fdata[pc] = fdata.groupby('ifplId_group')[pc].ffill()
    fdata = fdata.drop('ifplId_group', axis=1)

    # Final clean
    fdata = fdata.drop_duplicates(keep='last').reset_index(drop=True)

    # Calculate clean data metrics (L1)
    metrics.calculate_metrics_fdata(date, fdata, 'clean')

    ### Save data ####################
    folder = NM_PARQUET_FDATA_PATH
    if not folder.exists():
        folder.mkdir(parents=True)
    path = folder / f'nm.fdata.{date}.parquet'
    fdata.to_parquet(path, index=False)


### VECTORS ---------------------------------------------------------------------------------------

def vectors_clean_parquet(date: str) -> None:
    file_paths = list(OPENSKY_RAW_VECTORS_PATH.glob(f'flightDate={date}/*.parquet'))

    dir = OPENSKY_PARQUET_VECTORS_PATH / f'flightDate={date}'
    if not dir.exists():
        dir.mkdir(parents=True)
    for file_path in tqdm(file_paths, desc=f'{date} VECTORS | Clean  ', ncols=125, disable=False):
        data = pd.read_parquet(file_path, engine='pyarrow', dtype_backend='pyarrow')
        data = vectors_clean(data)
        data.to_parquet(OPENSKY_PARQUET_VECTORS_PATH / f'flightDate={date}' / file_path.name, index=False)

def vectors_clean(data: pd.DataFrame) -> pd.DataFrame:
    """Processes individual vectors data problems

    Args:
        data: Dataframe with a day of vectors data
    """

    # Remove unused columns
    data = data.drop(['sensors', 'spi', 'position_source'], axis=1)

    # Rename vector attributes
    data = data.rename(columns=params.mapping_opensky)

    # Data types
    data['longitude'] = data.longitude.astype('Float32[pyarrow]')
    data['latitude'] = data.latitude.astype('Float32[pyarrow]')
    data['baro_altitude'] = data.baro_altitude.astype('Float32[pyarrow]')
    data['geo_altitude'] = data.geo_altitude.astype('Float32[pyarrow]')
    data['true_track'] = data.true_track.astype('Float32[pyarrow]')
    data['velocity'] = data.velocity.astype('Float32[pyarrow]')
    data['vertical_rate'] = data.vertical_rate.astype('Float32[pyarrow]')
    data['time_position'] = data.time_position.astype('Int64[pyarrow]')//10**9
    data['last_contact'] = data.last_contact.astype('Int64[pyarrow]')//10**9
    data['on_ground'] = data.on_ground.astype('Boolean[pyarrow]')

    # Remove vectors with null or incorrect values
    to_remove = (
        data.longitude.isna() |
        data.latitude.isna() |
        data.icao24.isna() |
        (~data.longitude.between(-180,180)) |
        (~data.latitude.between(-90,90))
    )
    data = data[~to_remove].copy()

    # Remove vectors constructed with reused positions
    data = data.drop_duplicates(subset=['icao24','time_position','latitude','longitude'])

    # Clean trailing spaces in callsign
    data['callsign'] = data.callsign.str.strip(' ')

    # Format
    data['icao24'] = data.icao24.str.upper()
    data['callsign'] = data.callsign.str.upper()

    # Define NA value for text attributes
    data['callsign'] = data.callsign.replace('', pd.NA)
    data['origin_country'] = data.origin_country.replace('', pd.NA)

    # Sort vectors
    data = data.sort_values(by=['icao24','time_position'])

    # Add columns
    # Use latest position time as the state vector timestamp
    data['timestamp'] = data.time_position.copy()
    # Use baro_altitude as default altitude
    data['altitude'] = data.geo_altitude.copy()

    # Unique ID
    # data['vectorId'] = date.replace('-','') + '-' + data.icao24 + '-' + data.timestamp.astype(str)

    # Sort columns
    data = data[params.vector_attribute_names]

    return data


### TAF REPORTS -----------------------------------------------------------------------------------

def taf_clean_parquet(month: str) -> None:
    """Parse TAF weather data (decoded) from a parquet file and write into a parquet file

    Args:
        month: String with a month in format 'YYYY-MM'
    """

    folder = TAF_RAW_PATH / f'month={month}'
    data = pd.read_parquet(folder, engine='pyarrow', dtype_backend='pyarrow')

    # Drop unused columns
    data = data.drop(['form', 'raw_text'], axis=1)

    # Data types
    data['station_id'] = data.station_id.astype('string[pyarrow]')
    data['change_indicator'] = data.change_indicator.astype('string[pyarrow]')
    data['wx_string'] = data.wx_string.astype('string[pyarrow]')

    data['probability'] = data.probability.astype('int32[pyarrow]')
    data['wind_dir_degrees'] = data.wind_dir_degrees.astype('int32[pyarrow]')
    data['wind_speed_kt'] = data.wind_speed_kt.astype('int32[pyarrow]')
    data['wing_gust_kt'] = data.wing_gust_kt.astype('int32[pyarrow]')
    data['wind_shear_hgt_ft_agl'] = data.wind_shear_hgt_ft_agl.astype('int32[pyarrow]')
    data['wind_shear_dir_degrees'] = data.wind_shear_dir_degrees.astype('int32[pyarrow]')
    data['wind_shear_speed_kt'] = data.wind_shear_speed_kt.astype('int32[pyarrow]')
    data['vert_vis_ft'] = data.vert_vis_ft.astype('int32[pyarrow]')

    data['altim_in_hg'] = data.altim_in_hg.astype('float32[pyarrow]')
    data['visibility_statute_mi'] = data.visibility_statute_mi.astype('float32[pyarrow]')

    # Add columns
    # Temperature
    def extract_temps(temp_records):
        if len(temp_records)==0:
            return [pd.NA]*4
        elif len(temp_records)==1:
            if temp_records[0]['min_temp_c']:
                return [pd.NA,pd.NA,temp_records[0]['min_temp_c'],temp_records[0]['valid_time']]
            elif temp_records[0]['max_temp_c']:
                return [temp_records[0]['max_temp_c'],temp_records[0]['valid_time'],pd.NA,pd.NA]
            else:
                return [pd.NA]*4
        elif len(temp_records)>1:
            res = [pd.NA]*4
            for rec in temp_records[:2]:
                if rec['min_temp_c']:
                    res[2] = rec['min_temp_c']
                    res[3] = rec['valid_time']
                elif rec['max_temp_c']:
                    res[0] = rec['max_temp_c']
                    res[1] = rec['valid_time']
            return res
    temperatures = list(map(extract_temps, data.temperature.values.tolist()))
    temperatures = pd.DataFrame(temperatures, columns=['max_temp','max_temp_timestamp','min_temp','min_temp_timestamp'])
    data[['max_temp','max_temp_timestamp','min_temp','min_temp_timestamp']] = temperatures.values
    data[['max_temp','min_temp']] = data[['max_temp','min_temp']].astype('int32[pyarrow]')

    # Sky condition
    def extract_sky_conditions(sky_record):
        if len(sky_record)==0:
            return [pd.NA]*3
        elif len(sky_record)>0:
            return [
                sky_record[0]['sky_cover'],
                sky_record[0]['cloud_base_ft_agl'],
                sky_record[0]['cloud_type'] if sky_record[0]['cloud_type'] else pd.NA
            ]
    sky_conditions = list(map(extract_sky_conditions, data.sky_condition.values.tolist()))
    sky_conditions = pd.DataFrame(sky_conditions, columns=['sky_cover','cloud_base_ft_agl','cloud_type'])
    data[['sky_cover','cloud_base_ft_agl','cloud_type']] = sky_conditions.values
    data[['sky_cover','cloud_type']] = data[['sky_cover','cloud_type']].astype('string[pyarrow]')

    # Calculate raw data metrics (L0)
    metrics.calculate_metrics_taf(month, data, 'raw')

    # Icing condition
    # Almost always empty, it is not worth

    # Fix column values
    data['wind_dir_degrees'] = data.wind_dir_degrees.astype('float') % 360
    # Si valid_time_from es nulo, asignamos issue_time + 1h
    data['valid_time_from'] = data.valid_time_from.combine_first(data.issue_time+datetime.timedelta(hours=1))
    # Si valid_time_to es nulo, asignamos valid_time_from + 30h
    data['valid_time_to'] = data.valid_time_from.combine_first(data.issue_time+datetime.timedelta(hours=30))

    # Fix NA values
    for col in ['sky_condition','turbulence_condition','icing_condition','temperature']:
        data[col] = data[col].apply(lambda x: x if len(x)>0 else pd.NA)

    # Calculate raw data metrics (L0)
    metrics.calculate_metrics_taf(month, data, 'clean')

    folder = TAF_PARQUET_PATH
    if not folder.exists():
        folder.mkdir(parents=True)
    path = folder / f'taf.{month}.parquet'
    data.to_parquet(path, index=False)


### AIRPORTS --------------------------------------------------------------------------------------

def airports_json_to_parquet() -> None:
    """Parse and transforms airport data from a CSV file and write into a parquet file
    """
    with open(AIRPORTS_RAW_PATH, 'r', encoding='utf8') as file:
        data = json.load(file)['rows']
    data = pd.DataFrame.from_dict(data)
    data['alt'] = data.alt.astype(int)

    data = data.rename(dict(
        lat='latitude',
        lon='longitude',
        alt='altitude',
    ), axis=1)

    if not AIRPORTS_PATH.parent.exists():
        AIRPORTS_PATH.parent.mkdir(parents=True)
    data.to_parquet(AIRPORTS_PATH, engine='pyarrow', index=False)


if __name__ == '__main__':
    date_start, date_end = '2023-07-01','2023-07-16'

    dates = utils.get_dates_between(date_start, date_end)
    dates = [x.strftime('%Y-%m-%d') for x in dates]

    for date in dates:
        pass
        # nm_fplan_json_to_parquet(date)
        # nm_fdata_json_to_parquet(date)

        metrics.calculate_metrics_openskyVectors(date, state='raw')
        # vectors_clean_parquet(date)
        metrics.calculate_metrics_openskyVectors(date, state='clean')

    # taf_clean_parquet('2023-07')
    # airports_json_to_parquet()

### Currently not used ----------------------------------------------------------------------------

# OpenSky flights not used
def op_json_to_parquet(date: str) -> None:
    """Parse OpenSky flight data from a JSON file and write into a parquet file

    Assigns each flight to the day in which it ends. Loads data from both data
    and the previous day, and filters them based on their lastSeen timestamp.

    Args:
        date: String with a date in format 'YYYY-MM-DD'
    """
    # TODO: Revisar
    date_dt = datetime.datetime.strptime(date, '%Y-%m-%d')
    date_prev_dt = date_dt - datetime.timedelta(days=1)
    date_prev = date_prev_dt.strftime('%Y-%m-%d')

    file_paths = []
    if (OPENSKY_RAW_FLIGHTS_PATH / f'flightDate={date_prev}').exists():
        file_paths += list(OPENSKY_RAW_FLIGHTS_PATH.glob(f'flightDate={date_prev}/*.json'))
    file_paths += list(OPENSKY_RAW_FLIGHTS_PATH.glob(f'flightDate={date}/*.json'))

    data = []
    for file_path in file_paths:
        chunk_df = pd.read_json(file_path)
        data.append(chunk_df)
    data = pd.concat(data)
    data = data[data.lastSeen.between(
        date_dt.timestamp(),
        (date_dt+datetime.timedelta(days=1,seconds=-1)).timestamp()
    )]
    data['flightDate'] = date

    # Clean flights
    data = data.sort_values(by=['icao24','firstSeen'])
    # Data type
    data['firstSeen'] = data.firstSeen.astype(int)
    data['lastSeen'] = data.lastSeen.astype(int)

    # Data format
    data['icao24'] = data.icao24.str.strip().str.upper()
    data['callsign'] = data.callsign.str.upper()

    # Column projection
    data = data[['icao24','firstSeen','estDepartureAirport','lastSeen','estArrivalAirport','callsign','flightDate']].copy()

    # Duplicates
    data = data.drop_duplicates()

    # Unique ID
    data['flightId'] = data.flightDate.str.replace('-','') + '-' + data.index.astype(str).str.ljust(6, '0')

    folder = OPENSKY_PARQUET_FLIGHTS_PATH
    if not folder.exists():
        folder.mkdir(parents=True)
    path = folder / f'os.flights.{date}.parquet'
    data.to_parquet(path, index=False)

# OpenSky vectors are already in parquet format
def vectors_json_to_parquet(date: str) -> None:
    """Parse OpenSky state vectors from a JSON file and write into a parquet file

    Args:
        date: String with a date in format 'YYYY-MM-DD'
    """
     # 'category'
    file_paths = OPENSKY_RAW_VECTORS_JSON_PATH.glob(f'flightDate={date}/*.json')

    for file_path in list(file_paths):
        data = []
        with open(file_path, 'r', encoding='utf8') as file:
            # One-shot
            # Requires exploding into columns
            # chunk_df = pd.read_json(file, lines=True)

            # Iterative, line by line
            chunks = []
            for line in tqdm(file, desc=f'{date} VECTORS', ncols=125):
                record = json.loads(line)
                chunk = pd.DataFrame(record['states'], columns=params.vector_attribute_names)
                chunk['timestamp'] = record['time']
                chunks.append(chunk)
            chunks_df = pd.concat(chunks)
            data.append(chunks_df)
        data = pd.concat(data)

        # Clean vectors
        data = vectors_clean(data)

        folder = OPENSKY_PARQUET_VECTORS_PATH / f'flightDate={date}'
        if not folder.exists():
            folder.mkdir()
        data.to_parquet(folder / f'{file_path.stem}.parquet', index=False)

