import datetime
import json

import pandas as pd

import utils
import params
import metrics
from paths import *
from tqdm import tqdm

### PARAMETERS
airport_orig = ['EHAM', 'EDDF', 'LIRF', 'LFPG', 'LGAV', 'EKCH', 'EGLL', 'LEMD']
airport_dest = ['EHAM', 'EDDF', 'LIRF', 'LFPG', 'LGAV', 'EKCH', 'EGLL', 'LEMD']

### FLIGHTS ---------------------------------------------------------------------------------------

def nm_merge_fp_fd(date: str) -> None:
    """Merge NM flight plan and flight data from a given date

    Args:
        date: String with a date in format 'YYYY-MM-DD'
    """

    fplan = pd.read_parquet(NM_PARQUET_FPLAN_PATH / f'nm.fplan.{date}.parquet', 
                            engine='pyarrow', dtype_backend='pyarrow')
    fdata = pd.read_parquet(NM_PARQUET_FDATA_PATH / f'nm.fdata.{date}.parquet', 
                            engine='pyarrow', dtype_backend='pyarrow')
    fplan = fplan.drop('uuid', axis=1)
    fdata = fdata.drop(['uuid', 'timestamp'], axis=1)

    # Find last flight plan version
    last_fplan = (fplan.groupby('ifplId').timestamp.idxmax())
    fplan = fplan.loc[last_fplan]

    # Find last flight data version
    # last_fdata = (fdata.groupby('ifplId').flightDataVersionNr.idxmax())
    # fdata = fdata.loc[last_fdata]
    # Para evitar últimos FDATA que modifican actualTimeOfArrival
    last_fdata = []
    for grid, gr in fdata.groupby('ifplId'):
        if len(gr[gr.flightState == 'TERMINATED'])>1:
            l = gr[gr.flightState == 'TERMINATED'].iloc[0]
        else:
            l = gr.loc[gr.flightDataVersionNr.idxmax()]
        last_fdata.append(l)
    fdata = pd.DataFrame(last_fdata)

    # Join FP-FD and consolidate duplicated columns
    flights = pd.merge(fplan, fdata, on = 'ifplId')
    flights['icao24'] = flights.icao24_y.combine_first(flights.icao24_x)
    flights = flights.drop(['icao24_x', 'icao24_y'], axis = 1)
    flights['callsign'] = flights.callsign_y.combine_first(flights.callsign_x)
    flights = flights.drop(['callsign_x', 'callsign_y'], axis = 1)
    flights['estimatedOffBlockTime'] = flights.estimatedOffBlockTime_y.combine_first(flights.estimatedOffBlockTime_x)
    flights = flights.drop(['estimatedOffBlockTime_x', 'estimatedOffBlockTime_y'], axis = 1)
    flights['aerodromeOfDeparture'] = flights.aerodromeOfDeparture_y.combine_first(flights.aerodromeOfDeparture_x)
    flights = flights.drop(['aerodromeOfDeparture_x', 'aerodromeOfDeparture_y'], axis = 1)
    flights['aerodromeOfDestination'] = flights.aerodromeOfDestination_y.combine_first(flights.aerodromeOfDestination_x)
    flights = flights.drop(['aerodromeOfDestination_x', 'aerodromeOfDestination_y'], axis = 1)
    flights['operator'] = flights.operator_y.combine_first(flights.operator_x)
    flights = flights.drop(['operator_x', 'operator_y'], axis = 1)
    flights['operatingOperator'] = flights.operatingOperator_y.combine_first(flights.operatingOperator_x)
    flights = flights.drop(['operatingOperator_x', 'operatingOperator_y'], axis = 1)
    flights['aircraftType'] = flights.aircraftType_y.combine_first(flights.aircraftType_x)
    flights = flights.drop(['aircraftType_x', 'aircraftType_y'], axis = 1)

    # Use only flights that end in date
    # -- This situation is not possible due to how flight plan messages are partitioned
    # flights = flights[flights.actualTimeOfArrival.between(
    #     date_dt.timestamp(), 
    #     (date_dt+datetime.timedelta(days=1,seconds=-1)).timestamp())
    # ]

    # Ensure that joined FP-FD pairs refer to the same flight (due to ifplId reutilization)
    flights = flights[(flights.actualTakeOffTime > flights.estimatedOffBlockTime - 24*3600) &
                      (flights.actualTimeOfArrival < flights.estimatedOffBlockTime + 2*24*3600)]
    
    flights = flights.drop_duplicates()

    # Drop timestamp and FDATA message version column
    flights = flights.drop(['timestamp', 'flightDataVersionNr'], axis=1)

    # Sort columns
    flights = flights[params.flight_attribute_names]

    folder = NM_PARQUET_FLIGHTS_PATH
    if not folder.exists():
        folder.mkdir(parents=True)
    path = NM_PARQUET_FLIGHTS_PATH / f'nm.flights.{date}.parquet'
    flights.to_parquet(path, index=False)


### VECTORS ---------------------------------------------------------------------------------------

def nm_integrate_flight_vectors(date: str, airports_dep: list|tuple = tuple(), airports_dest: list|tuple = tuple()) -> None:
    """Join flight data and state vectors to identify individual trajectories

    Args:
        date: String with a date in format 'YYYY-MM-DD'
        source: The data source for flight data
        airports: A list with the desired origin airports
    """

    integration_metrics = {}
    
    ## Data load --------------------------------------------------------------

    # Flight data
    filters = []
    # Filter by flight state
    filters.append(('flightState','in',('TERMINATED','ATC_ACTIVATED','TATC_ACTIVATED')))
    # Filter by airports
    if airports_dep:
        filters.append(('aerodromeOfDeparture','in',airports_dep))
    if airports_dest:
        filters.append(('aerodromeOfDestination','in',airports_dep))
    flights = pd.read_parquet(NM_PARQUET_FLIGHTS_PATH / f'nm.flights.{date}.parquet',
                              engine='pyarrow', dtype_backend='pyarrow', filters=filters)
    integration_metrics['num_flights_initial'] = flights.shape[0]
    # Remove flights with the same origin and destination
    flights = flights[flights.aerodromeOfDeparture != flights.aerodromeOfDestination]
    integration_metrics['num_flights'] = flights.shape[0]
    integration_metrics['returned_flights'] = integration_metrics['num_flights_initial'] - integration_metrics['num_flights']

    # OpenSky data
    # Include vectors from the next day (in case the flight takes place between two days)
    date_dt =  datetime.datetime.strptime(date, '%Y-%m-%d')
    date_next_dt = date_dt + datetime.timedelta(days=1)
    date_next = date_next_dt.strftime('%Y-%m-%d')

    file_paths = []
    file_paths += list(OPENSKY_PARQUET_VECTORS_PATH.glob(f'flightDate={date}/*.parquet'))
    first_day_files = len(file_paths)
    if (OPENSKY_PARQUET_VECTORS_PATH / f'flightDate={date_next}').exists():
        file_paths += list(OPENSKY_PARQUET_VECTORS_PATH.glob(f'flightDate={date_next}/*.parquet'))


    ## Integration ------------------------------------------------------------

    print(f'#       {"Vectors":<15}{"Flights":<15}{"Join vectors":<15}' +
          f'{"% join vec":<18}{"Join flights":<15}{"% join fl":<18}')
    
    joined_flights_acc = []
    joined_vectors_acc = []
    integration_metrics['num_vectors'] = 0
    for idx, file_path in enumerate(file_paths):
        vectors = pd.read_parquet(file_path, engine='pyarrow', dtype_backend='pyarrow')
        num_initial_vectors = vectors.shape[0]

        # Merge by icao24
        vectors_icao = vectors[vectors.icao24.isin(flights.icao24.unique())]
        if vectors_icao.shape[0] == 0:
            continue
        joined = pd.merge(vectors_icao.drop('callsign',axis=1), flights, on='icao24', how='inner')
        joined = joined[(joined.timestamp >= joined.actualTakeOffTime - params.TIME_SLACK) &
                        (joined.timestamp <= joined.actualTimeOfArrival + params.TIME_SLACK)]
        joined = joined.loc[:, vectors_icao.columns.to_list()+['ifplId']].drop_duplicates()
        joined_icao24 = joined.ifplId.drop_duplicates()
        if len(joined)>0:
            joined_flights_acc.append(flights[flights.ifplId.isin(joined_icao24)])
            joined_vectors_acc.append(joined)

        # Merge by callsign
        # vectors_callsign = vectors[vectors.callsign.isin(flights.callsign.unique()) & ~vectors.index.isin(vectors_icao.index)].copy()
        # joined = pd.merge(vectors_callsign, flights.drop('icao24',axis=1), on='callsign', how='inner')
        # joined = joined[(joined.timestamp >= joined.actualTakeOffTime - params.TIME_SLACK) &
        #                 (joined.timestamp <= joined.actualTimeOfArrival + params.TIME_SLACK)]
        # joined_flights_acc.append(flights[flights.ifplId.isin(joined.ifplId.drop_duplicates())])
        # joined_vectors = joined[vectors_callsign.columns.to_list()+['ifplId']].drop_duplicates()
        # joined_vectors_acc.append(joined_vectors)
        
        # Metrics
        nvec = num_initial_vectors
        nfl = flights.shape[0]
        njvec = joined.shape[0]
        njfl = joined_flights_acc[-1].shape[0]
        print(f'{idx+1:>3}/{len(file_paths)}  {nvec:<15}{nfl:<15}{njvec:<15}'+
              f'{njvec/nvec*100:<18.2f}{njfl:<15}{njfl/nfl*100:<18.2f}')
        if idx<first_day_files:
            integration_metrics['num_vectors'] += nvec

    ## Write data -------------------------------------------------------------

    if joined_vectors_acc:
        joined_vectors = pd.concat(joined_vectors_acc).drop_duplicates()
        joined_vectors = joined_vectors.sort_values(by=['ifplId', 'timestamp']).reset_index(drop=True)
        del joined_vectors_acc
        integration_metrics['num_joined_vectors'] = joined_vectors.shape[0]
        integration_metrics['num_joined_flights'] = len(joined_vectors.ifplId.unique())

        # Remove trajectories with too few vectors
        too_few_vectors = joined_vectors.groupby('ifplId').count()
        too_few_vectors = too_few_vectors[too_few_vectors.icao24>=params.VECTOR_NUMBER_MIN]
        joined_vectors = joined_vectors[joined_vectors.ifplId.isin(too_few_vectors.index)]
        integration_metrics['num_joined_vectors_final'] = joined_vectors.shape[0]
        integration_metrics['num_joined_flights_final'] = too_few_vectors.shape[0]
        integration_metrics['removed_short_trajectories'] = integration_metrics['num_joined_flights'] - too_few_vectors.shape[0]
        del too_few_vectors

        # Write trajectory data
        folder = NM_TRAJECTORIES_RAW_PATH
        if not folder.exists():
            folder.mkdir(parents=True)
        joined_vectors.reset_index(drop=True).to_parquet(folder / f'tray.{date}.parquet', index=False)

        # Write trajectory metadata
        folder = NM_TRAJECTORIES_RAW_PATH / f'flightDate={date}'
        if not folder.exists():
            folder.mkdir(parents=True)
        for g, gdata in joined_vectors.groupby('ifplId'):
            flight = flights[flights.ifplId == g]

            metadata = dict(
                date=date,
                ifplId=g,
                callsign=flight.callsign.values[0],
                icao24=flight.icao24.values[0],
                aerodromeOfDeparture=flight.aerodromeOfDeparture.values[0],
                aerodromeOfDestination=flight.aerodromeOfDestination.values[0],
                airline=flight.operator.values[0] if pd.notna(flight.operator.values[0]) else flight.operatingOperator.values[0],
                estimatedTakeOffTime=int(flight.estimatedTakeOffTime.values[0]),
                estimatedTimeOfArrival=int(flight.estimatedTimeOfArrival.values[0]),
                actualTakeOffTime=int(flight.actualTakeOffTime.values[0]),
                actualTimeOfArrival=int(flight.actualTimeOfArrival.values[0]),
                num_vectores=len(gdata),
                ts_start=gdata.timestamp.min(),
                ts_end=gdata.timestamp.max(),
                data_source_surveillance='opensky',
                data_source_flights='nm',
                flightState=flight.flightState.values[0],
                trajectory_status='L2_cleaned',
            )
            with open(folder / f'tray.{g}.json', 'w+', encoding='utf8') as file:
                json.dump(metadata, file, indent=2, default=utils.custom_json_encoder)
        
        # Write integration metrics
        with open(INTEGRATION_METRICS_PATH / f'integration.{date}.json', 'w+', encoding='utf8') as file:
            json.dump(integration_metrics, file, indent=2, default=utils.custom_json_encoder)

    # if joined_flights_acc:
    #     joined_flights = pd.concat(joined_flights_acc)
    #     del joined_flights_acc
    #     joined_flights = joined_flights.drop_duplicates()
    #     return joined_flights
    
    return None


if __name__ == '__main__':
    date_start, date_end = '2023-07-03','2023-07-05'

    dates = utils.get_dates_between(date_start, date_end)
    dates = [x.strftime('%Y-%m-%d') for x in dates]

    for date in dates:
        print('='*40, date,'='*40)
        # nm_merge_fp_fd(date)

        # nm_integrate_flight_vectors(date, airports_dep=airport_orig, airports_dest=airport_dest)
        metrics.calculate_metrics_trajectories(date, 'raw')


### Currently not used ----------------------------------------------------------------------------

# OpenSky flights not used
def opensky_integrate_flight_vectors(date: str, source: str, airports_dep: list|tuple = tuple()) -> None:
    """Join flight data and state vectors to identify individual trajectories

    Args:
        date: String with a date in format 'YYYY-MM-DD'
        source: The data source for flight data
        airports: A list with the desired origin airports
    """

    if source == 'opensky':
        # TODO Integración con OpenskyFlights
        flights = pd.read_parquet(OPENSKY_PARQUET_FLIGHTS_PATH / f'flightDate={date}')
        # Filter by airports
        if airports_dep:
            flights = flights[flights.estDepartureAirport.isin(airports_dep) & flights.estArrivalAirport.isin(airports_dep)]
        # Remove flights with the same origin and destination
        flights = flights[flights.estDepartureAirport != flights.estArrivalAirport]
    elif source == 'nm':
        flights = pd.read_parquet(NM_PARQUET_FLIGHTS_PATH / f'nm.flights.{date}.parquet')
        # flights = flights[flights.flightState == 'TERMINATED']
        # Filter by airports
        if airports_dep:
            flights = flights[flights.aerodromeOfDeparture.isin(airports_dep) & flights.aerodromeOfDestination.isin(airports_dep)]
        # Remove flights with the same origin and destination
        flights = flights[flights.aerodromeOfDeparture != flights.aerodromeOfDestination]
    else:
        print('Choose a valid flight data source.')
        return
    

    date_dt =  datetime.datetime.strptime(date, '%Y-%m-%d')
    date_prev_dt = date_dt - datetime.timedelta(days=1)
    date_prev = date_prev_dt.strftime('%Y-%m-%d')

    file_paths = []
    if (OPENSKY_PARQUET_VECTORS_PATH / f'flightDate={date_prev}').exists():
        file_paths += list(OPENSKY_PARQUET_VECTORS_PATH.glob(f'flightDate={date_prev}/*.parquet'))
    file_paths += list(OPENSKY_PARQUET_VECTORS_PATH.glob(f'flightDate={date}/*.parquet'))

    print(f'#        {"Vectors":<15}{"Flights":<15}{"Join vectors":<15}'+
          f'{"% join vec":<18}{"Join flights":<15}{"% join fl":<18}')
    
    joined_flights = []
    joined_vectors_acc = []
    for idx, file_path in enumerate(file_paths):
        vectors = pd.read_parquet(file_path, engine='pyarrow', dtype_backend='pyarrow').dropna(subset=['timestamp'])

        num_initial_vectors = vectors.shape[0]
        vectors = vectors[vectors.icao24.isin(flights.icao24.unique())]
        if vectors.shape[0] == 0:
            continue

        joined = pd.merge(vectors.drop('callsign',axis=1), flights, on='icao24', how='inner')
        if source == 'opensky':
            joined = joined[(joined.timestamp >= joined.firstSeen) &
                            (joined.timestamp <= joined.lastSeen)]
            joined_flights.append(flights[flights.flightId.isin(joined.flightId.drop_duplicates())])
            joined_vectors = joined[vectors.columns.to_list()+['flightId']].drop_duplicates()
        elif source == 'nm':
            joined = joined[(joined.timestamp >= joined.actualTakeOffTime - params.TIME_SLACK) &
                            (joined.timestamp <= joined.actualTimeOfArrival + params.TIME_SLACK)]
            joined_flights.append(flights[flights.ifplId.isin(joined.ifplId.drop_duplicates())])
            joined_vectors = joined[vectors.columns.to_list()+['ifplId']].drop_duplicates()
            
            # joined_vectors = joined_vectors.rename(parameters.NM_FLIGHTS_RENAME, axis=1)
        
        # if source == 'opensky':
        #     folder = OPENSKY_JOINED_VECTORS_PATH / f'flightDate={date}'
        # elif source == 'nm':
        #     folder = NM_TRAJECTORIES_RAW_PATH / f'flightDate={date}'
        
        # if not folder.exists():
        #     folder.mkdir(parents=True)
        # if len(joined_vectors)>0:
        #     path = folder / f'vectors.{joined_vectors.timestamp.min()}.parquet'
        #     joined_vectors.to_parquet(path, index=False,)

        joined_vectors_acc.append(joined_vectors)

        nvec = num_initial_vectors
        nfl = flights.shape[0]
        # njvec = joined_vectors[-1].shape[0]
        njvec = joined_vectors.shape[0]
        njfl = joined_flights[-1].shape[0]
        print(f'{idx+1:>3}/{len(file_paths)}  {nvec:<15}{nfl:<15}{njvec:<15}'+
              f'{njvec/nvec*100:<18.2f}{njfl:<15}{njfl/nfl*100:<18.2f}')
    
    if joined_vectors_acc:
        joined_vectors = pd.concat(joined_vectors_acc)
        joined_vectors = joined_vectors.drop_duplicates()
        joined_vectors = joined_vectors.sort_values(by=['ifplId', 'timestamp']).reset_index(drop=True)
        folder = NM_TRAJECTORIES_RAW_PATH
        if len(joined_vectors)>0:
            path = folder / f'vectors.{date}.parquet'
            joined_vectors.to_parquet(path, index=False,)
    
    if joined_flights:
        joined_flights = pd.concat(joined_flights)
        joined_flights = joined_flights.drop_duplicates()

    return joined_flights

    if source == 'opensky':
        # TODO Integración con OpenskyFlights
        folder = OPENSKY_JOINED_FLIGHTS_PATH / f'flightDate={date}'
        # Rename columns for later steps
        # joined_flights = joined_flights.rename(parameters.OP_FLIGHTS_RENAME, axis=1)
    elif source == 'nm':
        folder = NM_JOINED_FLIGHTS_PATH / f'flightDate={date}'
        # Rename columns for later steps
        # joined_flights = joined_flights.rename(parameters.NM_FLIGHTS_RENAME, axis=1)
    if not folder.exists():
        folder.mkdir()
    path =  folder / f'flights.{date.replace("-", "")}.parquet'
    joined_flights.to_parquet(path, index=False)
