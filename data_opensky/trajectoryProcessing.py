import concurrent
import json
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

import utils
import params
import metrics
from utils import haversine_np, haversine_np_track
from paths import *
from trajectory import Trajectory

from params import *

import warnings
warnings.simplefilter("ignore", category=FutureWarning)

# Parámetros del proceso
overlap_fast     = 5
window_size_fast = 20
overlap_slow     = 75
window_size_slow = 101

# No se aplican
# max_kchanges = 200
max_iteraciones = 100

airports = pd.read_parquet(AIRPORTS_PATH)

def fill_missing_data(trajectory: Trajectory) -> Trajectory:
    # trajectory.vectors = trajectory.vectors.drop_duplicates()

    return trajectory

def detect_outliers(trajectory: Trajectory) -> Trajectory:
    data = trajectory.vectors.copy()
    # The first vector is assumed to be correct
    first = data.iloc[0]
    latest = (first.latitude, first.longitude, first.timestamp, first.altitude, first.vspeed)
    flags = [False]

    for idx, row in list(data.iterrows())[1:]:
        # Time check
        diff_time = row.timestamp - latest[2]
        if diff_time == 0:
            flags.append(True)
            continue
        if diff_time > 60:
            flags.append(False)
            latest = (row.latitude, row.longitude, row.timestamp, row.altitude, row.vspeed)
            continue

        # Altitude check
        diff_altitude = abs(latest[3] - row.altitude) / diff_time
        exceeds_altitude = (diff_altitude > ((abs(latest[4]) + abs(row.vspeed)) / 120 + DIFF_ALTITUDE_THRESHOLD))
        if exceeds_altitude:
            flags.append(True)
            continue

        # Position check
        diff = haversine_np(float(row.latitude), float(row.longitude),
                            latest[0], latest[1]) / (row.timestamp - latest[2])
        exceeds_position = diff > DIFF_SPEED_THRESHOLD
        if exceeds_position:
            flags.append(True)
            continue

        latest = (row.latitude, row.longitude, row.timestamp, row.altitude, row.vspeed)
        flags.append(False)

    if suma := sum(flags):
        print(data.iloc[0].fpId, f'{suma:3}/{len(flags):5}/{data.shape[0]:5}')

    data['is_outlier'] = flags

    trajectory.vectors = data

    return trajectory

def resolve_position_outliers():
    pass

def resolve_time_outliers():
    pass

def sample_trajectory():
    pass


def calculate_distance(array: np.array, angle = False) -> np.array:
    array = array.astype('float32')

    if not angle:
        distances = haversine_np(array[1:,0], array[1:,1],
                                array[:-1,0], array[:-1,1],
                                None, None, angle)
    else: # Use track
        distances = haversine_np_track(array[1:,0], array[1:,1],
                                array[:-1,0], array[:-1,1],
                                array[1:,2], array[:-1,2], angle)

    return np.concatenate([[0], distances])


def generate_windows(data_size:int, window_size:int, overlap:int, min_index:int=0, max_index:int=0) -> tuple[tuple[int,int,int]]:
    '''Genera ventanas deslizantes a partir del tamaño de una estructura tabular

    Genera índices para las ventanas deslizantes de tamaño window_size aplicadas
    sobre una estructura de datos de longitud data_size.

    Args:
        data_size: Longitud de la estructura de datos
        window_size: Tamaño de la ventana deslizante
        overlap: Número de elementos que se solapan entre ventanas consecutivas
        min_index: Índice a partir del cual se calculan las ventanas
        max_index: Índice hasta el que se generan las ventanas

    Returns:
        Un conjunto de tripletas (num. ventana, índice inicial, índice final)
    '''
    if max_index and max_index <= data_size: #
        data_size = max_index

    windows = []
    number_of_windows = ((data_size-min_index)//(window_size-overlap))+1
    for x in range(number_of_windows):
        window_min = x*(window_size-overlap) + min_index
        window_max = x*(window_size-overlap) + min_index + window_size
        window_max = window_max if window_max <= data_size else data_size
        windows.append((x, window_min, window_max))
        if window_max == data_size:
            break

    return windows


def generate_changes(start:int, end:int, skip:int = 1) -> list[tuple[int,int]]:
    '''
        Genera reemplazos de cada elemento con cada uno de los elementos siguientes
        (sin contar el inmediatamente siguiente) en base a sus índices correlativos
        en la secuencia. Ignora el primer y último elementos.

        start
            Índice del primer elemento a considerar
        end
            Índice del último elemento a considerar
        skip
            Número de elementos al comienzo de la secuencia para los que no se generan cambios

    '''
    changes = [(v1,v2) for v1 in range(start+skip, end-start-2) for v2 in range(v1+2, end-start-1)]
    return changes


def sort_windows(array: np.array, windows: tuple, angle: bool = False) -> np.array:
    for it, start, end in windows:
        data = array[start:end].copy()
        lat_index = params.vector_attribute_names.index('latitude')
        lon_index = params.vector_attribute_names.index('longitude')
        track_index = params.vector_attribute_names.index('true_track')
        current_dist = initial_dist = np.sum(calculate_distance(data[:,[lat_index,lon_index,track_index]], angle))

        # Ojo: índices relativos a la ventana, no a data entero
        changes = generate_changes(0, end-start, skip = 2 if angle else 1)
        # changes = list(np.random.permutation(changes))

        for j in range(max_iteraciones):
            improvements = 0
            for i, (v1, v2) in enumerate(changes):
                candidate = np.concatenate([data[:v1+1],
                                            data[v2:v1:-1],
                                            data[v2+1:]])
                candidate_dist = np.sum(calculate_distance(candidate[:,[lat_index,lon_index,track_index]], angle))

                if candidate_dist < current_dist:
                    improvements += 1
                    data, current_dist = candidate, candidate_dist
                    array[start:end] = candidate

            if not improvements:
                break

    return array


def calculate_rotation(tracks: pd.Series) -> float:
    if tracks.shape[0] < 3:
        return 0

    # Rotación = Σ Variación de track entre mensajes adyacentes
    delta = tracks.astype(int).rolling(2,2).apply(lambda x: x.iloc[1]-x.iloc[0])
    delta = delta.apply(lambda x: x if abs(x) < 180 else (x - np.sign(x)*360))
    # Primer elemento es nan
    # track_variation = delta.iloc[1:].sum()
    track_variation = delta.iloc[1:].cumsum()
    # track_variation = track_variation.iloc[track_variation.abs().argmax()]
    track_variation = track_variation.abs().max()

    return track_variation


def detect_rotation_oscillations(data: pd.DataFrame) -> int:
    # https://stackoverflow.com/a/64747209
    if data.shape[0] < 3:
        return 0

    lat1, lat2 = data.latitude.values[:-1], data.latitude.values[1:]
    lon1, lon2 = data.longitude.values[:-1], data.longitude.values[1:]
    dlon = lon2-lon1

    x = np.cos(np.radians(lat2)) * np.sin(np.radians(dlon))
    y = np.cos(np.radians(lat1)) * np.sin(np.radians(lat2)) - np.sin(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.cos(np.radians(dlon))
    angles = np.arctan2(x,y)
    angles = np.degrees(angles) % 360
    angles = np.round(angles, 0)

    # print(f'{angles=}')
    # print(f'{data.true_track.values[1:]=}')
    # print(data.track.values[1:])

    return np.sum(np.abs((angles - data.true_track.values[1:]))>MIN_OSCILLATION)/angles.shape[0]


def detect_rotation_oscillations2(tracks: pd.Series) -> int:
    if tracks.shape[0] == 0:
        return 0

    delta = tracks.astype(int).rolling(2,2).apply(lambda x: x.iloc[1]-x.iloc[0])
    delta = delta.apply(lambda x: x if abs(x) < 180 else (x - np.sign(x)*360))

    delta[delta.abs()<=MIN_OSCILLATION] = 0

    delta = delta.rolling(2,2).apply(lambda x: np.sign(x.iloc[0])!=np.sign(x.iloc[1]))

    return delta.sum()/tracks.shape[0]


def sort_nearest_vector(array: np.array, stop: str = 'undefined') -> np.array:
    '''
    Args:
        array: Array of vector attributes to be sorted
        stop: ID of the last vector in the trajectory. Once this vector is reached,
            unsorted vectors are concatenated with sorted ones without being sorted.
    '''
    if array.shape[0]==0:
        return array
    # Calculate distance from each vector to each other vector
    distances = []
    for x in array:
        lat_index = params.vector_attribute_names.index('latitude')
        lon_index = params.vector_attribute_names.index('longitude')
        distances.append(haversine_np(array[:,lat_index].astype('float'),
                                      array[:,lon_index].astype('float'),
                                      x[lat_index], x[lon_index]))
    distances = np.array(distances)

    # Starting from the first vector, closest vector is recursively identified,
    # excluding those that were already visited
    mins = [0]
    # Mask used to hide visited vectors
    mask = np.zeros(shape=array.shape[0]).astype(bool)
    while len(mins) < array.shape[0]:
        mask[mins[-1]] = True
        current_vector = np.ma.array(distances[mins[-1]], mask=mask)
        mins.append(current_vector.argmin(fill_value=np.inf))
        # If we reach last vector, we stop iterating
        if array[mins[-1],0] == stop:
            break

    # Rearranged vectors
    sorted = array[mins]
    # Unsorted vectors (if any)
    mask = np.ones(array.shape[0]).astype(bool)
    mask[mins] = False
    unsorted = array[mask,:]

    return np.concatenate([sorted,unsorted], axis=0)


def is_resorted(x, acc_list: list, show_log: bool = False) -> bool:
    found, missing = acc_list
    removed, added = [], []
    # rem_f, rem_m = False, False
    reordenado = False
    tipo = ''

    f, m = len(found), len(missing)
    acc = f - m
    if x.ordenInicial in missing:
        tipo += 'a'

        removed.append(f'm{x.ordenInicial}')
        missing.remove(x.ordenInicial)

    elif x.ordenFinal - (f-m) in found:
        tipo += 'b'

        to_remove=[]
        for i in range(x.ordenFinal - (f-m), x.ordenInicial):
            if i not in found:
                break
            to_remove.append(i)
        removed.extend(f'f{i}' for i in to_remove)
        found.difference_update(to_remove)

    if (x.ordenInicial + len(found) == x.ordenFinal + len(missing)):
        tipo += '1'

        reordenado = False

    elif (x.ordenInicial + len(found) < x.ordenFinal + len(missing)):
        tipo += '2'

        reordenado = True
        # added.extend([f'm{x}' for x in range(x.ordenFinal + len(missing), x.ordenInicial + len(found))])
        # missing.update(set(range(x.ordenFinal + len(missing), x.ordenInicial + len(found))))
        added.append(f'm{x.ordenInicial}')
        missing.add(x.ordenInicial )

    elif (x.ordenInicial + len(found) > x.ordenFinal + len(missing)):
        tipo += '3'

        reordenado = True
        added.append(f'f{x.ordenInicial}')
        found.add(x.ordenInicial)

    if show_log:
        print(f'{tipo:<5} ({x.ordenFinal:3}  {x.ordenInicial:3} | {x.ordenFinal - x.ordenInicial:4})  {m:>3}|{f:<3}' + #    {acc:3}
              f'   {x.ordenFinal - x.ordenInicial - (f-m):4}  {"x" if reordenado else " "}  '  +
              f'{"+["+" ".join(added)+"]" if added else "":<7} {" -["+" ".join(removed)+"]" if removed else ""}')
    # if reordenado: print('\t', found, missing)

    return reordenado


def is_resorted_simp(x, found: set, show_log: bool = False) -> bool:
    # WIP: Detección de vectores reordenados asumiendo que los vectores solo se han desordenado
    # hacia adelante (o sea, se recolocan en posiciones anteriores a su posición inicial)
    removed, added = [], []
    reordenado = False

    if x.ordenInicial == x.ordenFinal + len(found):
        reordenado = False
    elif x.ordenInicial > x.ordenFinal + len(found):
        reordenado = False
    elif x.ordenInicial < x.ordenFinal + len(found):
        reordenado = True
        found.add(x.ordenInicial)

    if x.ordenInicial in found:
        found.remove(x.ordenInicial)

    return reordenado


def fix_timestamp(trajectory: Trajectory) -> pd.DataFrame:
    try:
        df = trajectory.vectors.copy()
    except AttributeError:
        print(trajectory.vectors.head())
        print(trajectory.vectors.columns)
        exit()
    
    # acc para usarlo como objeto mutable y evitar el uso de una variable global
    # [found, missing]
    acc = (set(), set())
    # print(f'Tipo  (oFi  oIn | diff)    m|f oF-oI-acc Re Changes')    # Log
    df['reordenado'] = (df[['ordenFinal','ordenInicial']].astype('Int32[pyarrow]')
                                                         .apply(is_resorted, args=[acc], axis=1))
    try:
        df['fixed_timestamp'] = df[~df.reordenado]['timestamp'].astype('Int64[pyarrow]')
    except AttributeError as e:
        print(e)
        print(trajectory.date, trajectory.ifplId)
        print(df.head())
    except KeyError as e:
        print(e)
        print(trajectory.date, trajectory.ifplId)
        print(df.head())
    # print(f'Found:   {acc[0]}')
    # print(f'Missing: {acc[1]}')

    # Usando la distancia recorrida para realizar la interpolación
    cum_sum = np.cumsum(calculate_distance(df[['latitude','longitude','true_track']].values)) # [::-1]
    df['fixed_timestamp'] = (df.set_index(cum_sum)['fixed_timestamp']
                               .interpolate(method='index', limit_direction='forward', limit_area='inside')
                               .interpolate(method='linear', limit_direction='both', limit_area='outside', order=1)
                               .round(0).astype('int64[pyarrow]').to_numpy())
    df['original_timestamp'] = df['timestamp'].copy()
    df['timestamp'] = df['fixed_timestamp'].to_numpy()

    trajectory.vectors = df

    return df

def fix_altitude(trajectory: Trajectory) -> pd.DataFrame:
    df = trajectory.vectors.copy()

    origin_airport = airports[airports.icao == trajectory.aerodromeOfDeparture].iloc[0]
    destination_airport = airports[airports.icao == trajectory.aerodromeOfDestination].iloc[0]

    # Fill ground vectors with airport's altitude
    df.loc[(df.distance_org<TMA_AREA_MIN)&(df.on_ground),'altitude'] = origin_airport.altitude
    df.loc[(df.distance_dst<TMA_AREA_MIN)&(df.on_ground),'altitude'] = destination_airport.altitude

    def calculate_altitudes(data: pd.DataFrame):
        if len(data)<5:
            data['original_altitude'] = data.altitude.copy()
            return data

        data['incorrect_altitude'] = data.altitude.isna()
        data['filtered_altitude'] = data.altitude.interpolate(method='slinear', limit_area='inside')
        # try:
        #     data['filtered_altitude'] = data.altitude.interpolate(method='slinear', limit_area='inside')
        # except ValueError:
        #     print(data.ifplId)
        #     exit()

        # data['filtered_altitude'] = data.altitude.interpolate(method='polynomial', limit_area='inside', order=5)
        data['median_value'] = (data.filtered_altitude
                                    .rolling(ALTITUDE_CHECK_WINDOW_SIZE, min_periods=3, center=True, closed='both')
                                    .median()
                                ).to_numpy()
        data['incorrect_altitude'] = (data.incorrect_altitude | 
                                      (abs(data.filtered_altitude - data.median_value) > DIFF_ALTITUDE_THRESHOLD))
        
        try:
            data.loc[[data.index[0], data.index[-1]], 'incorrect_altitude'] = False
        except IndexError:
            print(f'Error al procesar la altitud en {trajectory.ifplId}: {data.shape}')
            return None
        data['filtered_altitude'] = data[~data.incorrect_altitude].filtered_altitude

        # df['interpolated_altitude'] = df['filtered_altitude'].interpolate(method='linear', limit = 3, limit_area='inside')
        data['interpolated_altitude'] = (data.set_index('timestamp')['filtered_altitude']
                                             .interpolate(method='index', limit = 7, limit_area='inside')
                                             .to_numpy()) # .reset_index(drop=True)
        data['original_altitude'] = data['altitude'].copy()
        # data['altitude'] = data.filtered_altitude.combine_first(data.interpolated_altitude)
        data['altitude'] = data.interpolated_altitude
        data = data.drop(['incorrect_altitude', 'filtered_altitude', 'median_value', 'interpolated_altitude'], axis=1)
        
        return data
    
    latest = 0
    results = []
    # Gaps
    diffs = df.timestamp.iloc[1:].values - df.timestamp.iloc[:-1].values
    gaps = [dict(index=i, size=d) for i, d in enumerate(diffs) if d>60]
    for g in gaps:
        data = df.iloc[latest:g['index']+1].copy()
        latest = g['index']+1

        results.append(calculate_altitudes(data))
    else:
        data = df.iloc[latest:].copy()        
        results.append(calculate_altitudes(data))

    df = pd.concat(results)
    trajectory.vectors = df

    return df


def sort_trajectory(trajectory: Trajectory) -> Trajectory:
    '''
        Ordena las filas de un dataframe de acuerdo a su latitud y longitud. Mantiene el
        índice original (el orden anterior queda registrado en la columna ordenInicial)

        data : Trajectory
            Dataframe con los vectores ordenados por timestamp
    '''
    ### Duplicated position vectors
    # Opensky keeps emitting the latest position for few iterations if no new data have been received.
    # However, the position is not real and should be removed during cleaning. The downside is that the
    # following vectors may contain updated values for other columns, that are consequentially lost.
    # These vectors mess with the sorting algorithms since any path between these vectors have a distance 0.
    # Decision: Keep only the original vector with the duplicated position values.

    ### Metrics
    ts_start = time.time()
    stats = {}
    data = trajectory.vectors.copy()

    stats['initial_num_vectors'] = len(data)
    stats['initial_distance'] = float(calculate_distance(data[['latitude','longitude','true_track']].values).sum())
    stats['dupl_vectors'] = int(data.drop('timestamp', axis=1).duplicated().sum())
    stats['dupl_position_vectors'] = int(data.duplicated(subset=['latitude','longitude']).sum())
    stats['dupl_position_vectors_pos_ts'] = int(data.duplicated(subset=['latitude','longitude','time_position']).sum())
    stats['dupl_position_vectors_gen_ts'] = int(data.duplicated(subset=['latitude','longitude','last_contact']).sum())

    data = data[~data.duplicated(subset=['latitude','longitude'])]
    stats['distance_duplVectors'] = float(calculate_distance(data[['latitude','longitude','true_track']].values).sum())
    data['ordenInicial'] = range(len(data))
    data['ordenInicial'] = data['ordenInicial'].astype('Int32[pyarrow]')

    # Calculate distances to airports
    origin_airport = airports[airports.icao == trajectory.aerodromeOfDeparture].iloc[0]
    destination_airport = airports[airports.icao == trajectory.aerodromeOfDestination].iloc[0]
    data['distance_org'] = haversine_np(data.latitude, 
                                        data.longitude,
                                        origin_airport.latitude, 
                                        origin_airport.longitude
                                       ).astype('Float[pyarrow]')
    data['distance_dst'] = haversine_np(data.latitude, 
                                        data.longitude,
                                        destination_airport.latitude, 
                                        destination_airport.longitude
                                       ).astype('Float[pyarrow]')

    stats['distance_airports'] = haversine_np(origin_airport.latitude, 
                                              origin_airport.longitude, 
                                              destination_airport.latitude, 
                                              destination_airport.longitude)
    
    ### Ground vectors in origin airport - By timestamp
    ground_org = data[(data.distance_org<AIRPORT_AREA) & (data.on_ground)].copy()
    if len(ground_org)>0:
        data = data[~data.index.isin(ground_org.index)]
    ### Ground vectors in destination airport - By timestamp
    ground_dst = data[(data.distance_dst<AIRPORT_AREA) & (data.on_ground)].copy()
    if len(ground_dst)>0:
        data = data[~data.index.isin(ground_dst.index)]
    ### Maneuver segments - NV or by timestamp
    maneuver_org = data[data.distance_org<TMA_AREA_MAX/2].copy()
    maneuver_dst = data[data.distance_dst<TMA_AREA_MAX].copy()
    ### Cruise segment - Distance to destination
    cruise = data[~(data.index.isin(maneuver_org.index)|data.index.isin(maneuver_dst.index))]
    
    ######################### Presort #########################

    # Segmento de salida ordenado por NV
    candidate = pd.DataFrame(sort_nearest_vector(maneuver_org.to_numpy()),  
                            index=maneuver_org.index,
                            columns=maneuver_org.columns
                            ).astype(maneuver_org.dtypes)
    dst = calculate_distance(maneuver_org[['latitude','longitude','true_track']].values).sum()
    dst_candidate = calculate_distance(candidate[['latitude','longitude','true_track']].values).sum()
    if dst_candidate<=dst:
         maneuver_org = candidate

    # Segmento de crucero ordenado por distancia decreciente al aeropuerto de destino
    candidate = cruise.sort_values(by='distance_dst', ascending=False)
    dst = calculate_distance(cruise[['latitude','longitude','true_track']].values).sum()
    dst_candidate = calculate_distance(candidate[['latitude','longitude','true_track']].values).sum()
    if dst_candidate<=dst:
         cruise = candidate

    # Segmento de entrada ordenado por NV
    estimated_rotation = calculate_rotation(maneuver_dst[maneuver_dst.distance_dst.between(TMA_AREA_MIN, TMA_AREA_MAX)].true_track)
    if estimated_rotation<LOOP_ROTATION:
        candidate = pd.DataFrame(sort_nearest_vector(maneuver_dst.to_numpy()),
                                index=maneuver_dst.index,
                                columns=maneuver_dst.columns
                                ).astype(maneuver_dst.dtypes)
        dst = calculate_distance(maneuver_dst[['latitude','longitude','true_track']].values).sum()
        dst_candidate = calculate_distance(candidate[['latitude','longitude','true_track']].values).sum()
        if dst_candidate<=dst:
            maneuver_dst = candidate

    # Calculate segments indices for 2-opt
    initial_ground_segment = ground_org.shape[0]
    end_origin_segment = initial_ground_segment + maneuver_org.shape[0]
    end_cruise_segment = end_origin_segment + cruise.shape[0]
    # Merge flight segments
    data = pd.concat([ground_org, maneuver_org, cruise, maneuver_dst, ground_dst])

    # Cleanup
    del candidate, ground_org, maneuver_org, cruise, maneuver_dst, ground_dst

    stats['distance_presort'] = float(calculate_distance(data[['latitude','longitude','true_track']].values).sum())

    # Oscilaciones en el track
    # TODO: Comprender
    temp = data[data.distance_dst.between(TMA_AREA_MAX, TMA_AREA_MAX+200)]
    stats['oscillation'] = float(detect_rotation_oscillations(temp[['latitude','longitude','true_track']]))
    stats['oscillation_track'] = float(detect_rotation_oscillations2(temp['true_track']))
    stats['oscillation_sample'] = temp.shape[0]

    ### Holding maneuver detection
    # A trajectory following a holding procedure should have an accumulated rotation of >180 degrees
    # in the "maneuver zone" in the TMA. These maneuvers are standardized, so it would be feasible
    # to easily adapt this calculation to each airport.
    temp = data[(data.distance_dst.between(TMA_AREA_MIN, TMA_AREA_MAX)) & (data.altitude>0)].copy()
    track_variation = calculate_rotation(temp.true_track)
    stats['rotation'] = float(track_variation)

    del temp

    # data = data.drop(['distance_org', 'distance_dst'], axis=1)

    #################### TSP sorting ####################
    # Primer tramo
    windows = generate_windows(data.shape[0], window_size_slow,
                               overlap_slow, 0, end_origin_segment)
    data.iloc[:] = sort_windows(data.to_numpy(), windows, angle=False)

    # Segundo tramo
    windows = generate_windows(data.shape[0], window_size_fast,
                               overlap_fast, end_origin_segment, end_cruise_segment)
    data.iloc[:] = sort_windows(data.to_numpy(), windows, angle=False)

    # Tercer tramo
    # Para descartar reordenación del tercer tramo si el resultado es peor que el inicial
    # by_nearest = data.copy()
    by_time = pd.concat([
        data.iloc[:end_cruise_segment],
        data.iloc[end_cruise_segment:].sort_values(by='timestamp')
                                      .set_index(data.iloc[end_cruise_segment:].index)
    ])

    windows = generate_windows(len(data), window_size_slow,
                               overlap_slow, end_cruise_segment-1, len(data))
    if abs(track_variation) > HOLDING_ROTATION:
        stats['loop'] = 'Holding'
        # data.iloc[:] = sort_windows(data.values, windows, angle=False)
        data = (pd.DataFrame(sort_windows(data.to_numpy(), windows, angle=True),
                             columns=data.columns, index=data.index).astype(data.dtypes))
    elif abs(track_variation) > LOOP_ROTATION:
        stats['loop'] = 'Loop'
        # data.iloc[:] = sort_windows(data.values, windows, angle=True)
        data = (pd.DataFrame(sort_windows(data.to_numpy(), windows, angle=True),
                             columns=data.columns, index=data.index).astype(data.dtypes))
    else:
        # data.iloc[:] = sort_windows(data.values, windows, angle=False)
        data = (pd.DataFrame(sort_windows(data.to_numpy(), windows, angle=False),
                             columns=data.columns, index=data.index).astype(data.dtypes))

    final_distance = calculate_distance(data[['latitude','longitude','true_track']].to_numpy()).sum()
    time_distance = calculate_distance(by_time[['latitude','longitude','true_track']].to_numpy()).sum()
    # time_presort = calculate_distance(by_nearest[['latitude','longitude','true_track']].to_numpy()).sum()

    # print(f"{data.ifplId.iloc[0]}     Time: {time_distance:8.2f}     \
    #       Pres: {time_presort:8.2f} ({100*(time_presort-time_distance)/time_distance:5.1f}%)     \
    #       TSP: {final_distance:8.2f} ({100*(final_distance-time_distance)/time_distance:5.1f}%)")


    if time_distance < final_distance:
        # print(f'\n¡No funcionó! {trajectory.ifplId}')
        data = by_time
        stats['final_distance'] = float(time_distance)
        stats['sorted_end_segment_by'] = 'timestamp'
    else:
        stats['final_distance'] = final_distance
        stats['sorted_end_segment_by'] = '2opt'

    data['ordenFinal'] = range(data.shape[0])
    trajectory.vectors = data

    stats['final_num_vectors'] = len(data)
    stats['process_time'] = time.time() - ts_start

    folder = SORT_TRAJECTORIES_METRICS_PATH
    if not folder.exists():
        folder.mkdir(parents=True)
    with open(folder / f'sortTray.{trajectory.date}.{trajectory.ifplId}.json', 'w+', encoding='utf8') as file:
        json.dump(stats, file, indent=2)

    return trajectory


def sort_trajectories(date: str):
    tray_ids = (NM_TRAJECTORIES_RAW_PATH / f'flightDate={date}').glob('*.json')
    tray_ids = [str(x).split('.')[1] for x in tray_ids]
    trays = [Trajectory(x, date) for x in tray_ids]

    with concurrent.futures.ProcessPoolExecutor(max_workers=7) as executor:
        result = list(tqdm(executor.map(sort_trajectory, trays), total=len(trays), ncols=125, leave=False))

    for t in tqdm(result, ncols=125, disable=False, desc=date, leave=False):
        fix_timestamp(t)
        fix_altitude(t)
        # detect_outliers(t)
        t.vectors.drop(['distance_org', 'distance_dst'], axis=1, inplace=True)
        t.trajectory_status = 'L3_sorted'
        t.save()
    result = [t.vectors for t in result]

    ### No paralelizado
    # result = []
    # for trajectoryId in tqdm(tray_ids):
    #     trajectory = Trajectory(trajectoryId, date)
    #     sort_trajectory(trajectory)
    #     result.append(trajectory.vectors)
    #     trajectory.save()

    result = pd.concat(result)

    folder = NM_TRAJECTORIES_PATH
    if len(result)>0:
        path = folder / f'tray.{date}.parquet'
        result.to_parquet(path, index=False,)

# TEST
# if __name__ == '__main__':
#     tray = Trajectory('AT02982288', '2023-07-14')
#     tray.save_single_tray = True
#     sort_trajectory(tray)
#     fix_timestamp(tray)
#     fix_altitude(tray)
#     tray.vectors.drop(['distance_org', 'distance_dst'], axis=1, inplace=True)
#     tray.save()
#     metrics.calculate_metrics_trajectories('2022-07-14', 'clean')
#     exit()

if __name__ == '__main__':
    date_start, date_end = '2023-07-14','2023-07-16'

    dates = utils.get_dates_between(date_start, date_end)
    dates = [x.strftime('%Y-%m-%d') for x in dates]

    for date in tqdm(dates, ncols=125, leave=True):
        sort_trajectories(date)
        metrics.calculate_metrics_trajectories(date, 'clean')

### TEST: Borrar luego
# tray = Trajectory('AT03135753', '2022-07-11')
# tray = Trajectory('AT03110917', '2022-07-11')
# tray.save_single_tray = True
# sort_trajectory(tray)
# fix_timestamp(tray)
# fix_altitude(tray)
# tray.vectors.drop(['distance_org', 'distance_dst'], axis=1, inplace=True)
# tray.save()
# metrics.calculate_metrics_trajectories('2022-07-11', 'clean')
# exit()

def test_is_resorted(ordenInicial, ordenFinal):
    idx = 0
    counter = 0
    M,F = set(), set()
    res = []
    print('i j | m f     M F')
    while idx<len(ordenFinal):
        print(ordenInicial[idx],ordenFinal[idx],'|',len(M),len(F),'   ', M, F, end=' ')
        if ordenInicial[idx] == ordenFinal[idx] + len(M) - len(F):
            print('--> Advance')
            idx+=1
            res.append(False)
        else:
            m, f = len(M), len(F)

            if ordenInicial[idx] not in M:
                M.add(ordenFinal[idx])
                # M=M.union(set(range(ordenFinal[idx]+m-f, ordenInicial[idx])))
                F.add(ordenInicial[idx])
                # print('--> aAdvance')
                # idx+=1
            elif ordenInicial[idx] in M:
                M.remove(ordenInicial[idx])
                # print('--> bAdvance')
                idx+=1
            
            if ordenFinal[idx] in F:
                F.remove(ordenFinal[idx])
            elif ordenFinal[idx] not in F:
                pass
            print()
        counter += 1
        if counter>15: break
    print(res)

def test():
    x1 = [(x,y) for x,y in zip(range(1,11),[1,6,5,2,4,3,8,9,7,10])]
    x1 = pd.DataFrame(x1, columns=('ordenFinal', 'ordenInicial'))
    acc = (set(), set())
    print(f'Tipo  (oFi  oIn | diff)    m|f oF-oI-acc Re Changes')
    x1.apply(is_resorted, args=[acc,True], axis=1)

    x2 = [(x,y) for x,y in zip(range(1,11),[1,2,4,6,3,5,8,9,7,10])]
    x2 = pd.DataFrame(x2, columns=('ordenFinal', 'ordenInicial'))
    acc = (set(), set())
    print(f'Tipo  (oFi  oIn | diff)    m|f oF-oI-acc Re Changes')
    x2.apply(is_resorted, args=[acc,True], axis=1)

    # test_is_resorted([1,6,5,2,4,3,8,9,7,10],range(1,11))
    # print()
    # test_is_resorted([1,2,4,6,3,5,8,9,7,10],range(1,11))


def fix_altitude_bckp(trajectory: Trajectory) -> pd.DataFrame:

    ## TODO: Excluir vectores en tierra para que su altitud nula no se propague a vectores en aire

    df = trajectory.vectors.copy()

    org_ground = df[(df.distance_org<TMA_AREA_MAX) & df.on_ground]
    dst_ground = df[(df.distance_dst<TMA_AREA_MAX) & df.on_ground]
    if len(dst_ground)>0:
        df = df.iloc[len(org_ground):-len(dst_ground)-1]
    else:
        df = df.iloc[len(org_ground):]

    df['incorrect_altitude'] = False # df.altitude.isna()
    df['filtered_altitude'] = df.altitude.astype('Float[pyarrow]')

    num_filters = 2
    for i in range(num_filters):
        if i == 0:
            # Primer filtro más grueso
            altitude_threshold = 2000
            win_size = 15
        else:
            # Segundo filtro más fino
            altitude_threshold = 500
            win_size = 5

        df['median_value'] = (df.filtered_altitude # .dropna()
                                .rolling(win_size, min_periods=3, center=True, closed='both')
                                .median()).values
        df['incorrect_altitude'] = ((abs(df.filtered_altitude-df.median_value) > altitude_threshold) | df['incorrect_altitude']).copy()
        try:
            df.loc[[df.index[0], df.index[-1]], 'incorrect_altitude'] = False # loc o iloc??
        except IndexError:
            print(f'Error al procesar la altitud en {trajectory.ifplId}: {df.shape}')
            return None
        df['filtered_altitude']  = df[~df.incorrect_altitude].filtered_altitude

    # df['interpolated_altitude'] = df['filtered_altitude'].interpolate(method='linear', limit = 3, limit_area='inside')
    df['interpolated_altitude'] = (df.set_index('timestamp')['filtered_altitude']
                                     .interpolate(method='index', limit = 5) # , limit_area='inside'
                                     .values) # .reset_index(drop=True)
    df['altitude'] = df.filtered_altitude.combine_first(df.interpolated_altitude)
    trajectory.vectors = pd.concat(x for x in [org_ground, df, dst_ground] if len(x)>0)

    return df

def is_resorted_bckp(x, acc_list: list, show_log: bool = True) -> bool:
    found, missing = acc_list
    removed, added = [], []
    # rem_f, rem_m = False, False
    reordenado = False
    tipo = ''

    f, m = len(found), len(missing)
    acc = f - m
    if x.ordenInicial  in missing:
        tipo += 'a'

        removed.append(f'm{x.ordenInicial}')
        missing.remove(x.ordenInicial)

    elif x.ordenFinal - acc in found:
        tipo += 'b'

        to_remove=[]
        for i in range(x.ordenFinal + acc, x.ordenInicial):
            if i not in found:
                break
            to_remove.append(i)
        removed.extend(f'f{i}' for i in to_remove)
        found.difference_update(to_remove)

    if (x.ordenInicial + len(found) == x.ordenFinal + len(missing)):
        tipo += '1'

        reordenado = False

    elif (x.ordenInicial + len(found) < x.ordenFinal + len(missing)):
        tipo += '2'

        reordenado = True
        # added.extend([f'm{x}' for x in range(x.ordenFinal + len(missing), x.ordenInicial + len(found))])
        # missing.update(set(range(x.ordenFinal + len(missing), x.ordenInicial + len(found))))
        added.append(f'm{x.ordenInicial}')
        missing.add(x.ordenInicial )

    elif (x.ordenInicial + len(found) > x.ordenFinal + len(missing)):
        tipo += '3'

        reordenado = True
        added.append(f'f{x.ordenInicial}')
        found.add(x.ordenInicial)

    if show_log:
        print(f'{tipo:<5} ({x.ordenFinal:3}  {x.ordenInicial:3} | {x.ordenFinal - x.ordenInicial:4})  {m:>3}|{f:<3}' + #    {acc:3}
              f'   {x.ordenFinal - x.ordenInicial - acc:4}  {"x" if reordenado else " "}  '  +
              f'{"+["+" ".join(added)+"]" if added else "":<7} {" -["+" ".join(removed)+"]" if removed else ""}')
    # if reordenado: print('\t', found, missing)

    return reordenado
