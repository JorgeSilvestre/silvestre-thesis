import json
# from dataclasses import dataclass
from datetime import datetime

import pandas as pd

import utils
from paths import *

# TODO Reflejar de alguna forma el estado de la trayectoria o las transformaciones aplicadas

# airports_data = pd.read_csv(AIRPORTS_PATH, sep = ',', usecols=['id','lat','lon'])

# @dataclass
class Trajectory():
    """Trajectory class
    Attributes:
        ID : str
        callsign : str
        icao24 : str
        originAirport : str
        destinationAirport : str
        date : datetime
        plannedDeparture : timestamp
        plannedArrival : timestamp
        actualDeparture : timestamp
        actualArrival : timestamp
        airline : str
        dataSource : str
        vectors : pd.DataFrame
    Calculated attributes:
        delayArrival = actualDeparture - plannedDeparture
        delayDeparture = actualArrival - plannedArrival
        initialDistanceOrigin = vectors[0][position] - originAirport[position]
        finalDistanceDestination = vectors[-1][position] - destinationAirport[position]
    Methods:
        detect_outliers
        clean_outliers
        sort_trajectory
    """
    ifplId : str
    callsign : str
    icao24 : str
    aerodromeOfDeparture : str
    aerodromeOfDestination : str
    date : datetime.date
    plannedDeparture : datetime.timestamp
    plannedArrival : datetime.timestamp
    actualDeparture : datetime.timestamp
    actualArrival : datetime.timestamp
    ts_start : datetime.timestamp
    ts_end : datetime.timestamp
    airline : str
    data_source_surveillance : str
    data_source_flights : str
    vectors : pd.DataFrame
    # raw_vectors: pd.DataFrame = vectors.copy()


    def __init__(self, trajectoryId, date, state='raw'):
        if state == 'raw':
            folder = NM_TRAJECTORIES_RAW_PATH 
        elif state == 'clean':
            folder = NM_TRAJECTORIES_PATH

        self.vectors = pd.read_parquet(folder / f'flightDate={date}' / f'tray.{date}.parquet', 
                                    engine='pyarrow', dtype_backend='pyarrow',
                                    filters=[('ifplId', '==', trajectoryId)])
        with open(folder / f'tray.{trajectoryId}.json', 'r', encoding='utf8') as file:
            metadata = json.load(file)
        for k, v in metadata.items():
            setattr(self, k, v)
        self.state = state
        
        self.save_single_tray = False
        
        # self.ifplId = metadata['ifplId']
        # self.callsign = metadata['callsign']
        # self.icao24 = metadata['icao24']
        # self.aerodromeOfDeparture = metadata['aerodromeOfDeparture']
        # self.aerodromeOfDestination = metadata['aerodromeOfDestination']
        # self.plannedDeparture = metadata['estimatedTakeOffTime']
        # self.plannedArrival = metadata['estimatedTimeOfArrival']
        # self.actualDeparture = metadata['actualTakeOffTime']
        # self.actualArrival = metadata['actualTimeOfArrival']
        # self.airline = metadata['airline']
        # self.data_source_surveillance = metadata['data_source_surveillance']
        # self.data_source_flights = metadata['data_source_flights']
        # self.ts_start = metadata['ts_start']
        # self.ts_end = metadata['ts_end']
        # self.trajectory_status = metadata['trajectory_status']
    
    # def delayArrival(self) -> int:
    #     return self.actualDeparture - self.plannedDeparture
    # def delayDeparture(self) -> int:
    #     return self.actualArrival - self.plannedArrival
    # def initialDistanceOrigin(self) -> int:
    #     airport_coord = airports_data[airports_data.id == self.originAirport]
    #     return haversine_np(self.vectors.iloc[0].latitude, self.vectors.iloc[0].longitude,
    #                         airport_coord.lat, airport_coord.lon)
    # def finalDistanceDestination(self) -> int:
    #     airport_coord = airports_data[airports_data.id == self.destinationAirport]
    #     return haversine_np(self.vectors.iloc[0].latitude, self.vectors.iloc[0].longitude,
    #                         airport_coord.lat, airport_coord.lon)

    # def traveled_distance(self, status:Literal['raw','current']) -> float:
    #     """
    #     Args:
    #         status: Whether to calculate current (improved) traveled distance 
    #             vs. raw traveled distance
    #     """
    #     data = self.vectors if status == 'current' else self.raw_vectors
    #     return haversine_np(data.iloc[0:-1].latitude, data.iloc[0:-1].longitude,
    #                         data.iloc[1:].latitude, data.iloc[1:].longitude)

    def update():
        '''Update values of attributes based on current vectors.'''
        # TODO
        
    def save(self):
        folder = NM_TRAJECTORIES_RAW_PATH / f'flightDate={self.date}'
        with open(folder / f'tray.{self.ifplId}.json', 'r', encoding='utf8') as file:
            old_metadata = json.load(file)
        metadata = {}
        for k in old_metadata.keys():
            metadata[k] = getattr(self, k)
        
        folder = NM_TRAJECTORIES_PATH / f'flightDate={self.date}'
        if not folder.exists(): folder.mkdir(parents=True)
        with open(folder / f'tray.{self.ifplId}.json', 'w+', encoding='utf8') as file:
            json.dump(metadata, file, default=utils.custom_json_encoder)

        if self.save_single_tray:
            self.vectors.to_parquet(NM_TRAJECTORIES_PATH / f'tray.{self.date}.parquet', engine='pyarrow')