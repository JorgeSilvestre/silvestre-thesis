import datetime
import json
import re

import pandas as pd
import plotly.express as px
import streamlit as st

from utils import haversine_np
from paths import *

output_config = {
  'toImageButtonOptions': {
    'format': 'svg', # one of png, svg, jpeg, webp
    'filename': 'newplot',
    'height': 2*450,
    'width': 2*650,
    'scale':1 # Multiply title/legend/axis/canvas sizes by this factor
  },
  'scrollZoom': True
}

st.set_page_config(layout="wide", page_title='Dashboard L3 Trajectories')

dates = NM_PARQUET_FLIGHTS_PATH.glob('*')
dates = [re.search(r'nm.flights.([\d-]{10}).parquet', str(x))[1] for x in dates]

# @st.cache_data
def load_data(date):
    data_flight = pd.read_parquet(NM_PARQUET_FLIGHTS_PATH / f'nm.flights.{date.strftime(format="%Y-%m-%d")}.parquet', 
                                  engine='pyarrow', dtype_backend='pyarrow')
    raw_trajectories = pd.read_parquet(NM_TRAJECTORIES_RAW_PATH / f'tray.{date.strftime(format="%Y-%m-%d")}.parquet', 
                                  engine='pyarrow', dtype_backend='pyarrow')
    clean_trajectories = pd.read_parquet(NM_TRAJECTORIES_PATH / f'tray.{date.strftime(format="%Y-%m-%d")}.parquet', 
                                  engine='pyarrow', dtype_backend='pyarrow')
    
    raw_trajectories['time_position'] = pd.to_datetime(raw_trajectories.time_position, unit='s')
    raw_trajectories['last_contact'] = pd.to_datetime(raw_trajectories.last_contact, unit='s')
    raw_trajectories['timestamp'] = pd.to_datetime(raw_trajectories.timestamp, unit='s')
    raw_trajectories['vector_timestamp'] = pd.to_datetime(raw_trajectories.timestamp, unit='s')

    clean_trajectories['time_position'] = pd.to_datetime(clean_trajectories.time_position, unit='s')
    clean_trajectories['last_contact'] = pd.to_datetime(clean_trajectories.last_contact, unit='s')
    clean_trajectories['timestamp'] = pd.to_datetime(clean_trajectories.timestamp, unit='s')
    clean_trajectories['vector_timestamp'] = pd.to_datetime(clean_trajectories.timestamp, unit='s')

    data_flight = data_flight[data_flight.ifplId.isin(clean_trajectories.ifplId.drop_duplicates())]

    data_flight['estimatedOffBlockTime'] = pd.to_datetime(data_flight.estimatedOffBlockTime, unit='s')
    data_flight['actualTakeOffTime'] = pd.to_datetime(data_flight.actualTakeOffTime, unit='s')
    data_flight['actualTimeOfArrival'] = pd.to_datetime(data_flight.actualTimeOfArrival, unit='s')
    data_flight['estimatedTakeOffTime'] = pd.to_datetime(data_flight.estimatedTakeOffTime, unit='s')
    data_flight['estimatedTimeOfArrival'] = pd.to_datetime(data_flight.estimatedTimeOfArrival, unit='s')
    data_flight['calculatedTakeOffTime'] = pd.to_datetime(data_flight.calculatedTakeOffTime, unit='s')
    data_flight['calculatedTimeOfArrival'] = pd.to_datetime(data_flight.calculatedTimeOfArrival, unit='s')

    return data_flight, raw_trajectories, clean_trajectories

columns_content = st.columns([1,2,2])
with columns_content[0]:
    columns_filter = st.columns([1,1])
    with columns_filter[0]:
        date = st.date_input(
            label='Date',
            value=datetime.datetime.strptime(dates[0], '%Y-%m-%d'),
            min_value=datetime.datetime.strptime(dates[0], '%Y-%m-%d'),
            max_value=datetime.datetime.strptime(dates[-1], '%Y-%m-%d')
        )

data_flight, raw_trajectories, clean_trajectories = load_data(date)
flight_list = data_flight.ifplId.drop_duplicates().tolist()

with columns_filter[1]:
    flight_index = st.number_input(label='Flight index', min_value=0, max_value=len(flight_list)-1)
with columns_content[0]:
    flight_id = st.selectbox(label='Flight id', index=flight_index, options=flight_list)

margins_map = dict(l=5, r=5, b=5, t=40, pad=0 )
margins_alt = dict(l=5, r=5, b=5, t=40, pad=0 )

### Left column
with columns_content[0]:
    flight_infobox = st.expander(label='**🔎 Información del vuelo**')
    with flight_infobox:
        st.dataframe(data_flight[data_flight.ifplId == flight_id].astype(str).transpose(), 
                    use_container_width=True, height=420,
                    column_config = {'_index' : st.column_config.Column(width="small",)},)
    metrics_infobox = st.expander(label='**📈 Métricas de reordenación**')
    with metrics_infobox:
        with open(SORT_TRAJECTORIES_METRICS_PATH / f'sortTray.{date}.{flight_id}.json', 'r', encoding='utf8') as file:
            sort_metadata = json.load(file)
        st.json(sort_metadata)
        
    with st.expander('**⚙️ Configure diagrams**', expanded=True):
        dims_x_labels = ['Índices', 'Timestamps']

        dimensiones_mapa = st.radio('Mapa', ['Índices', 'Timestamps', ], horizontal=False,)
        dimensiones_x = st.radio('Eje X', dims_x_labels, horizontal=False,)
        dimensiones_y = st.radio('Eje Y', ['altitude', 'latitude', 'longitude' ], horizontal=False,)
        color_coded = st.radio('Colorear vectores', ['Sí', 'No' ], horizontal=True,)

raw_tray_data = raw_trajectories[raw_trajectories.ifplId == flight_id].copy()
raw_tray_data['altitude'] = raw_tray_data.altitude.fillna(-321)
clean_tray_data = clean_trajectories[clean_trajectories.ifplId == flight_id].copy()
clean_tray_data['altitude'] = clean_tray_data.altitude.fillna(-321)

with columns_content[1]:
    raw_map = px.scatter_mapbox(
        raw_tray_data, lat='latitude', lon='longitude',
        color=raw_tray_data.index if dimensiones_mapa=='Índices' else raw_tray_data.timestamp.astype('int64')//10**9,
        color_continuous_scale='balance',
        mapbox_style='open-street-map', zoom=3.5,
        title='Initial order', height=550,  
        hover_data={'timestamp':True, 
                    'altitude':':.2f',
                    'on_ground':True,},
    )
    raw_map.update_layout(
        margin=margins_map, 
        coloraxis=dict(colorbar=dict(orientation='h', y=-0.15)), 
    ) 
    st.plotly_chart(raw_map, use_container_width=True, theme=None, config = output_config)

    dims_x_values = [range(len(raw_tray_data)), 'timestamp']
    raw_altitude = px.scatter(
        data_frame=raw_tray_data, y=dimensiones_y,
        x=dims_x_values[dims_x_labels.index(dimensiones_x)],                       
        # color='reused_position' if color_coded=='Sí' else None,
        color_discrete_map={True:'red'},
        title=f'Profile of {dimensiones_y}', height=400,
        # range_y=(0,15000) if dimensiones_y == 'altitude' else None, 
    )
    raw_altitude.update_layout(
        xaxis_title=None, yaxis_title=None, margin=margins_alt,
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99)
    )
    st.plotly_chart(raw_altitude, use_container_width=True, config=output_config)

    with open(NM_TRAYS_METRICS_L2_PATH / f'tray.{date}.{flight_id}.json', 'r', encoding='utf8') as file:
        raw_tray_metadata = json.load(file)
    st.write('**Métricas originales**')
    with st.container(height=400, border=False):
        st.json(raw_tray_metadata)    

with columns_content[2]:
    clean_map= px.scatter_mapbox(
        clean_tray_data, lat='latitude', lon='longitude',
        color='ordenFinal' if dimensiones_mapa=='Índices' else clean_tray_data.timestamp.astype('int64')//10**9,
        color_continuous_scale='balance', #Bluered
        mapbox_style='open-street-map', zoom=3.5,
        title='Final order', height=550, 
        hover_data={'timestamp':True, 
                    'altitude':':.2f', 
                    'on_ground':True, 
                    'ordenInicial':True,
                    'ordenFinal':True,}, 
    )
    clean_map.update_layout(
        margin=margins_map, 
        coloraxis=dict(colorbar=dict(orientation='h', y=-0.15)), 
    )
    st.plotly_chart(clean_map, use_container_width=True, theme=None, config = output_config)

    dims_x_values = ['ordenFinal', 'timestamp']
    clean_altitude = px.scatter(
        data_frame=clean_tray_data, y=dimensiones_y,
        x=dims_x_values[dims_x_labels.index(dimensiones_x)],
        color='reordenado' if color_coded=='Sí' else None,
        color_discrete_map={True:'red'},
        title=f'Profile of {dimensiones_y}',  height=400,
        hover_data={'ordenInicial':True,'ordenFinal':True, 'timestamp':True, 'altitude':':.2f', 'on_ground':True, }, 
        # range_y=(0,15000) if dimensiones_y == 'altitude' else None, 
    )
    clean_altitude.update_layout(
        xaxis_title=None, yaxis_title=None, margin=margins_alt,
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99)
    )
    st.plotly_chart(clean_altitude, use_container_width=True, config=output_config)

    with open(NM_TRAYS_METRICS_L3_PATH / f'tray.{date}.{flight_id}.json', 'r', encoding='utf8') as file:
        tray_metadata = json.load(file)
    st.write('**Métricas tras reordenación**')
    with st.container(height=400, border=False):
        st.json(tray_metadata)


if __name__ == '__main__':
    pass