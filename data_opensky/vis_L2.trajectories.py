import streamlit as st
from paths import *
import datetime
import re
import json
import pandas as pd
import plotly.express as px
import metrics

st.set_page_config(layout="wide", page_title='Dashboard L2 Trajectories')

dates = NM_PARQUET_FLIGHTS_PATH.glob('*')
dates = [re.search(r'nm.flights.([\d-]{10}).parquet', str(x))[1] for x in dates]

# @st.cache_data
def load_data(date):
    data_flight = pd.read_parquet(NM_PARQUET_FLIGHTS_PATH / f'nm.flights.{date.strftime(format="%Y-%m-%d")}.parquet', 
                                  engine='pyarrow', dtype_backend='pyarrow')
    
    data_vectors = pd.read_parquet(NM_TRAJECTORIES_RAW_PATH / f'tray.{date.strftime(format="%Y-%m-%d")}.parquet', 
                                  engine='pyarrow', dtype_backend='pyarrow')
    data_vectors['time_position'] = pd.to_datetime(data_vectors.time_position, unit='s')
    data_vectors['last_contact'] = pd.to_datetime(data_vectors.last_contact, unit='s')
    data_vectors['timestamp'] = pd.to_datetime(data_vectors.timestamp, unit='s')
    data_vectors['vector_timestamp'] = pd.to_datetime(data_vectors.timestamp, unit='s')

    data_flight = data_flight[data_flight.ifplId.isin(data_vectors.ifplId.drop_duplicates())]

    data_flight['estimatedOffBlockTime'] = pd.to_datetime(data_flight.estimatedOffBlockTime, unit='s')
    data_flight['actualTakeOffTime'] = pd.to_datetime(data_flight.actualTakeOffTime, unit='s')
    data_flight['actualTimeOfArrival'] = pd.to_datetime(data_flight.actualTimeOfArrival, unit='s')
    data_flight['estimatedTakeOffTime'] = pd.to_datetime(data_flight.estimatedTakeOffTime, unit='s')
    data_flight['estimatedTimeOfArrival'] = pd.to_datetime(data_flight.estimatedTimeOfArrival, unit='s')
    data_flight['calculatedTakeOffTime'] = pd.to_datetime(data_flight.calculatedTakeOffTime, unit='s')
    data_flight['calculatedTimeOfArrival'] = pd.to_datetime(data_flight.calculatedTimeOfArrival, unit='s')
    
    return data_flight, data_vectors

columns_content = st.columns([1,4])
with columns_content[0]:
    columns_filter = st.columns([1,1])
    with columns_filter[0]:
        date = st.date_input(
            label='Date',
            value=datetime.datetime.strptime(dates[0], '%Y-%m-%d'),
            min_value=datetime.datetime.strptime(dates[0], '%Y-%m-%d'),
            max_value=datetime.datetime.strptime(dates[-1], '%Y-%m-%d')
        )

data_flight, data_vectors = load_data(date)
flight_list = data_flight.ifplId.drop_duplicates().tolist()

with columns_filter[1]:
    flight_index = st.number_input(label='Flight index', min_value=0, max_value=len(flight_list)-1)
with columns_content[0]:
    flight_id = st.selectbox(label='Flight id', index=flight_index, options=flight_list)

### Left column
with columns_content[0]:
    flight_infobox = st.expander(label='**🔎 Información del vuelo**')
    with flight_infobox:
        st.dataframe(data_flight[data_flight.ifplId == flight_id].transpose().astype(str), 
                     use_container_width=True, height=420, 
                     column_config = {'_index' : st.column_config.Column(width="small",)},
    )
    metrics_infobox = st.expander(label='**📈 Métricas de reordenación**')
    with metrics_infobox:
        with open(NM_TRAYS_METRICS_L2_PATH / f'tray.{date}.{flight_id}.json', 'r', encoding='utf8') as file:
            tray_metadata = json.load(file)
        with st.container(height=400, border=False):
            st.json(tray_metadata )

tray_data = data_vectors[data_vectors.ifplId == flight_id].copy()
# tray_data['category'] = '#16ba00'
# tray_data['diffs'] = [0]+(tray_data.timestamp.iloc[1:].values-tray_data.timestamp.iloc[:-1].values).tolist()
# tray_data.loc[tray_data.diffs>metrics.THRESHOLD_GAP_TIME,'category'] = '#ff1100'
# tray_data.loc[tray_data.diffs.between(metrics.THRESHOLD_CONTINUITY, metrics.THRESHOLD_GAP_TIME),'category'] = '#e86c00'

with columns_content[1]:
    columns_graphs = st.columns([1,1])

    with columns_graphs[0]:
        raw_map = px.scatter_mapbox(
            tray_data, lat='latitude', lon='longitude',
            mapbox_style='carto-positron', zoom=3.5,
            height=450,  
            hover_data={'timestamp':True, 
                        'altitude':':.2f',
                        'on_ground':True}, 
        )
        raw_map.update_layout(margin=dict(l=0, r=0, b=0, t=0)) 
        st.plotly_chart(raw_map, use_container_width=True)
    
    with columns_graphs[1]:
        fig = px.scatter_3d(tray_data, x='latitude', y='longitude', z='altitude', height=450,)
        fig.update_layout(margin=dict(l=0, r=0, b=0, t=0)) 
        fig.update_traces( marker=dict(size=4))
        st.plotly_chart(fig, use_container_width=True)
    
    st.write('**Vectores**')
    st.dataframe(tray_data, height=420, )

    # st.write('**Vuelos**')
    # st.dataframe(data_flight, height=420)

if __name__ == '__main__':
    pass