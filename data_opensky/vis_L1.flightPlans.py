import streamlit as st
from paths import *
import datetime
import re
import pandas as pd
import plotly.express as px

st.set_page_config(layout="wide", page_title='Dashboard L1 Flight Plans')

color_map = dict(
    FILED='#0563fa',
    FILED_SLOT_ALLOCATED='#4885e8',
    FILED_SLOT_ISSUED='#86aceb',
    TATC_ACTIVATED='#076600',
    ATC_ACTIVATED='#13d904',
    TERMINATED='#d90404',
)

dates = NM_PARQUET_FLIGHTS_PATH.glob('*')
dates = [re.search(r'nm.flights.([\d-]{10}).parquet', str(x))[1] for x in dates]

# @st.cache_data
def load_data(date):
    data_fp = pd.read_parquet(NM_PARQUET_FPLAN_PATH / f'nm.fplan.{date.strftime(format="%Y-%m-%d")}.parquet', 
                              engine='pyarrow', dtype_backend='pyarrow')
    data_fd = pd.read_parquet(NM_PARQUET_FDATA_PATH / f'nm.fdata.{date.strftime(format="%Y-%m-%d")}.parquet', 
                              engine='pyarrow', dtype_backend='pyarrow').sort_values('flightDataVersionNr')
    data_flight = pd.read_parquet(NM_PARQUET_FLIGHTS_PATH / f'nm.flights.{date.strftime(format="%Y-%m-%d")}.parquet', 
                                  engine='pyarrow', dtype_backend='pyarrow')

    data_fp['timestamp'] = pd.to_datetime(data_fp.timestamp, unit='s')
    data_fp['estimatedOffBlockTime'] = pd.to_datetime(data_fp.estimatedOffBlockTime, unit='s')

    data_fd['timestamp'] = pd.to_datetime(data_fd.timestamp, unit='s')
    data_fd['estimatedOffBlockTime'] = pd.to_datetime(data_fd.estimatedOffBlockTime, unit='s')
    data_fd['actualOffBlockTime'] = pd.to_datetime(data_fd.actualOffBlockTime, unit='s')
    data_fd['actualTakeOffTime'] = pd.to_datetime(data_fd.actualTakeOffTime, unit='s')
    data_fd['actualTimeOfArrival'] = pd.to_datetime(data_fd.actualTimeOfArrival, unit='s')
    data_fd['estimatedTakeOffTime'] = pd.to_datetime(data_fd.estimatedTakeOffTime, unit='s')
    data_fd['estimatedTimeOfArrival'] = pd.to_datetime(data_fd.estimatedTimeOfArrival, unit='s')
    data_fd['calculatedTakeOffTime'] = pd.to_datetime(data_fd.calculatedTakeOffTime, unit='s')
    data_fd['calculatedTimeOfArrival'] = pd.to_datetime(data_fd.calculatedTimeOfArrival, unit='s')

    data_flight['estimatedOffBlockTime'] = pd.to_datetime(data_flight.estimatedOffBlockTime, unit='s')
    data_flight['actualTakeOffTime'] = pd.to_datetime(data_flight.actualTakeOffTime, unit='s')
    data_flight['actualTimeOfArrival'] = pd.to_datetime(data_flight.actualTimeOfArrival, unit='s')
    data_flight['estimatedTakeOffTime'] = pd.to_datetime(data_flight.estimatedTakeOffTime, unit='s')
    data_flight['estimatedTimeOfArrival'] = pd.to_datetime(data_flight.estimatedTimeOfArrival, unit='s')
    data_flight['calculatedTakeOffTime'] = pd.to_datetime(data_flight.calculatedTakeOffTime, unit='s')
    data_flight['calculatedTimeOfArrival'] = pd.to_datetime(data_flight.calculatedTimeOfArrival, unit='s')

    return data_fp, data_fd, data_flight

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

data_fp, data_fd, data_flight = load_data(date)
flight_list = data_flight.ifplId.drop_duplicates().tolist()

with columns_filter[1]:
    flight_index = st.number_input(label='Flight index', min_value=0, max_value=len(flight_list)-1)
with columns_content[0]:
    flight_id = st.selectbox(label='Flight id', index=flight_index, options=flight_list)

with columns_content[0]:
    flight_infobox = st.expander(label='**🔎 Información del vuelo**')
    with flight_infobox:
        st.dataframe(data_flight[data_flight.ifplId == flight_id].astype(str).transpose(), 
                     use_container_width=True, height=420)
        
    temp_col = len(data_fp[data_fp.ifplId == flight_id]) * ['FPLAN'] + len(data_fd[data_fd.ifplId == flight_id]) * ['FDATA']
    emission_data = pd.concat([
        data_fp[data_fp.ifplId == flight_id][['timestamp']],
        data_fd[data_fd.ifplId == flight_id][['timestamp','flightState']],
    ]).fillna('FPLAN')
    emission_data.insert(2, 'message_type', temp_col)
    emission_data = emission_data.sort_values('timestamp').reset_index(drop=True)
    fig = px.scatter(
        emission_data, y='timestamp', 
        color='flightState', 
        color_discrete_map = color_map,
        symbol='message_type', 
        symbol_sequence=['diamond-open-dot', 'circle'],
        height=325,)
    fig.update_traces(marker={'size': 6})
    fig.update_layout(
        legend=dict(yanchor="bottom", y=0.03, xanchor="right", x=0.99,), #, orientation="h",
        margin=dict(l=5, r=5, t=5, b=5),
        yaxis_title=None,
    ) 
    fig.add_hline(y=data_flight[data_flight.ifplId == flight_id].actualTakeOffTime.iloc[0], 
                  line_width=1, line_dash="solid", line_color="green")
    fig.add_hline(y=data_flight[data_flight.ifplId == flight_id].actualTimeOfArrival.iloc[0], 
                  line_width=1, line_dash="solid", line_color="red")
    fig.add_annotation(x=0, y=data_flight[data_flight.ifplId == flight_id].actualTakeOffTime.iloc[0],
                       yshift=10, showarrow=False, text="Takeoff", xref='paper')
    fig.add_annotation(x=0, y=data_flight[data_flight.ifplId == flight_id].actualTimeOfArrival.iloc[0],
                       yshift=10, showarrow=False, text="Arrival", xref='paper')
    st.plotly_chart(fig, use_container_width=True)

    timestamps_evolution = px.line(
        data_fd[data_fd.ifplId == flight_list[flight_index]].rename({
           'estimatedOffBlockTime':'estOffBlock', 
           'estimatedTakeOffTime':'estTakeOff', 
           'actualTakeOffTime':'actualTakeOff', 
           'estimatedTimeOfArrival':'estArrival',
           'actualTimeOfArrival':'actualArrival', 
        }, axis=1), 
        y=['estOffBlock', 'estTakeOff', 'actualTakeOff', 
           'estArrival', 'actualArrival'],
        x='timestamp',
    )
    timestamps_evolution.update_layout(
        legend=dict(yanchor="top", y=-0.3, xanchor="left", x=-0.15, orientation='h'),
        margin=dict(l=5, r=5, t=5, b=5),
        yaxis_title=None,
        hovermode="x unified"
    )
    st.plotly_chart(timestamps_evolution, use_container_width=True)

with columns_content[1]:
    st.write('**Mensajes FPLAN**')
    with st.container(height=275):
        st.dataframe(data_fp[data_fp.ifplId == flight_id]
                     .drop(['ifplId', 'uuid'], axis=1)
                     .rename({'aerodromeOfDeparture':'departure', 'aerodromeOfDestination':'destination'}, axis=1))
    st.write('**Mensajes FDATA**')
    with st.container(height=425):
        st.dataframe(data_fd[data_fd.ifplId == flight_id]
                     .drop(['ifplId', 'uuid'], axis=1)
                     .rename({'aerodromeOfDeparture':'departure', 'aerodromeOfDestination':'destination'}, axis=1))
st.dataframe(data_flight)
