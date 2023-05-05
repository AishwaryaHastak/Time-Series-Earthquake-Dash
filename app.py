# app.py

import sys
sys.path.append('/home/EarthquakeDash/.local/lib/python3.10/site-packages')
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import dash
from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose


# Prepare Dataset
earthquakes = pd.read_csv("/home/EarthquakeDash/mysite/data/earthquakes.csv", parse_dates=True)
earthqake_df = earthquakes[earthquakes['Year'] >= 1500]
earthqake_df['Count'] = earthqake_df.groupby('Year')['Year'].transform('count')
earthqake_temp = earthqake_df.dropna(subset=['Count','Mo'], how='any')
earthqake_temp = earthqake_temp[['Year','Count','Mo']]
earthqake_temp['Year'] = earthqake_temp['Year'].fillna(0).astype(int)
earthqake_temp['Count'] = earthqake_temp['Count'].fillna(0).astype(int)
earthqake_temp['Mo'] = earthqake_temp['Mo'].fillna(0).astype(int)
earthqake_temp.set_index('Year', inplace=True)
# Get user credentials dataset

"""
try:
    mydb = connection.connect(host="localhost", database = 'Student',user="root", passwd="root",use_pure=True)
    query = "Select * from studentdetails;"
    result_dataFrame = pd.read_sql(query,mydb)
    mydb.close() #close the connection
except Exception as e:
    mydb.close()
    print(str(e))
"""

#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
# Add dashboard elements

app = dash.Dash(__name__,suppress_callback_exceptions=True)

#-----------------------------------------------------------------------
# Add Plots for EDA Tab

# Major Earthquakes Worldwide(With Magnitude Greater Than 7)

# create a map of area, where houses from data set located
earthquake_map = earthquakes.dropna(subset=['Mag','Total Deaths']).copy()
#earthqake_map = earthqake_df[earthqake_df['Mag'] >= 5.0]
hover_data = ['Mag', 'Total Deaths']
fig1 = px.scatter_geo(earthquake_map, lat='Latitude', lon='Longitude',
color='Mag', size='Mag', hover_name='Location Name', hover_data=hover_data,
                        width=1400, height=600)


# Figure 2
earthqake_cnt = earthquakes.loc[earthquakes['Year'] >= 1500].copy()
earthqake_cnt['Count'] = earthqake_cnt.groupby('Year')['Year'].transform('count')
fig2 = px.line(earthqake_cnt, x='Year', y="Count")


# Figure3: 7 Mag

earthqake_maj = earthqake_df[earthqake_df['Mag'] >= 7.0].copy()
earthqake_maj.dropna(subset=['Mag','Deaths'],inplace=True)
fig3 = px.scatter_mapbox(earthqake_maj,
                        lat="Latitude", lon="Longitude",
                        color="Mag",
                        hover_name="Deaths",
                        hover_data=["Deaths",'Location Name'],
                        size_max=15,
                        zoom=6,
                        width=1400, height=600)


# Figure 4

result = seasonal_decompose(earthqake_temp['Count'], period=12, model='additive')

# Plot the trend component
fig4 = px.line(result.trend,
                 labels={
                     "Yr": "Year",
                     "value": "Frequency"
                 },
                title="Trend Component")

# Plot the seasonal component
fig5 = px.line(result.seasonal,
                 labels={
                     "Yr": "Year",
                     "value": "Frequency"
                 },
                title="Seasonal Component")

# Plot the Residual component
fig6 = px.line(result.resid,
                 labels={
                     "Yr": "Year",
                     "value": "Frequency"
                 },
                title="Residual Component")

# Figure 7
counts = earthqake_df.groupby('Location Name')['Location Name'].count().sort_values(ascending=False)
#counts.head(10).plot(kind='bar')
fig7 = px.bar(counts.head(10),
                 labels={
                     "value": "Frequency",
                     "index": "Location Name"
                 })

# Figure 8
earthqake_map = earthqake_df.pivot_table(values='Count', index='Location Name', columns='Year', aggfunc='count')
fig8 = px.imshow(earthqake_map, text_auto=True)

# Figure 9
earthqake_counts = earthqake_df[earthqake_df['Mag'] > 8].groupby(['Location Name', 'Mag'])['Count'].count().reset_index()
fig9 = px.scatter(earthqake_counts, x='Mag',y='Location Name',
                  color=earthqake_counts['Count']*10,
                  hover_data=['Mag'],
                 labels={
                     "Mag": "Magnitude"
                 }
                  )

# Figure 10
def get_plot(value) :
    earthqake_jpn = earthqake_df[earthqake_df["Location Name"].str.contains(value, na = False)].copy()
    earthqake_jpn.dropna(subset=['Deaths'],inplace=True)
    fig10 = px.scatter_mapbox(earthqake_jpn,
                            lat="Latitude", lon="Longitude",
                            color="Mag",
                            size="Deaths",
                            hover_name="Deaths",
                            hover_data=["Deaths"],
                            size_max=60,
                            zoom=6,
                            width=1400, height=600)
    #style of map
    fig10.update_layout(mapbox_style="carto-darkmatter")
    return fig10

# Figure11 Tsunami
earthqake_tsu = earthqake_df.dropna(subset=['Tsu','Mag','Deaths'])
fig11 = px.scatter_mapbox(earthqake_tsu, #our data set
                        lat="Latitude", lon="Longitude", #location
                        color="Mag",
                        size="Tsu",
                        hover_name="Mag",
                        hover_data=["Tsu","Deaths","Location Name"],
                        #color_continuous_scale=px.colors.cyclical.IceFire,
                        size_max=20,
                        zoom=3,
                        width=1400, height=600)
#style of map
fig11.update_layout(mapbox_style="open-street-map")


#-----------------------------------------------------------------------
# Fit ARIMA Model
from statsmodels.tsa.arima.model import ARIMA

earthqake_yr = earthqake_temp.drop(['Mo'], axis=1)
earthqake_yr.plot()

mask =  np.random.rand(len(earthqake_yr)) < 0.8

earthqake_train = earthqake_yr[mask].copy()
earthqake_test = earthqake_yr[~mask].copy()
model = ARIMA(earthqake_train, order=(6, 0, 1))
model_fit = model.fit()

residuals = model_fit.resid[1:]
plot1 = px.line(residuals,
             labels={
                 "Yr": "Year",
                 "value": "Residuals"
             })

forecast = model_fit.forecast(steps=115)
plot2 = go.Figure()
plot2.add_trace(go.Scatter(x=earthqake_train.index, y=earthqake_train.Count,
                    mode='lines', name='Training Data',
                         line=dict(color='firebrick')))

plot2.add_trace(go.Scatter(x=earthqake_test.index, y=earthqake_test.Count,
                    mode='lines', name='Testing Data',
                         line=dict(color='blue')))

plot2.add_trace(go.Scatter(x=np.arange(2023, 2138), y=forecast,
                    mode='lines', name='Forecast',
                         line=dict(color='black')))

#-----------------------------------------------------------------------
# Create Layout

eda = html.Div(
            className='graph',
            children = [
            # Yeild Curve (Figure1)
              html.H2(children='Map of Earthquakes around the World'),
              dcc.Graph(
                 id='graph1',
                 figure=fig1  ),
                html.Br(),
            # Count of Earthquakes (Figure2)
              html.H2(children='Frequency of Earthquakes over the years'),
              dcc.Graph(
                 id='graph2',
                 figure=fig2  ),
                html.Br(),
            # Seasonal Decompose (Figure4,5,6)
              html.H2(children='Plot the results of Additive Seasonal Decomposition'),
              dcc.Graph(
                 id='graph4',
                 figure=fig4  ),
                html.Br(),
              dcc.Graph(
                 id='graph5',
                 figure=fig5  ),
                html.Br(),
              dcc.Graph(
                 id='graph6',
                 figure=fig6  ),
                html.Br(),
            # Figure 7
              html.H2(children='Top 10 Most Active Earthquake Regions'),
              dcc.Graph(
                 id='graph7',
                 figure=fig7  ),
                html.Br(),
            # Figure 8
              html.H2(children='Earthquake Frequency by Location and Year'),
              dcc.Graph(
                 id='graph8',
                 figure=fig8  ),
                html.Br(),
            # Figure 9
              html.H2(children='Earthquake Frequency by Location and Magnitude (Magnitude > 8)'),
              dcc.Graph(
                 id='graph9',
                 figure=fig9  ),
                html.Br(),
            # Figure 11
              html.H2(children='Map of Earthquakes accompanied with Tsunamies'),
              dcc.Graph(
                 id='graph11',
                 figure=fig11  ),
                html.Br(),
          html.H2(children='Select a Location'),
        # Prediction Model Time Period
        html.Div([
            dcc.Dropdown(['INDIA', 'CHINA', 'TURKEY', 'JAPAN','MEXICO','UNITED KINGDOM','RUSSIA','CALIFORNIA'],
                         'CALIFORNIA', id='demo-dropdown'),
            html.Div(id='dd-output-container'),
            ]),
            ])


prediction =  html.Div(
            className='graph',
            children = [
            # Residual (Plot1)
              html.H2(children='Residual Plot'),
              dcc.Graph(
                 id='plot1',
                 figure=plot1  ),
                html.Br(),
            # Forecast (Plot2)
              html.H2(children='Forecast of Frequency of Earthquakes over next 50 Years'),
              dcc.Graph(
                 id='plot2',
                 figure=plot2  )
            ])


next_page = html.Div(
        className='background',
        #style={ "background-color": "#f2eee9"},
        children=[
         html.Div(
            className='hero-nav',
            children=[
            html.H1(children="Earthquake Dashboard:",
            className='heading'
                   ),
            html.Br(),
            html.P(
                children="A Time Series Project to Analyse and Forecast Earthquakes around the World",
                    className='sub-heading'
            ),
            ]),

        html.Br(),
        # Create 2 tabs
        html.Div(className = "page-content",
        children = [
        dcc.Tabs(id="tabs", value='eda',
                parent_className='custom-tabs',
                className='custom-tabs-container',
                 children=[
                    dcc.Tab(label='Exploratory Data Analysis', value='eda',
                            children = eda,
                        className='custom-tab',selected_className='custom-tab--selected'),
                    dcc.Tab(label='Forecasting', value='prediction',
                            children = prediction,
                        className='custom-tab',selected_className='custom-tab--selected'),
                    ]),
             html.Div(id='tabs-content'),
                ]),
        html.Br(),
    ])


@app.callback(
    Output('dd-output-container', 'children'),
    Input('demo-dropdown', 'value'),
      prevent_initial_callback=True)

def update_output(value):
    return html.Div( className='graph',
                    children = [
              html.H2(children='Earthquake Fatalities in and around Selected Location'),
             # Figure 10
              dcc.Graph(
                 id='graph10',
                 figure=get_plot(value) )
                    ])



login = html.Div(
        className='background',
        #style={ "background-color": "#f2eee9"},
        children=[
         html.Div(
            className='hero-nav',
            children=[
            html.H1(children="Earthquake Dashboard:",
            className='heading'# hero-nav__inner',
                   ),
            html.Br(),
            html.P(
                children="A Time Series Project to Analyse and Forecast Earthquakes around the World",
                    className='sub-heading'# hero-nav__inner',
            ),
            ]),

        html.Br(),
        html.H1(children='Good to See you again!',
        style={'margin-left':'35%','margin-top':'60px','color':'white'
        }),
        html.H3(children='Username',
        style={'margin-left':'35%', 'font-size':'16px','color':'white'
        }),
        html.Div(
        dcc.Input(id="user", type="text", placeholder="Enter Username",className="inputbox1",
        style={'margin-left':'35%','width':'450px','height':'45px','padding':'10px',
        'font-size':'16px','border-width':'3px','border-color':'#a0a3a2'
        }),
        ),
        html.H3(children='Password',
        style={'margin-left':'35%','margin-top':'60px','color':'white'
        }),
        html.Div(
        dcc.Input(id="passw", type="text", placeholder="Enter Password",className="inputbox2",
        style={'margin-left':'35%','width':'450px','height':'45px','padding':'10px','margin-top':'10px',
        'font-size':'16px','border-width':'3px','border-color':'#a0a3a2',
        }),
        ),
        html.Div(
        html.Button('Verify', id='verify', n_clicks=0, style={'width':'50px','height':'30px',
                                                              'border-width':'2px','font-size':'14px'}),
        style={'margin-left':'47%','padding-top':'30px'}),
        html.Div(id='output1')
        ])

@app.callback(
    Output('output1', 'children'),
   Input('verify', 'n_clicks'),
    [State('user', 'value') ,State('passw', 'value')])

def update_output(n_clicks, uname, passw):
    li={'admin':'admin','root':'root'}
    if uname =='' or uname == None or passw =='' or passw == None:
        return html.Div(children='',style={'padding-left':'550px','padding-top':'10px'})
    if uname not in li:
        return html.Div(children='Incorrect Username',style={'color':'white','padding-left':'550px','padding-top':'40px','font-size':'16px'})
    if li[uname]==passw:
        #return next_page
        return html.Div(dcc.Link('Access Granted! CLick here for Next Page', href='/next_page',style={'color':'white','font-family': 'serif', 'font-weight': 'bold', "text-decoration": "none",'font-size':'20px'})
                        ,style={'padding-left':'580px','padding-top':'40px'})
    else:
        return html.Div(children='Incorrect Password',style={'color':'white','padding-left':'550px','padding-top':'40px','font-size':'16px'})


@app.callback(dash.dependencies.Output('page-content', 'children'),
[dash.dependencies.Input('url', 'pathname')])

def display_page(pathname):
    if pathname == '/next_page':
        return next_page
    else:
        return login


app.layout = html.Div([
dcc.Location(id='url', refresh=False),
html.Div(id='page-content')
                 ])


#-------------------------------------------------------------------------#
# Run the dashboard

if __name__ == '__main__':
    app.run_server(host='0.0.0.0')
    #app.run_server(debug=True)

