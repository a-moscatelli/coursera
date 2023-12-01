# Import required libraries
import pandas as pd
import dash
#import dash_html_components as html - deprecated
from dash import html
#import dash_core_components as dcc - deprecated
from dash import dcc
from dash.dependencies import Input, Output
import plotly.express as px


if __name__ == '__main__':
	print('Reading the airline data into pandas dataframe ...')
	spacex_df = pd.read_csv("spacex_launch_dash.csv")
	max_payload = spacex_df['Payload Mass (kg)'].max()
	min_payload = spacex_df['Payload Mass (kg)'].min()

	print('df columns:')
	print('='*8)
	[ print(c) for c in spacex_df.columns.tolist() ]
	print('='*8)
	print(spacex_df.info())

	sites = spacex_df['Launch Site'].unique()

	dict1 = [{'label': 'All Sites', 'value': 'ALL'}]
	dict2 = [{'label':a, 'value':a} for a in sites ]

	dictz = dict1 + dict2
	print('launch sites:')
	print('='*8)
	for dd in dictz:
		for k,v in dd.items():
			print(k,':', v)
	print('='*8)

# Create a dash application
app = dash.Dash(__name__)

# Create an app layout
app.layout = html.Div(children=[

								html.H1('SpaceX Launch Records Dashboard',
                                        style={'textAlign': 'center', 'color': '#503D36',
                                               'font-size': 40}),
                                
								# TASK 1: Add a dropdown list to enable Launch Site selection
                                # The default select value is for ALL sites
                                dcc.Dropdown(id='site-dropdown',
                                      options=dictz,
                                    value='ALL',
                                    placeholder='Select a Launch Site here',
                                    searchable =True
                                ),
                                html.Br(),

                                # TASK 2: Add a pie chart to show the total successful launches count for all sites
                                # If a specific launch site was selected, show the Success vs. Failed counts for the site
                                html.Div(dcc.Graph(id='success-pie-chart')),
                                html.Br(),

                                html.P("Payload range (Kg):"),
                                
								# TASK 3: Add a slider to select payload range
                                #dcc.RangeSlider(id='payload-slider',...)
                                dcc.RangeSlider(id='payload-slider',
                                                min=0, max=10000, step=1000,
                                                marks={0: '0',
                                                    100: '100'},
                                                value=[min_payload, max_payload]),

                                
								# TASK 4: Add a scatter chart to show the correlation between payload and launch success
                                html.Div(dcc.Graph(id='success-payload-scatter-chart')),
                                ])

# TASK 2:
# Add a callback function for `site-dropdown` as input, `success-pie-chart` as output
# Function decorator to specify function input and output
@app.callback(Output(component_id='success-pie-chart', component_property='figure'),
	Input(component_id='site-dropdown', component_property='value'))
def get_pie_chart(entered_site):
    # AM: check https://plotly.com/python/pie-charts/
    filtered_df = spacex_df
    if entered_site == 'ALL':
        data = filtered_df
        fig = px.pie(data, 
            values='class', 
            names='Launch Site', 
            title='all-sites')
        return fig
    else:
        # return the outcomes piechart for a selected site
        data = filtered_df[filtered_df['Launch Site']==entered_site]
        data2 = data[['class']].value_counts().reset_index()
        print(data2.columns)
        print('for selection:',entered_site,data[['Launch Site','class']].value_counts().to_dict()) 
        fig = px.pie(data2, 
            values='count', 
            names='class', 
            title='single-site')
        return fig

# TASK 4:
# Add a callback function for `site-dropdown` and `payload-slider` as inputs, `success-payload-scatter-chart` as output

@app.callback(
	Output(component_id='success-payload-scatter-chart', component_property='figure'),
	[Input(component_id='site-dropdown', component_property='value'), 
	Input(component_id="payload-slider", component_property="value")])
def get_scatter_chart(entered_site, entered_range):
    # AM: check https://plotly.com/python/pie-charts/
    filtered_df = spacex_df
    print('records without slider data:', len(filtered_df))
    filtered_df = filtered_df[ (filtered_df['Payload Mass (kg)'] >= entered_range[0]) & (filtered_df['Payload Mass (kg)'] <= entered_range[1]) ]
    print('records with slider data:', len(filtered_df))
    print('entered_range',entered_range)	# [2000, 7000]
    if entered_site == 'ALL':
        data = filtered_df
        fig = px.scatter(data, 
            y='class', 
            x='Payload Mass (kg)', 
            title='all-sites')
        return fig
    else:
        # return the outcomes piechart for a selected site
        data = filtered_df[filtered_df['Launch Site']==entered_site]
        #data2 = data[['class']].value_counts().reset_index()
        #print(data2.columns)
        #print('for selection:',entered_site,data[['Launch Site','class']].value_counts().to_dict()) 
        fig = px.scatter(data, 
            y='class', 
            x='Payload Mass (kg)', 
            title='single-site')
        return fig


# Run the app
if __name__ == '__main__':
    app.run_server()
	#app.run_server(debug=True, threaded=True)		# with auto-reload? no. main code will run twice.
	# https://community.plotly.com/t/app-loading-twice/35093/3
	
