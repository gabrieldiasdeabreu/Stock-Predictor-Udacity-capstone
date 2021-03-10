import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px
import model 
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

def menu():
    dropdown = dcc.Dropdown(
        id='dropdown',
        options=[
            {'label': 'Ibovespa', 'value': '^BVSP'},
            {'label': 'apple', 'value': 'aapl'},
        ],
        value='^BVSP',        
    )

    slider = dcc.Slider(
        id='slider',
        min=1,
        max=28,
        step=1,
        value=14,
    )

    return [
        html.Div(children=[html.H3("Select a stock"),
        dropdown,]),
        html.Div(children=[html.H3("Select the number of days ahead to predict"),
        html.Div(id='slider-output-container'),
        slider])        
    ]

app.layout = html.Div([
    html.H1("Stock Predictor"),        
    html.Div(
        children=menu(),
        style={
            'display': 'flex',                
            'justify-content': 'space-around',

        }
    ),    
    html.Br(),
    html.Br(),
    dcc.Loading(
            id="loading",
            type="default",
            children=html.Div(id="loading-output")
    ),
])


@app.callback(
    Output(component_id='loading-output', component_property='children'),
    Input(component_id='dropdown', component_property='value'),
    Input(component_id='slider', component_property='value')
)
def update_output_div(dropdown_value, slider_value):
    window_size = 90
    days_ahead, predicted = model.train_and_execute_model(dropdown_value, window_size, slider_value)
    print(days_ahead, predicted)    
    print(dropdown_value, slider_value)
    
    fig = px.line(x=days_ahead, y=predicted, title=f'Next {slider_value} days prediction', )

    fig.update_layout(xaxis_title='days',
                  yaxis_title='close price')

    fig.update_traces(marker={'size':5,})

    return dcc.Graph(
        id='graph',
        figure=fig
    )

@app.callback(
    dash.dependencies.Output('slider-output-container', 'children'),
    [dash.dependencies.Input('slider', 'value')])
def update_output(value):
    return 'You have selected "{}"'.format(value)



if __name__ == '__main__':
    app.run_server(debug=True)
