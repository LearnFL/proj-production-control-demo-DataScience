from collections import namedtuple
from pathlib import Path
import pandas as pd
import os
from os import path
import matplotlib.pyplot as plt
import matplotlib 
import datetime as dt
from datetime import timedelta
import numpy as np
import plotly.express as px
import plotly.graph_objects as go # or plotly.express as px
import dash
import dash_core_components as dcc
import dash_html_components as html

file_path=Path(r"SOME_PATH") 
col = [0,2,4,7,8,9,13]

if path.exists(file_path):
    print('Path exists')
else:
    print('Check path') 

df = pd.read_excel(file_path, usecols = col, dtype='object', na_values=[''], parse_dates=True)
df=df.iloc[6:]
print(df.head(5))
print(df.dtypes)
print('Creating columns\n')
df.columns=["WORK_ORDER","CUSTOMER","ITEM","START_DATE","DUE_DATE", "QUANTITY", "NOTES"]

print('Converting to datetime and creating index\n')
df['DUE_DATE'] =pd.to_datetime(df['DUE_DATE'])
df['START_DATE'] =pd.to_datetime(df['START_DATE'])

print('Creating start day column')

df['DURATION'] = (df.DUE_DATE - df.START_DATE).dt.days
df['COMPLETION'] = 0

print(df.head(5))
print(df.dtypes)

'''DATE FILTER'''
df = df[(df['START_DATE'] >= '12/01/2021') & (df['DUE_DATE'] <= '02/28/2022')]

proj_start = df.START_DATE.min()

# number of days from project start to task start
df['start_num'] = (df.START_DATE - proj_start).dt.days

# number of days from project start to end of tasks
df['end_num'] = (df.DUE_DATE - proj_start).dt.days

# days between start and end of each task
df['days_start_to_end'] = df.end_num - df.start_num
df['WORK_ORDER'] = df['WORK_ORDER'].astype(str)
print(df.head(5))
fig, ax = plt.subplots(1, figsize=(16,6))
ax.barh(df.WORK_ORDER, df.days_start_to_end, left=df.start_num)
xticks = np.arange(0, df.end_num.max()+1, 3)
xticks_labels = pd.date_range(proj_start, end=df.DUE_DATE.max()).strftime("%m/%d")
xticks_minor = np.arange(0, df.end_num.max()+1, 1)
ax.set_xticks(xticks)
ax.set_xticks(xticks_minor, minor=True)
ax.set_xticklabels(xticks_labels[::3], rotation=90)


#COMPLETION

# days between start and current progression of each task
df['current_num'] = (df.days_start_to_end * df.COMPLETION)
from matplotlib.patches import Patch
fig, ax = plt.subplots(1, figsize=(16,6))

# bars
ax.barh(df.WORK_ORDER, df.current_num, left=df.start_num)
ax.barh(df.WORK_ORDER, df.days_start_to_end, left=df.start_num, alpha=0.5)

# texts
xticks = np.arange(0, df.end_num.max()+1, 3)
xticks_labels = pd.date_range(proj_start, end=df.DUE_DATE.max()).strftime("%m/%d")
xticks_minor = np.arange(0, df.end_num.max()+1, 1)
ax.set_xticks(xticks)
ax.set_xticks(xticks_minor, minor=True)
ax.set_xticklabels(xticks_labels[::3], rotation=90)

plt.show()

fig1 = px.timeline(df, x_start="START_DATE", x_end="DUE_DATE", y='ITEM', text='WORK_ORDER', color="ITEM", title='Production Schedule')
fig1.update_yaxes(autorange="reversed") # otherwise tasks are listed from the bottom up
fig1.show()

fig2 = px.timeline(df, x_start="START_DATE", x_end="DUE_DATE", y="ITEM", text='WORK_ORDER', color="COMPLETION", range_color=(0,100), title='Production Schedule Completion')
fig2.update_yaxes(autorange="reversed")
fig2.show()

def generate_table(dataframe, max_rows):
    return html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in dataframe.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
            ]) for i in range(min(len(dataframe), max_rows))
        ])
    ])

app = dash.Dash()

colors = {
    'background': 'white',
    'fig_background': 'lightgray',
    'text': '#7FDBFF',
    'fig_text': 'Black'
}

fig1.update_layout(
    plot_bgcolor=colors['fig_background'],
    paper_bgcolor=colors['fig_background'],
    font_color=colors['fig_text']
)

fig2.update_layout(
    plot_bgcolor=colors['fig_background'],
    paper_bgcolor=colors['fig_background'],
    font_color=colors['fig_text']
)

app.layout = html.Div(style={'backgroundColor': colors['background']}, children=[

    html.Div(children='NAME.', style={
        'textAlign': 'center',
        'color': colors['text'],
    }),

    html.Div([
        html.H2(children='PRODUCTION SCHEDULE'),

        html.Div(children='''
            Shows 3 month of data.
        '''),

        dcc.Graph(
            id='FIG_1',
            style={'height': '90vh'},
            figure=fig1
        ),  
    ]),

    html.Div([
        html.H2(children='PRODUCTION SCHEDULE % COMPLETION'),

        html.Div(children='''
            Shows 3 month with persent completed.
        '''),

        dcc.Graph(
            id='FIG_2',
            style={'height': '90vh'},
            figure=fig2
        ),  
    ]),
    
    html.Div([
        html.H2(children='SCHEDULE TABLE'),
        generate_table(df, 200)
    ]),
])

app.run_server(debug=True, use_reloader=False)  # Turn off reloader if inside Jupyter
