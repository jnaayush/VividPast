#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 22:05:16 2018

@author: justinelo
"""

import plotly
import plotly.graph_objs as go
plotly.offline.init_notebook_mode(connected=True)
import numpy as np

# PLOT WITHOUT FUSION:

y0 = np.array([0, 2, 3, 5, 1])
y1 = np.array([13, 6, 20, 18])
y2 = np.array([10, 6, 20, 15, 18])
y3 = np.array([7, 6, 20, 15, 18])
y4 = np.array([13, 6, 1])
y5 = np.array([40, 6, 20, 15, 18])

##### color: 'mediumblue', 'maroon'
##### isFusion 'true' or 'false'
def box_plot(y0, y1, y2, y3, y4, y5, color, isFusion):
    trace0 = go.Box(
        y=y0,
        name = '1',
        marker=dict(
            color=color
        )
    )
    trace1 = go.Box(
        y=y1,
        name = '2',
        marker=dict(
            color=color
        )
    )
    trace2 = go.Box(
        y=y2,
        name = '3',
        marker=dict(
            color=color
        )
    )
    trace3 = go.Box(
        y=y3,
        name = '4',
        marker=dict(
            color= color
        )
    )
    trace4 = go.Box(
        y=y4,
        name = '5',
        marker=dict(
            color=color
        )
    )
    trace5 = go.Box(
        y=y5,
        name = '6',
        marker=dict(
            color=color
        )
    )
    
    data = [trace0, trace1, trace2, trace3, trace4, trace5]
    
    if (isFusion):
        title_name = 'With Fusion: Loss vs. Objects in Image'
    else: 
        title_name = 'Without Fusion: Loss vs. Objects in Image'
    
    layout = go.Layout(
        title= title_name,
        xaxis=dict(
            title='Number of Objects in Image',
            titlefont=dict(
    #            family='Courier New, monospace',
                size=18,
    #            color='#7f7f7f'
            )
        ),
        yaxis=dict(
            title='Loss',
            titlefont=dict(
    #            family='Courier New, monospace',
                size=18,
    #            color='#7f7f7f'
            )
        )
    )
    
    fig = go.Figure(data=data, layout=layout)
    plotly.offline.plot(fig)


# PLOT WITH FUSION:

#y0 = np.array([0, 2, 3, 5, 1])
#y1 = np.array([13, 6, 20, 18])
#y2 = np.array([10, 6, 20, 15, 18])
#y3 = np.array([7, 6, 20, 15, 18])
#y4 = np.array([13, 6, 1])
#y5 = np.array([40, 6, 20, 15, 18])
#
#trace0 = go.Box(
#    y=y0,
#    name = '1',
#    marker=dict(
#        color='green'
#    )
#)
#trace1 = go.Box(
#    y=y1,
#    name = '2',
#    marker=dict(
#        color='green'
#    )
#)
#trace2 = go.Box(
#    y=y2,
#    name = '3',
#    marker=dict(
#        color='green' 
#    )
#)
#trace3 = go.Box(
#    y=y3,
#    name = '4',
#    marker=dict(
#        color= 'green'
#    )
#)
#trace4 = go.Box(
#    y=y4,
#    name = '5',
#    marker=dict(
#        color='green'
#    )
#)
#trace5 = go.Box(
#    y=y5,
#    name = '6',
#    marker=dict(
#        color='green'
#    )
#)
#
#data2 = [trace0, trace1, trace2, trace3, trace4, trace5]
#
#layout = go.Layout(
#    title='With Fusion: Loss vs. Objects in Image',
#    xaxis=dict(
#        title='Number of Objects in Image',
#        titlefont=dict(
##            family='Courier New, monospace',
#            size=18,
##            color='#7f7f7f'
#        )
#    ),
#    yaxis=dict(
#        title='Loss',
#        titlefont=dict(
##            family='Courier New, monospace',
#            size=18,
##            color='#7f7f7f'
#        )
#    )
#)
#
#fig2 = go.Figure(data=data2, layout=layout)
#plotly.offline.plot(fig2)