import os
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from sklearn.metrics import confusion_matrix

from src.config import DATA_DIR, class_dict
from src.models.inference import predict_label

## Bar Plot
def counts_bar(labels, label_counts):
    '''
    Creates a plotly bar plot with Target label value_counts.


    Parameters
    ----------
    labels : list like input for target / classes
    label_counts : array_like input for counts of classes


    Returns
    -------
    bar plot plotly fig
    '''
    fig = go.Figure()
    fig.add_trace(go.Histogram(histfunc="sum",
                            x=labels,
                            y=label_counts,
                            opacity=0.3,
                            marker=dict(color=['Yellow', 'Green', 'Red'])))

    fig.update_layout(
        title="Bar plot",
        yaxis_title="Count",
        # legend_title="Legend Title",
        font=dict(
            family="Courier New, monospace",
            size=18,
            color="RebeccaPurple"
        )
    )
    fig.update_layout()
    fig.show()
    return fig


def metrics_plotly(history, metrics, title):
    '''
    Plots line chart for given history metrics with title

    Parameters
    ----------
    history : dict type input of history.history 
    metrics : array_like input for history metric names
    title : string like title of plot

    Returns
    -------
    line plot plotly fig
    '''

    # Create traces
    fig = go.Figure()

    for metric in metrics:
        fig.add_trace(go.Scatter(y=history[metric],
                            mode='lines+markers',
                            name=metric))
        
    fig.update_layout(
        title=title,
        xaxis_title="Epochs",
        yaxis_title="Accuracy/Loss",
        # legend_title="Legend Title",
        font=dict(
            family="Courier New, monospace",
            size=18,
            color="RebeccaPurple"
        )
    )

    return fig

def plotly_cm(cm, label_list):
    '''
    Plots annotated heatmap of confusion matrix

    Parameters
    ----------
    cm : confusion matrix
    label_list : array_like input for history metric names

    Returns
    -------
    line plot plotly fig
    '''

    z = cm.values

    x = label_list
    y =  label_list

    # change each element of z to type string for annotations
    z_text = [[str(y) for y in x] for x in z]

    # set up figure 
    fig = ff.create_annotated_heatmap(z, x=list(x), y=list(y), annotation_text=z_text, colorscale='Viridis')

    # add title
    fig.update_layout(title_text='<i><b>Confusion matrix</b></i>',
                    #xaxis = dict(title='x'),
                    #yaxis = dict(title='x')
                    )

    # add custom xaxis title
    fig.add_annotation(dict(font=dict(color="black",size=14),
                            x=0.5,
                            y=-0.15,
                            showarrow=False,
                            text="Predicted value",
                            xref="paper",
                            yref="paper"))

    # add custom yaxis title
    fig.add_annotation(dict(font=dict(color="black",size=14),
                            x=-0.35,
                            y=0.5,
                            showarrow=False,
                            text="Real value",
                            textangle=-90,
                            xref="paper",
                            yref="paper"))

    # adjust margins to make room for yaxis title
    fig.update_layout(margin=dict(t=50, l=200))

    # add colorbar
    fig['data'][0]['showscale'] = True

    return fig

def plot_map(df, col):
    '''
    Plots choropleth maps of live covid statistics based on column passed.
    Data is displayed in a log scale.

    Parameters
    ----------
    df : DataFrame input for preprocessed covid stats
    col : string input for column names

    Returns
    -------
    choropleth map plot plotly fig
    '''

    # df = df[df[col]>0]
    fig = px.choropleth(df, locations="country_name", locationmode='country names', 
                  color=np.log10(df[col]), hover_name="country_name", 
                  title=col, hover_data=[col], color_continuous_scale=px.colors.sequential.Plasma)
    fig.update_layout(coloraxis_colorbar=dict(
        title=col,
        tickvals=[2,3,4,5,6,7],
        ticktext=["100","1K","10K", "100K","1M", "10M"]))
    return fig

def grid_plot(model, function, label='Covid-19'):
    '''
    Plots grid plot for data and prediction as well.

    Parameters
    ----------
    label : string like input for Category name
    function : string input either "Show" or "Predict"

    Returns
    -------
    None
    '''
    image_dir = os.path.join(DATA_DIR, label)
    images_list = os.listdir(image_dir)

    matplotlib.rcParams.update({'font.size': 9})

    plt.figure(figsize=(15,15))
    for i in range(16):
        plt.subplot(4, 4, i + 1)
        image = cv2.imread(os.path.join(image_dir, images_list[i]))

        if function == "Show":
            plt.title("Filename: {}\nClass: {}".format(images_list[i], label))

        elif function == "Predict":
            image, pred_label, probs = predict_label(image, model)
            plt.title("Filename: {}\nActual: {}\nPrediction: {}".format(images_list[i], label, pred_label))

        plt.imshow((image),cmap='gray'), plt.axis("off")
    plt.tight_layout()
    plt.show()