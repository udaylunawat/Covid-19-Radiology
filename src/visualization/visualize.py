import os
import plotly.graph_objects as go
import joblib

covid_image_dir = 'data/0_raw/COVID-19 Radiography Database/COVID-19'
# COVID19images = os.listdir(covid_image_dir)

def output_grid():
    matplotlib.rcParams.update({'font.size': 15})

    plt.figure(figsize=(15,15))
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        image, pred_label, probs = predict_label(os.path.join(covid_image_dir,COVID19images[i]))
        # image = cv2.imread(os.path.join(covid_image_dir,COVID19images[i]))
        plt.imshow((image),cmap='gray'), plt.axis("off")
        plt.title("Actual : COVID19\n Prediction : {}".format(pred_label))
    plt.show()


# Load the history from the file 
history = joblib.load('output/history.pkl')  

def metrics_plotly(metrics, title):
    # Create traces
    fig = go.Figure()

    for metric in metrics:
        fig.add_trace(go.Scatter(y=history[metric],
                            mode='lines+markers',
                            name=metric,
                            hovertemplate=
                            "<br>Epoch: %{x} </br>"+metric+": %{y:.2f}"))
        
    fig.update_layout(
        title=title,
        xaxis_title="Epochs",
        yaxis_title="Accuracy",
        # legend_title="Legend Title",
        # hovermode='y',
        font=dict(
            family="Courier New, monospace",
            size=18,
            color="RebeccaPurple"
        )
    )
    return fig

import plotly.figure_factory as ff

def plotly_cm(cm):
    z = cm.values

    x = ['COVID-19', 'NORMAL', 'Viral Pneumonia'] # encoder.classes_
    y =  ['COVID-19', 'NORMAL', 'Viral Pneumonia'] # encoder.classes_

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