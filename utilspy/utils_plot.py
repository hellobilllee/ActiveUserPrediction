import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Plots the disribution of a variable colored by value of the target
def corr_coefficient(df,method="pearson"):
    import plotly.graph_objs as pgo
    import plotly.offline as po
    data = [
        pgo.Heatmap(
            z=df.corr(method=method).values,
            x=df.columns.values,
            y=df.columns.values,
            colorscale='Viridis',
            reversescale=False,
            text=True,
            opacity=1.0)
    ]

    layout = pgo.Layout(
        title=method+' Correlation of features',
        xaxis=dict(ticks='', nticks=36),
        yaxis=dict(ticks=''),
        width=900, height=700,
        margin=dict(
            l=240,
        ), )

    fig = pgo.Figure(data=data, layout=layout)
    po.iplot(fig, filename='labelled-heatmap')


def kde_target(var_name, df,target="label"):
    import matplotlib.pyplot as plt  # for plotting
    import seaborn as sns  # for making plots with seaborn
    # Calculate the correlation coefficient between the new variable and the target
    corr = df[target].corr(df[var_name])

    # Calculate medians for repaid vs not repaid
    avg_repaid = df.ix[df[target] == 0, var_name].median()
    avg_not_repaid = df.ix[df[target] == 1, var_name].median()

    plt.figure(figsize=(12, 6))

    # Plot the distribution for target == 0 and target == 1
    sns.kdeplot(df.ix[df[target] == 0, var_name], label='label == 0')
    sns.kdeplot(df.ix[df[target] == 1, var_name], label='label == 1')

    # label the plot
    plt.xlabel(var_name);
    plt.ylabel('Density');
    plt.title('%s Distribution' % var_name)
    plt.legend();

    # print out the correlation
    print('The correlation between %s and the TARGET is %0.4f' % (var_name, corr))
    # Print out average values
    print('Median value for loan that was not repaid = %0.4f' % avg_not_repaid)
    print('Median value for loan that was repaid =     %0.4f' % avg_repaid)
def value_count_bar_default(df,series_name,title="value count bar"):
    temp = df[series_name].value_counts()
    temp = pd.DataFrame({'labels': temp.index,
                       'values': temp.values
                       })
    temp.iplot(kind='bar', xTitle=series_name, yTitle="Count",
               title=title, colors=['#75e575'])
    import seaborn as sns
    plt.figure(figsize=(12, 5))
    plt.title("Distribution of register day")
    ax = sns.distplot(df_user_register["register_day"])

def value_count_bar(df,series_name,title="value count plot"):
    import plotly.graph_objs as pgo
    import plotly.offline as po
    temp = df[series_name].value_counts()
    # print("Total number of states : ",len(temp))
    trace = pgo.Bar(
        x=temp.index,
        y=(temp / temp.sum()) * 100,
    )
    data = [trace]
    layout = pgo.Layout(
        title=title,
        xaxis=dict(
            title='Name of type of the Suite',
            tickfont=dict(
                size=14,
                color='rgb(107, 107, 107)'
            )
        ),
        yaxis=dict(
            title='Count of Name of type of the Suite in %',
            titlefont=dict(
                size=16,
                color='rgb(107, 107, 107)'
            ),
            tickfont=dict(
                size=14,
                color='rgb(107, 107, 107)'
            )
        )
    )
    fig = pgo.Figure(data=data, layout=layout)
    po.iplot(fig, filename=series_name)

def value_count_pie(df,series_name,title="value count pie",hole=0.0):
    temp = df[series_name].value_counts()
    df = pd.DataFrame({'labels': temp.index,
                       'values': temp.values
                       })
    df.iplot(kind='pie', labels='labels', values='values', title=title,hole=hole)
def value_count_hole_pie(df, series_name,title="value count hole pie"):
    from plotly.offline import iplot
    temp = df[series_name].value_counts()
    fig = {
        "input": [
            {
                "values": temp.values,
                "labels": temp.index,
                "domain": {"x": [0, .48]},
                # "name": "Types of Loans",
                # "hoverinfo":"label+percent+name",
                "hole": .7,
                "type": "pie"
            },

        ],
        "layout": {
            "title": title,
            "annotations": [
                {
                    "font": {
                        "size": 20
                    },
                    "showarrow": False,
                    "text": series_name,
                    "x": 0.17,
                    "y": 0.5
                }

            ]
        }
    }
    iplot(fig, filename='donut')
def value_count_bar_with_target(df,series_name,target,title="value count with regard to target"):
    import plotly.graph_objs as pgo
    from plotly.offline import iplot
    temp = df["NAME_FAMILY_STATUS"].value_counts()
    # print(temp.values)
    temp_y0 = []
    temp_y1 = []
    for val in temp.index:
        temp_y1.append(np.sum(df[target][df[series_name] == val] == 1))
        temp_y0.append(np.sum(df[target][df[series_name] == val] == 0))
    trace1 = pgo.Bar(
        x=temp.index,
        y=(temp_y1 / temp.sum()) * 100,
        name='YES'
    )
    trace2 = pgo.Bar(
        x=temp.index,
        y=(temp_y0 / temp.sum()) * 100,
        name='NO'
    )
    data = [trace1, trace2]
    layout = pgo.Layout(
        title=title,
        # barmode='stack',
        width=1000,
        xaxis=dict(
            title=series_name,
            tickfont=dict(
                size=14,
                color='rgb(107, 107, 107)'
            )
        ),
        yaxis=dict(
            title='Count in %',
            titlefont=dict(
                size=16,
                color='rgb(107, 107, 107)'
            ),
            tickfont=dict(
                size=14,
                color='rgb(107, 107, 107)'
            )
        )
    )

    fig = pgo.Figure(data=data, layout=layout)
    iplot(fig)
def plot_feature_importances(df):
    """
    Plot importances returned by a model. This can work with any measure of
    feature importance provided that higher importance is better.

    Args:
        df (dataframe): feature importances. Must have the features in a column
        called `features` and the importances in a column called `importance

    Returns:
        shows a plot of the 15 most importance features

        df (dataframe): feature importances sorted by importance (highest to lowest)
        with a column for normalized importance
        """

    # Sort features according to importance
    df = df.sort_values('importance', ascending=False).reset_index()

    # Normalize the feature importances to add up to one
    df['importance_normalized'] = df['importance'] / df['importance'].sum()

    # Make a horizontal bar chart of feature importances
    plt.figure(figsize=(10, 6))
    ax = plt.subplot()

    # Need to reverse the index to plot most important on top
    ax.barh(list(reversed(list(df.index[:15]))),
            df['importance_normalized'].head(15),
            align='center', edgecolor='k')

    # Set the yticks and labels
    ax.set_yticks(list(reversed(list(df.index[:15]))))
    ax.set_yticklabels(df['feature'].head(15))

    # Plot labeling
    plt.xlabel('Normalized Importance');
    plt.title('Feature Importances')
    plt.show()

    return df