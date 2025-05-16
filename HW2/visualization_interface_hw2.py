import matplotlib.pyplot as plt
import seaborn as sns


def visual_histogram(x,num,color):
    plt.hist(x, bins= num, histtype='barstacked', color=color, edgecolor='cyan', alpha=1)
    plt.xlabel(x.name)
    plt.ylabel('quantity')
    plt.title("Coffee Shop Dataset")
    plt.show()

def visual_kde(df,x):
    sns.displot(df, x=x , kind="kde", bw_adjust=1.2, fill=True)
    plt.show()

def visual_distribution(df,x,y):
    sns.relplot(data=df, x=x, y=y)
    plt.show()

def visual_fancy_scatterplot(df,x,y,size,hue):
    sns.set_theme(style="whitegrid")
    cmap = sns.cubehelix_palette(rot=-.2, as_cmap=True)
    g = sns.relplot(
        data=df.head(50),
        x=x, y=y, size=size,
        hue=hue,
        palette=cmap, sizes=(10, 500),
    )
    g.set(xscale="log", yscale="log")
    g.ax.xaxis.grid(True, "minor", linewidth=.25)
    g.ax.yaxis.grid(True, "minor", linewidth=.25)
    g.despine(left=True, bottom=True)
    plt.show()

def visual_boxplot(df,x,color):
    sns.boxplot(data=df, x=x, color=color, linecolor="#137", linewidth=.75)
    plt.show()

def visual_predictions(y_true, y_pred, num_points ,xlabel,ylabel,title):
    plt.figure(figsize=(10, 6))
    plt.scatter(range(num_points), y_true[:num_points], color='green', label='Истинные значения')
    plt.scatter(range(num_points), y_pred[:num_points], color='orange', label='Предсказанные значения')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title+"первые  "+str(num_points)+"  точек")
    plt.legend()
    plt.show()
