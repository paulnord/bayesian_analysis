#!/opt/anaconda3/bin/python3
import pickle
import pystan
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
from plot_trace import plot_trace


def theory_curve(x, A0, t0, A1, t1, background, C=0):
    y0 = A0 - A0*0.5**(x/t0)
    y1 = A1 - A1*0.5**(x/t1)
    b = background*x
    answer = y0 + y1 + b + C
    return answer


## Diagnostics #################################################################

def replot():

    sm = pickle.load(open("model.p","rb"))
    fit = pickle.load(open("fit.p","rb"))
    data = pickle.load(open("data.p","rb"))

    x = data['a']
    y = data['d']

    summary_dict = fit.summary()
    df = pd.DataFrame(summary_dict['summary'], 
                      columns=summary_dict['summary_colnames'], 
                      index=summary_dict['summary_rownames'])

    A0_mean, t0_mean = df['mean']['A0'], df['mean']['t0']
    A1_mean, t1_mean = df['mean']['A1'], df['mean']['t1']
    background_mean = df['mean']['background']
    try:
        C_mean = df['mean']['C']
    except:
        C_mean = 0.0

    print(df)

    # Extracting traces
    A0 = fit['A0']
    t0 = fit['t0']
    A1 = fit['A1']
    t1 = fit['t1']
    try:
        C = fit['C']
    except:
        C = [0]*len(A0)
    background = fit['background']
    start = [0] * len(A0)
    lp = fit['lp__']

    # Plotting regression line
    x_min, x_max = min(x)*0.9, max(x)*1.1
    x_plot = np.logspace(.1, 4, 100)-1

    #plt.xscale("log")
    #plt.yscale("log")
    plt.xlim([min(x_plot)*0.9,max(x_plot)*1.1])
    plt.ylim([min(y)*0.9,max(y)*1.1])
    #plt.rcParams['axes.facecolor'] = 'black'

    # Plot a subset of sampled regression lines
    for i in np.random.randint(0, len(A0), 1000):
      plt.plot(x_plot, theory_curve(x_plot,A0[i],t0[i],A1[i],t1[i],background[i],C[i]), color='lightsteelblue', 
               alpha=0.005, linewidth=1)

    # Plot mean regression line
    plt.plot(x_plot, theory_curve(x_plot,A0_mean,t0_mean,A1_mean,t1_mean,background_mean,C_mean))
    plt.scatter(x, y)

    plt.xlabel('$time     (minutes)$')
    plt.ylabel('$counts$')
    plt.title('Fitted Theory Curve')
    plt.xlim(x_min, x_max)

    plt.show()


    plt.subplot(211)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlim([min(x_plot)*0.9,max(x_plot)*1.1])
    plt.ylim([min(y)*0.9,max(y)*1.1])
    #plt.rcParams['axes.facecolor'] = 'black'

    # Plot a subset of sampled regression lines
    for i in np.random.randint(0, len(A0), 1000):
      plt.plot(x_plot, theory_curve(x_plot,A0[i],t0[i],A1[i],t1[i],background[i],C[i]), color='lightsteelblue', 
               alpha=0.005, linewidth=1)

    # Plot mean regression line
    plt.plot(x_plot, theory_curve(x_plot,A0_mean,t0_mean,A1_mean,t1_mean,background_mean,C_mean))
    #print(list(zip(x_plot, theory_curve(x_plot,A0_mean,t0_mean,A1_mean,t1_mean,background_mean))))

    plt.scatter(x, y)

    plt.xlabel('$time     (minutes)$')
    plt.ylabel('$counts$')
    plt.title('Fitted Theory Curve')
    plt.xlim(x_min, x_max)

    plt.subplot(212)
    plt.xlabel('$time     (minutes)$')
    plt.ylabel('$residual$')
    residuals = y - theory_curve(x,A0_mean,t0_mean,A1_mean,t1_mean,background_mean,C_mean)
    plt.xscale("log")
    plt.scatter(x,residuals)
    plt.show()

    for index, row in df.iterrows():
        print(index)
        plot_trace(fit[index],index)
        plt.show()

if __name__ == '__main__':
    replot()
