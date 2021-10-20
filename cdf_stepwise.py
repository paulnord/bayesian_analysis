#!/opt/anaconda3/bin/python3
import sys
import pystan
import pandas as pd
import pickle
import numpy as np

import multiprocessing
multiprocessing.set_start_method("fork")

model = """
data {
    int<lower=0> N;
    vector[N] start;
    vector[N] stop;
    int d[N];
}
parameters {
    real<lower=0> A0;  // Initial number of atoms a 
    real<lower=0.0, upper=20> t0;  // Half Life a
    real<lower=0> A1;  // Initial number of atoms b
    real<lower=400, upper=1000> t1;  // Half Life b
    real<lower=0, upper=20> background;
}
model {
    for(n in 1:N) {
        real delta_t = stop[n] - start[n];
        d[n] ~ poisson( A0*(0.5^((start[n])/t0)) + A1*(0.5^((start[n])/t1)) - A0*(0.5^((stop[n])/t0)) - A1*(0.5^((stop[n])/t1)) + background*delta_t );
    }
}
"""

def main(filename):

    df = pd.read_csv(filename+".csv")
    t = df["t"].to_numpy()
    count = df["count"].to_numpy()
    d = [0]*len(count)

    x2 = t
    x1 = np.insert(t, [0], [0.0])[0:len(t)]


    d[0] = count[0]
    for i in range(1,len(t)):
        d[i] = count[i] - count[i-1]

    # Put our data in a dictionary
    data = {'N': len(t), 'start': x1, 'stop': x2, 'd': d, 'count': count}

    # Compile the model
    sm = pystan.StanModel(model_code=model)

    # Train the model and generate samples

    fit = sm.sampling(data=data, iter=2000, chains=5, warmup=1000, thin=1, seed=91801, n_jobs=4, control=dict(adapt_delta=0.9,max_treedepth=15))

    pickle.dump( sm, open( filename+"_model.p", "wb" ) )
    pickle.dump( fit, open( filename+"_fit.p", "wb" ) )
    pickle.dump( data, open( filename+"_data.p", "wb" ) )

## Diagnostics #################################################################

    print(fit)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        filename = 'my_data'
    print(filename)
    main(filename)
