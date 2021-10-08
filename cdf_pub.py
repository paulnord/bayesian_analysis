#!/opt/anaconda3/bin/python3
import pystan
import pandas as pd
import pickle

import multiprocessing
multiprocessing.set_start_method("fork")

model = """
data {
    int<lower=0> N;
    vector[N] a;
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
        d[n] ~ poisson( A0 + A1 -  A0*(0.5^((a[n])/t0)) - A1*(0.5^((a[n])/t1)) + background*a[n] );
    }
}
"""

def main():

    df = pd.read_csv("my_data.csv")
    x = df["t"].to_numpy()
    y = df["count"].to_numpy()

    # Put our data in a dictionary
    data = {'N': len(x), 'a': x, 'd': y}

    # Compile the model
    sm = pystan.StanModel(model_code=model)

    # Train the model and generate samples

    fit = sm.sampling(data=data, iter=2000, chains=5, warmup=1000, thin=1, seed=91801, n_jobs=4, control=dict(adapt_delta=0.9))

    pickle.dump( sm, open( "model.p", "wb" ) )
    pickle.dump( fit, open( "fit.p", "wb" ) )
    pickle.dump( data, open( "data.p", "wb" ) )

## Diagnostics #################################################################

    print(fit)

if __name__ == '__main__':
    main()
