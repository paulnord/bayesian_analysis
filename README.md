RUN the MCMC analysis of the data in my_data.csv

python cdf_pub.py  


Make some plots of what you've done

python replot.py

Alternate analysis method cdf_stepwise.py uses differential counts expected in a particular interval.
This seems to get similar answers.  It's unclear which method is better.

cdf_stepwise.py method:
<pre>d[n] ~ poisson( A0*(0.5^((start[n])/t0)) + A1*(0.5^((start[n])/t1)) - A0*(0.5^((stop[n])/t0)) - A1*(0.5^((stop[n])/t1)) + background*delta_t );</pre>


cdf_pub.py method:
<pre>count[n] ~ poisson( A0 + A1 -  A0*(0.5^((stop[n])/t0)) - A1*(0.5^((stop[n])/t1)) + background*stop[n] );</pre>
