"""Second exploratory round, motivated by first-round losses. Not confirmatory."""
from pathlib import Path
import numpy as np
from scipy.ndimage import maximum_filter1d, gaussian_filter1d
import screen

OUT=Path(__file__).resolve().parent/"round2"
OUT.mkdir(exist_ok=True)
N=3000
SCENARIOS=["original_sinusoid","growing_sinusoid_50","growing_sinusoid_1000",
           "quadratic_sinusoid","smooth_level_blocks","smooth_multiscale_blocks",
           "smooth_triangles","growing_triangles","smooth_chirp","long_plateau_short_burst",
           "maxpooled_gaussian_cycles","maxpooled_drift","correlated_lognormal",
           "correlated_lognormal_drift","mean_bias_random_walk","recurrent_bias_shift"]

def generate(name,seed):
    r=np.random.default_rng(seed);t=np.arange(N);u=t/(N-1)
    noise=r.normal(0,.3,N)
    original=10*(np.sin(2*np.pi*(t+1)/200)+.5)+2
    triangle=20*np.abs(2*((t%200)/200)-1)
    if name=="original_sinusoid": s=original+noise
    elif name.startswith("growing_sinusoid"):
        end=int(name.split("_")[-1]);s=np.exp(np.log(end)*u)*np.maximum(0,original+noise)
    elif name=="quadratic_sinusoid":s=20*u*u+original+noise
    elif name=="smooth_level_blocks":s=np.repeat([1,20,1,50,1],600)+noise
    elif name=="smooth_multiscale_blocks":s=np.repeat([1,10,2,100,3,30,1,300,10,1000],300)+noise
    elif name=="smooth_triangles":s=triangle+noise
    elif name=="growing_triangles":s=np.exp(np.log(50)*u)*(triangle+noise)
    elif name=="smooth_chirp":s=10*(np.sin(2*np.pi*(t/400+4*u*u))+.5)+2+noise
    elif name=="long_plateau_short_burst":s=np.where(t%500<400,1,100)+noise
    elif name=="maxpooled_gaussian_cycles":s=maximum_filter1d((2+1.9*np.sin(2*np.pi*t/200))*np.abs(r.normal(size=N)),25)
    elif name=="maxpooled_drift":s=maximum_filter1d(np.exp(np.log(30)*u)*np.abs(r.standard_t(3,N)),25)
    elif name=="correlated_lognormal":s=np.exp(gaussian_filter1d(r.normal(size=N),15)*8)
    elif name=="correlated_lognormal_drift":s=np.exp(np.log(30)*u+gaussian_filter1d(r.normal(size=N),15)*8)
    elif name=="mean_bias_random_walk":s=np.abs(np.cumsum(r.normal(0,1,N)))+.1*np.abs(noise)
    elif name=="recurrent_bias_shift":s=np.abs(np.repeat(r.normal(0,20,20),150)+noise)
    else:raise ValueError(name)
    return np.maximum(0,s)

if __name__=="__main__":
    screen.OUT=OUT;screen.SCENARIOS=SCENARIOS;screen.generate=generate
    screen.main()
