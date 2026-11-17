"""Parent-requested exploratory period sensitivity and 99% coverage screen."""
from pathlib import Path
import numpy as np
import screen

OUT=Path(__file__).resolve().parent/"round3"
OUT.mkdir(exist_ok=True)
N=3000
SCENARIOS=["period_50","period_100","period_200","period_400","period_800",
           "period_grows_100_600","period_shrinks_600_100","period_regime_100_600",
           "amplitude_period_grow","amplitude_period_shrink"]

def generate(name,seed):
    r=np.random.default_rng(seed);t=np.arange(N);u=t/(N-1)
    if name in ["period_50","period_100","period_200","period_400","period_800"]:
        p=float(name.split("_")[-1]); phase=(t+1)/p
    elif name in ["period_grows_100_600","amplitude_period_grow"]: phase=np.cumsum(1/(100+500*u))
    elif name in ["period_shrinks_600_100","amplitude_period_shrink"]: phase=np.cumsum(1/(600-500*u))
    elif name=="period_regime_100_600":phase=np.cumsum(1/np.where(t<1500,100,600))
    else:raise ValueError(name)
    s=np.maximum(0,10*(np.sin(2*np.pi*phase)+.5)+2+r.normal(0,.3,N))
    if name.startswith("amplitude"):s*=np.exp(np.log(30)*u)
    return s

if __name__=="__main__":
    screen.OUT=OUT;screen.SCENARIOS=SCENARIOS;screen.generate=generate
    screen.ALPHAS=[.0001,.00025,.0005,.001,.002,.003,.004,.005,.0075,.01,.015,.02,.025,.03,.035,.04,.045,.05,.06,.075,.1,.125,.15,.2,.25]
    screen.TARGETS=[.90,.95,.99]
    screen.main()
