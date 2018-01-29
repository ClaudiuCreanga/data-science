import pandas as pd
from bokeh.plotting import figure
from bokeh.io import show

df = pd.DataFrame(
    {
        "kpi0": [3],
        "kpi1": [4],
        "kpi2": [5]
    }
)

p = figure()

p.vbar(df.iloc[0], width=0.2, top=df.iloc[0])

show(p)