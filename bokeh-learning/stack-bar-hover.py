import numpy as np
import pandas as pd
import bokeh.plotting as bk
from bokeh.models import HoverTool

my_df = pd.DataFrame(
    {
        "Cancertypes": [7,19,16,3,5,9,9,2,2,2],
        "Prevalence": [11,60,14,9,12,14,2,5,2,1],
        "ProbableAssociation": ["Age","Age","APOBEC", "Smoking","Smoking","Smoking","Smoking","Smoking","Smoking","Smoking"],
        "Signature": ["Signature 1A","Signature 1B","Signature 2","Signature 3","Signature 4","Signature 5","Signature 6","Signature 7","Signature 8","Signature 9"]
    }
)

source = bk.ColumnDataSource(my_df)

hover = HoverTool(
    tooltips = [
        ("Prevalence", "@Prevalence"),
        ("Cancer types", "@Cancertypes"),
        ("Probable Association", "@ProbableAssociation")
    ]
)

p = bk.figure(
    title = "Operative mutational signatures",
    tools=[hover],
    x_axis_label = "Signatures",
    y_axis_label = "Contribution from each signature",
    x_range=list(my_df["Signature"].values),
    plot_width=1000
)

p.vbar(
    x="Signature",
    width=0.4,
    top="Prevalence",
    source=source
)

bk.show(p)

