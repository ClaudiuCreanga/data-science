import numpy as np
import pandas as pd
import bokeh.plotting as bk

np.random.seed(0)
my_df = pd.DataFrame(
    ["Signature 1A", "7", "11.7%", "Age"]
    ["Signature 1B", "19", "60.7%", "Age"]
    ["Signature 2", "16", "14.4%", "APOBEC"]
    ["Signature 3", "3", "9.9%", "BRCA1/2 mutations"]
    ["Signature 4", "5", "12.1%", "Smoking"]
    ["Signature 5", "9", "14.4%"]
    ["Signature 6", "9", "2.6%", "DNA MMR deficiency"]
    ["Signature 7", "2", "5.0%", "Ultraviolet light (head and neck, melanoma)"]
    ["Signature 8", "2", "2.0%", "(breast, medulloblastoma)"]
    ["Signature 9", "2", "0.6%", "Immunoglobulin gene hypermutation (CLL, lymphoma B cell)"]
)

p = bk.figure(
    title = "Contribution from each signature",
    tools="pan,box_zoom,reset,save"
)

p.vbar(my_df, width=0.2, top=my_df[0], color=["#e31a1c", "#6a3d9a", "#f78071", "#8cd3c7", "#edf8b1", "#2877b4", "#f67f04", "#666666", "#f1b6da"])
bk.show(p)

