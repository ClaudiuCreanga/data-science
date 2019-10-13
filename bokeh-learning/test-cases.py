from bokeh.plotting import figure, output_file, show
import numpy as np

x = np.arange(1,6)
y = np.random.randint(7, size=5)

output_file("bokeh-learning/output/lines.html")

p = figure(title = "line graph", x_axis_label = "x", y_axis_label = "y")

p.line(x, y, legend="Career", line_width = 2)

show(p)