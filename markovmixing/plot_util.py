""" Make good looking plots.
"""
import numpy

def pyplot_bar(y, cmap='Blues'):
	""" Make a good looking pylot bar plot.

	Use a colormap to color the bars.

	y: height of bars
	cmap: colormap, defaults to 'Blues'
	"""
	import matplotlib.pyplot as plt
	
	from matplotlib.colors import Normalize
	from matplotlib.cm import ScalarMappable

	vmax = numpy.max(y)
	vmin = (numpy.min(y)*3. - vmax)/2.
	
	colormap = ScalarMappable(norm=Normalize(vmin, vmax), cmap='Blues')

	plt.bar(numpy.arange(len(y)), y, color=colormap.to_rgba(y), align='edge', width=1.0)