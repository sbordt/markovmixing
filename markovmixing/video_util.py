""" Make videos.
"""

def matplotlib_plots_to_video(path, ffigure, n):
	""" Make a video from a sequence of matplotlib figures.

	The function ffigure will be called subsequently to retrieve
	the sequence of matplotlib figures.

	The figure must have been created using 'plt.figure(figsize=(19.20, 10.80), dpi=100)'.

	The will video consists of n frames (or figures), where 100 frames correspond to 1 second.

	The video will have a resolution of 1920*1080.

	path: path of the video file (*.avi)
	ffigure: callback function that takes the frame number as a single argument
	n: number of frames
	"""
	import cv2, io
	import numpy as np
	import matplotlib.pyplot as plt
	
	from PIL import Image

	print "INFO: Creating a video that lasts "+`n/100.0`+" seconds."

	# Define the codec and create VideoWriter object
	fourcc = cv2.cv.CV_FOURCC(*'FMP4')
	out = cv2.VideoWriter(path, fourcc, 100.0, (1920,1080))

	for i in xrange(n):
		# get the next figure
		fig = ffigure(*(i,))
    	
		# figure to png in memory
		buf = io.BytesIO()
		fig.savefig(buf, format="png", dpi=100)
		buf.seek(0)

		# png in memory to PIL image
		pil_image = Image.open(buf).convert('RGB')

		# PIL image to numpy BGR array
		open_cv_image = np.array(pil_image) 

		# Convert RGB to BGR 
		open_cv_image = open_cv_image[:, :, ::-1].copy() 	
       
    	# write the frame
		out.write(open_cv_image)

		# close buffer 
		buf.close()

    	# clear figure
		plt.clf()
		plt.close()

		if i % 50 == 0 and i != 0:
			print `i/100.0`+" second(s) completed."

	# Release everything if job is finished
	out.release()
	cv2.destroyAllWindows()

	print "Video created."