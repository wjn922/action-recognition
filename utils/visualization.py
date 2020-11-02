import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt


def calculate_cam(feature_conv, weight_fc):
	"""
	feature_conv (1, 1024, T, N, N) and weight_fc (1024, 1, 1, 1) both numpy data
	"""
	feature_conv = np.squeeze(feature_conv).transpose((1,2,3,0))   # (T, N, N, 1024)
	weight_fc = np.squeeze(weight_fc)		# (1024)
	T, H, W, C = feature_conv.shape
	
	cam = np.zeros(dtype=np.float32, shape=(T,H,W))
	for i, w in enumerate(weight_fc[:]):
		cam += w * feature_conv[:, :, :, i]

	# normalization
	cam -= np.min(cam)
	cam /= np.max(cam)  
	# cam = np.uint8(255 * cam)
	# cam = 100 - cam
	return cam  # (T, H, W)


def calculate_optical_flow(video_path):
	"""
	Calculate the optical magnitude of a video.
	Returns motion list of this video.
	"""
	motion_list = dict()

	cap = cv2.VideoCapture(video_path)
	success, frame1 = cap.read()
	prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
	hsv = np.zeros_like(frame1)
	hsv[...,1] = 255

	frame_count = 1
	while success:
		frame_count += 1
		success, frame2 = cap.read()
		if not success and frame2 is None:
			continue
		
		next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
		flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

		mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
		hsv[...,0] = ang*180/np.pi/2
		hsv[...,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
		bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

		motion = mag.mean()
		motion_list.update({frame_count: motion})

		"""
		cv2.imshow('frame2', bgr)
		k = cv2.waitKey(30) & 0xff
		if k == 27:
			break
		elif k == ord('s'):
			cv2.imwrite('opticalfb.png', frame2)
			cv2.imwrite('opticalhsv.png', bgr)
		"""
		prvs = next

	cap.release()
	cv2.destroyAllWindows()

	return motion_list





