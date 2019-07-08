import numpy as np
import tensorflow as tf

from constants import overlap_thresh, max_boxes

def non_max_suppression_fast(boxes, probs):
	# code used from here: http://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
	# if there are no boxes, return an empty list
	if len(boxes) == 0:
		return []

	# grab the coordinates of the bounding boxes
	x1 = boxes[:, 0]
	y1 = boxes[:, 1]
	x2 = boxes[:, 2]
	y2 = boxes[:, 3]

	np.testing.assert_array_less(x1, x2)
	np.testing.assert_array_less(y1, y2)

	# if the bounding boxes integers, convert them to floats --
	# this is important since we'll be doing a bunch of divisions
	if boxes.dtype.kind == "i":
		boxes = boxes.astype("float")

	# initialize the list of picked indexes	
	pick = []

	# calculate the areas
	area = (x2 - x1) * (y2 - y1)

	# sort the bounding boxes 
	idxs = np.argsort(probs)

	# keep looping while some indexes still remain in the indexes
	# list
	while len(idxs) > 0:
		# grab the last index in the indexes list and add the
		# index value to the list of picked indexes
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)

		# find the intersection

		xx1_int = np.maximum(x1[i], x1[idxs[:last]])
		yy1_int = np.maximum(y1[i], y1[idxs[:last]])
		xx2_int = np.minimum(x2[i], x2[idxs[:last]])
		yy2_int = np.minimum(y2[i], y2[idxs[:last]])

		ww_int = np.maximum(0, xx2_int - xx1_int)
		hh_int = np.maximum(0, yy2_int - yy1_int)

		area_int = ww_int * hh_int

		# find the union
		area_union = area[i] + area[idxs[:last]] - area_int

		# compute the ratio of overlap
		overlap = area_int/(area_union + 1e-6)

		# delete all indexes from the index list that have
		idxs = np.delete(idxs, np.concatenate(([last],
			np.where(overlap > overlap_thresh)[0])))

		if len(pick) >= max_boxes:
			break

	# return only the bounding boxes that were picked using the integer data type
	boxes = boxes[pick].astype("int")
	probs = probs[pick]
	
	return boxes, probs

def rpn_to_roi(rpn_input, regr_layer):
	anchor_sizes = [8]
	anchor_ratios = [[1, 1]]

	assert rpn_input.shape[0] == 1

	rpn_layer = tf.map_fn(lambda x: tf.math.sigmoid(x), rpn_input).numpy()

	(rows, cols) = rpn_layer.shape[1:3]
	(rows, cols) = (int(rows), int(cols))

	curr_layer = 0
	A = np.zeros((4, rpn_layer.shape[1], rpn_layer.shape[2], rpn_layer.shape[3]))

	for anchor_size in anchor_sizes:
		for anchor_ratio in anchor_ratios:

			anchor_x = (anchor_size * anchor_ratio[0])
			anchor_y = (anchor_size * anchor_ratio[1])
			# regr = regr_layer[0, 4 * curr_layer:4 * curr_layer + 4, :, :]

			X, Y = np.meshgrid(np.arange(cols), np.arange(rows))

			A[0, :, :, curr_layer] = np.maximum(X - anchor_x//2, 0)
			A[1, :, :, curr_layer] = np.maximum(Y - anchor_y//2, 0)
			A[2, :, :, curr_layer] = np.minimum(X + anchor_x//2, cols-1)
			A[3, :, :, curr_layer] = np.minimum(Y + anchor_y//2, rows-1)

			curr_layer += 1

	all_boxes = np.reshape(A.transpose((0, 3, 1, 2)), (4, -1)).transpose((1, 0))
	all_probs = np.reshape(rpn_layer.transpose((0, 3, 1, 2)), (-1))

	boxes, probs = non_max_suppression_fast(all_boxes, all_probs)

	return boxes, probs

def roi_pooling(input_features, input_roi):
	features_maps = []
	np_features = input_features.numpy()
	for roi in input_roi:
		x1 = roi[0]
		y1 = roi[1]
		x2 = roi[2]
		y2 = roi[3]
		specific_features = np_features[:, x1:x2, y1:y2, :]
		specific_features = tf.image.resize(specific_features, (8, 8))
		features_maps.append(specific_features)
	return features_maps
