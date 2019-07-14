import numpy as np
import tensorflow as tf

from constants import overlap_thresh, max_boxes, anchor_size as s

# Most of this file comes from https://github.com/kbardool/keras-frcnn/blob/master/keras_frcnn/roi_helpers.py
# Probably a lot of transpositions are not necessary. I need to investigate this

def non_max_suppression_fast(boxes, probs):
	x1 = boxes[:, 0]
	y1 = boxes[:, 1]
	x2 = boxes[:, 2]
	y2 = boxes[:, 3]
	area = (x2 - x1) * (y2 - y1)

	np.testing.assert_array_less(x1, x2)
	np.testing.assert_array_less(y1, y2)

	picked_box_ids = []
	ids_sorted = np.argsort(probs)

	while (len(ids_sorted) > 0) & (len(picked_box_ids) < max_boxes):
		current_id = ids_sorted[-1]
		ids_sorted = ids_sorted[:-1]
		picked_box_ids.append(current_id)

		xx1_intersection = np.maximum(x1[current_id], x1[ids_sorted])
		yy1_intersection = np.maximum(y1[current_id], y1[ids_sorted])
		xx2_intersection = np.minimum(x2[current_id], x2[ids_sorted])
		yy2_intersection = np.minimum(y2[current_id], y2[ids_sorted])

		ww_intersection = np.maximum(0, xx2_intersection - xx1_intersection)
		hh_intersection = np.maximum(0, yy2_intersection - yy1_intersection)

		area_intersection = ww_intersection * hh_intersection

		area_union = area[current_id] + area[ids_sorted] - area_intersection

		overlap = area_intersection / area_union # division of ints resulting in a float

		ids_to_delete = np.where(overlap > overlap_thresh)[0]
		ids_sorted = np.delete(ids_sorted, ids_to_delete)

	picked_boxes = boxes[picked_box_ids].astype("int")
	picked_probs = probs[picked_box_ids]

	return picked_boxes, picked_probs

def apply_regr(X, T):
	x = X[0, :, :]
	y = X[1, :, :]
	w = X[2, :, :]
	h = X[3, :, :]

	tx = T[0, :, :]
	ty = T[1, :, :]
	tw = T[2, :, :]
	th = T[3, :, :]

	cx = x + w/2.
	cy = y + h/2.
	cx1 = tx * w + cx
	cy1 = ty * h + cy

	w1 = np.exp(tw.astype(np.float64)) * w
	h1 = np.exp(th.astype(np.float64)) * h
	x1 = cx1 - w1/2.
	y1 = cy1 - h1/2.

	x1 = np.round(x1)
	y1 = np.round(y1)
	w1 = np.round(w1)
	h1 = np.round(h1)
	return np.stack([x1, y1, w1, h1])

def remove_invalid_boxes(boxes, probs):
	x1 = boxes[:, 0]
	y1 = boxes[:, 1]
	x2 = boxes[:, 2]
	y2 = boxes[:, 3]
	invalid_ids = np.where((x1 - x2 >= 0) | (y1 - y2 >= 0))
	return np.delete(boxes, invalid_ids, 0), np.delete(probs, invalid_ids, 0)

def rpn_to_roi(rpn_input, regr_layer):
	# In this case, we only need one anchor and one anchor size. But faster-rcnn can use many
	anchor_sizes = [s]
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
			regr = regr_layer[0, :, :, 4 * curr_layer:4 * curr_layer + 4]
			regr = np.transpose(regr, (2, 0, 1))

			X, Y = np.meshgrid(np.arange(cols), np.arange(rows))

			A[0, :, :, curr_layer] = X - anchor_x//2
			A[1, :, :, curr_layer] = Y - anchor_y//2
			A[2, :, :, curr_layer] = anchor_x
			A[3, :, :, curr_layer] = anchor_y

			A[:, :, :, curr_layer] = apply_regr(A[:, :, :, curr_layer], regr)

			A[2, :, :, curr_layer] += A[0, :, :, curr_layer]
			A[3, :, :, curr_layer] += A[1, :, :, curr_layer]

			A[0, :, :, curr_layer] = np.maximum(A[0, :, :, curr_layer], 0)
			A[1, :, :, curr_layer] = np.maximum(A[1, :, :, curr_layer], 0)
			A[2, :, :, curr_layer] = np.minimum(A[2, :, :, curr_layer], cols-1)
			A[3, :, :, curr_layer] = np.minimum(A[3, :, :, curr_layer], rows-1)

			curr_layer += 1

	all_boxes = np.reshape(A.transpose((0, 3, 1, 2)), (4, -1)).transpose((1, 0))
	all_probs = np.reshape(rpn_layer.transpose((0, 3, 1, 2)), (-1))

	boxes, probs = remove_invalid_boxes(all_boxes, all_probs)
	boxes, probs = non_max_suppression_fast(boxes, probs)

	return boxes, probs
