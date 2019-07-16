import numpy as np
import tensorflow as tf

from constants import overlap_thresh, max_boxes, anchor_size as s, feature_size

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

def get_boxes(rpn_input):
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

			X, Y = np.meshgrid(np.arange(cols), np.arange(rows))

			A[0, :, :, curr_layer] = np.maximum(X - anchor_x//2, 0)
			A[1, :, :, curr_layer] = np.maximum(Y - anchor_y//2, 0)
			A[2, :, :, curr_layer] = np.minimum(X + anchor_x//2, cols) # The upper border can be equal to cols/row,
			A[3, :, :, curr_layer] = np.minimum(Y + anchor_y//2, rows) # in which case the box reaches the border of the mapping

			curr_layer += 1

	all_boxes = np.reshape(A.transpose((0, 3, 1, 2)), (4, -1)).transpose((1, 0))
	all_probs = np.reshape(rpn_layer.transpose((0, 3, 1, 2)), (-1))

	boxes, probs = non_max_suppression_fast(all_boxes, all_probs)

	return boxes, probs

def get_labels_boxes(boxes, target):
    labels_boxes = []
    for b in boxes:
        x1 = b[0]
        y1 = b[1]
        x2 = b[2]
        y2 = b[3]
        t = np.reshape(target[y1:y2, x1:x2], (-1))
        t = np.delete(t, np.where(t == 0))
        if (len(t) == 0):
            labels_boxes.append(0)
        else:
            (classes, occurences) = np.unique(t, return_counts=True)
            k = np.argmax(occurences)
            labels_boxes.append(classes[k])
    return labels_boxes

def get_final_box(b, regr):
	box_center_x = (b[2] + b[0]) / 2
	box_center_y = (b[3] + b[1]) / 2
	box_w = b[2] - b[0]
	box_h = b[3] - b[1]

	final_box_center_x = box_center_x + regr[0]
	final_box_center_y = box_center_y + regr[1]
	final_box_w = box_w + regr[2] # This is not the right correction for proper faster-rcnn
	final_box_h = box_h + regr[3] # But it is better suited to our case

	x1 = int(round(final_box_center_x - final_box_w / 2))
	x2 = int(round(final_box_center_x + final_box_w / 2))
	y1 = int(round(final_box_center_y - final_box_h / 2))
	y2 = int(round(final_box_center_y + final_box_h / 2))

	x1 = max(x1, 0)
	x2 = min(x2, feature_size)
	y1 = max(y1, 0)
	y2 = min(y2, feature_size)

	return x1, y1, x2, y2

def get_boxes_precision(boxes, regression_values, target):
	precision = []
	for i in range(len(boxes)):
		x1, y1, x2, y2 = get_final_box(boxes[i], regression_values[i])
		t = np.reshape(target[y1:y2, x1:x2], (-1))
		total_area = len(t)
		t = np.delete(t, np.where(t == 0))
		non_zero_area = len(t)
		precision.append([non_zero_area, total_area])
	return precision
