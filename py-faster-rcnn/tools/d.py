import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms

import numpy as np
import caffe
import cv2
import re

xmin_r = re.compile('<xmin>(.*?)</xmin>')
xmax_r = re.compile('<xmax>(.*?)</xmax>')
ymin_r = re.compile('<ymin>(.*?)</ymin>')
ymax_r = re.compile('<ymax>(.*?)</ymax>')

CLASSES = ('__background__', 'plane')

NETS = 'ZF_faster_rcnn_final.caffemodel'

def demo(net, image_name):
	im = cv2.imread(image_name + '.jpg')
	scores, boxes = im_detect(net, im)

        f = open(image_name + '.xml')
        data = f.read()
        f.close()
        xmin = xmin_r.findall(data)
        ymin = ymin_r.findall(data)
        xmax = xmax_r.findall(data)
        ymax = ymax_r.findall(data)
        for i in range(len(xmin)):
            cv2.rectangle(im, (int(xmin[i]), int(ymin[i])), (int(xmax[i]), int(ymax[i])), (0, 0, 255), thickness=4)


	for cls_ind, cls in enumerate(CLASSES[1:]):
		cls_ind += 1
		cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind+1)]
		cls_scores = scores[:, cls_ind]
		dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
		keep = nms(dets, 0.3)
		dets = dets[keep, :]

		inds = np.where(dets[:, -1] >= 0.8)[0]

		if len(inds) > 0:
			for i in inds:
				bbox = dets[i, :4]
				score = dets[i, -1]
				cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), thickness=4)
				#cv2.putText(im, str(score), (bbox[0], bbox[1]), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 4, (255, 0, 0), thickness=4, lineType=8)
				cv2.imwrite(image_name + '_detect.jpg', im)

if __name__ == '__main__':
	cfg.TEST.HAS_RPN = True
	prototxt = 'models/pascal_voc/ZF/faster_rcnn_alt_opt/faster_rcnn_test.pt'
	caffemodel = '/home/jianf/workplace/faster_rcnn/py-faster-rcnn2/output/faster_rcnn_alt_opt/voc_2007_trainval/ZF_faster_rcnn_final.caffemodel'
	caffe.set_mode_gpu()
	caffe.set_device(0)

	net = caffe.Net(prototxt, caffemodel, caffe.TEST)
	im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
	for i in xrange(2):
		_, _ = im_detect(net, im)

	im_names =["/home/jianf/testdata/out1",
"/home/jianf/testdata/out19",
"/home/jianf/testdata/out1_19",
"/home/jianf/testdata/out1_29",
"/home/jianf/testdata/out1_39",
"/home/jianf/testdata/out1_48",
"/home/jianf/testdata/out1_57",
"/home/jianf/testdata/out1_66",
"/home/jianf/testdata/out1_75",
"/home/jianf/testdata/out1_84",
"/home/jianf/testdata/out2",
"/home/jianf/testdata/out29",
"/home/jianf/testdata/out2_18",
"/home/jianf/testdata/out2_27",
"/home/jianf/testdata/out2_36",
"/home/jianf/testdata/out2_45",
"/home/jianf/testdata/out2_54",
"/home/jianf/testdata/out2_63",
"/home/jianf/testdata/out35",
"/home/jianf/testdata/out3_11",
"/home/jianf/testdata/out3_20",
"/home/jianf/testdata/out3_3",
"/home/jianf/testdata/out3_39",
"/home/jianf/testdata/out3_48",
"/home/jianf/testdata/out3_57",
"/home/jianf/testdata/out3_67",
"/home/jianf/testdata/out3_76",
"/home/jianf/testdata/out3_85",
"/home/jianf/testdata/out3_94",
"/home/jianf/testdata/out43",
"/home/jianf/testdata/out4_4",
"/home/jianf/testdata/out53",
"/home/jianf/testdata/out62",
"/home/jianf/testdata/out71",
"/home/jianf/testdata/out80",
"/home/jianf/testdata/out9"]
	for im_name in im_names:
		print im_name
		demo(net, im_name)
