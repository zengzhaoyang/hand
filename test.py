#-*-coding:utf-8-*-

import cv2
import numpy as np

res = []
res2 = []
res3 = []
res4 = []
res5 = []
maxscore = []
maxscore2 = []
maxscore3 = []
maxscore4 = []
maxscore5 = []
for i in range(1200):
	res.append(None)
	res2.append(None)
	res3.append(None)
	res4.append(None)
	res5.append(None)
	maxscore.append(0)
	maxscore2.append(0)
	maxscore3.append(0)
	maxscore4.append(0)
	maxscore5.append(0)

fs = ['hand', 'ok', 'shi', 'v', 'zan', 'quan']
mapp = {
	'quan':1,
	'zan':2,
	'shi':3,
	'v':4,
	'ok':5,
	'hand':6
}
for name in fs:
	f = open('test_result/' + name + '_VGGM.txt')
	idd = mapp[name]
	for line in f:
		tmp = line.split()
		num = int(tmp[0])
		score = float(tmp[1])
		if maxscore[num] > 0.076 and name == 'quan':
			continue
		if score > maxscore[num] and float(tmp[3]) > 24 and float(tmp[5]) < 432 and float(tmp[4]) < 608:
			# img = cv2.imread('color_image/%d.jpg'%int(tmp[0]))
			# black_cnt = 0
			# for i in range(int(float(tmp[2])), int(float(tmp[4]))):
			# 	for j in range(int(float(tmp[3])), int(float(tmp[5]))):
			# 		if sum(img[j, i]) < 30:
			# 			black_cnt += 1
			# if black_cnt * 1.0 / (float(tmp[4]) - float(tmp[2])) / (float(tmp[5]) - float(tmp[3])) > 0.3:
			# 	continue
			res[num] = ((idd, float(tmp[2]), float(tmp[3]), float(tmp[4]), float(tmp[5])))
			maxscore[num] = score
		# else:
		# 	if float(tmp[3]) > 24 and float(tmp[5]) < 432 and float(tmp[4]) < 608:
		# 		if (float(tmp[5]) - float(tmp[3])) * (float(tmp[4]) - float(tmp[2])) < (res[num][4] - res[num][2]) * (res[num][3] - res[num][1]):
		# 			res[num] = ((idd, float(tmp[2]), float(tmp[3]), float(tmp[4]), float(tmp[5])))
		# 			maxscore[num] = score

	f.close()

	f = open('test_result/' + name + '_VGG.txt')
	idd = mapp[name]
	for line in f:
		tmp = line.split()
		num = int(tmp[0])
		score = float(tmp[1])
		if maxscore[num] > 0.09 and name == 'quan':
			continue
		if score > maxscore2[num] and float(tmp[3]) > 24 and float(tmp[5]) < 432 and float(tmp[4]) < 608:
			res2[num] = ((idd, float(tmp[2]), float(tmp[3]), float(tmp[4]), float(tmp[5])))
			maxscore2[num] = score

	f.close()

	f = open('test_result/' + name + '_RES.txt')
	idd = mapp[name]
	for line in f:
		tmp = line.split()
		num = int(tmp[0])
		score = float(tmp[1])
		if score > maxscore3[num] and float(tmp[3]) > 24 and float(tmp[5]) < 432 and float(tmp[4]) < 608:
			res3[num] = ((idd, float(tmp[2]), float(tmp[3]), float(tmp[4]), float(tmp[5])))
			maxscore3[num] = score

	f.close()

	f = open('test_result/' + name + '_VGGM_FLIP.txt')
	idd = mapp[name]
	for line in f:
		tmp = line.split()
		num = int(tmp[0])
		score = float(tmp[1])
		if score > maxscore4[num] and float(tmp[3]) > 24 and float(tmp[5]) < 432 and float(tmp[4]) < 608:
			res4[num] = ((idd, 640 - float(tmp[4]), float(tmp[3]), 640 - float(tmp[2]), float(tmp[5])))
			maxscore4[num] = score

	f.close()

	f = open('test_result/' + name + '_VGG_FLIP.txt')
	idd = mapp[name]
	for line in f:
		tmp = line.split()
		num = int(tmp[0])
		score = float(tmp[1])
		if maxscore[num] > 0.09 and name == 'quan':
			continue
		if score > maxscore5[num] and float(tmp[3]) > 24 and float(tmp[5]) < 432 and float(tmp[4]) < 608:
			res5[num] = ((idd, 640 - float(tmp[4]), float(tmp[3]), 640 - float(tmp[2]), float(tmp[5])))
			maxscore5[num] = score

	f.close()


for i in range(1200):
	# print i
	img = cv2.imread('color_image/%d.jpg'%i)
	width = img.shape[1]
	height = img.shape[0]
	cnt = 0

	if res[i] == None:
		if res2[i] == None or (maxscore2[i] > 0.71 and maxscore2[i] < 0.82) or maxscore2[i] < 0.61:
			if res3[i] == None or (maxscore3[i] > 0.85 and maxscore3[i] < 0.95) or maxscore3[i] < 0.6:
				if res4[i] == None or (maxscore4[i] > 0.8 and maxscore4[i] < 0.95) or maxscore4[i] < 0.43:
					if res5[i] == None or (maxscore5[i] > 0.5 and maxscore5[i] < 0.95) or maxscore5[i] < 0.44:
						print '%d 0 0.0 0.0 0.0 0.0'%i
					else:
						print '%d %d %.6f %.6f %.6f %.6f'%(i, res5[i][0], res5[i][1] / width, res5[i][2] / height, res5[i][3] / width, res5[i][4] / height)
						cv2.rectangle(img, (int(res5[i][1]), int(res5[i][2])), (int(res5[i][3]), int(res5[i][4])), (0, 255, 0))
				else:
					print '%d %d %.6f %.6f %.6f %.6f'%(i, res4[i][0], res4[i][1] / width, res4[i][2] / height, res4[i][3] / width, res4[i][4] / height)
					cv2.rectangle(img, (int(res4[i][1]), int(res4[i][2])), (int(res4[i][3]), int(res4[i][4])), (0, 255, 0))
			else:
				print '%d %d %.6f %.6f %.6f %.6f'%(i, res3[i][0], res3[i][1] / width, res3[i][2] / height, res3[i][3] / width, res3[i][4] / height)
				cv2.rectangle(img, (int(res3[i][1]), int(res3[i][2])), (int(res3[i][3]), int(res3[i][4])), (0, 255, 0))
		else:
			print '%d %d %.6f %.6f %.6f %.6f'%(i, res2[i][0], res2[i][1] / width, res2[i][2] / height, res2[i][3] / width, res2[i][4] / height)
			cv2.rectangle(img, (int(res2[i][1]), int(res2[i][2])), (int(res2[i][3]), int(res2[i][4])), (0, 255, 0))
	else:
		a = res[i][0]
		b = res[i][1]
		c = res[i][2]
		d = res[i][3]
		e = res[i][4]
		maxs = maxscore[i]
		if res2[i] != None:
			if maxscore2[i] > maxs + 0.2:
				maxs = maxscore2[i]
				a = res2[i][0]
				b = res2[i][1]
				c = res2[i][2]
				d = res2[i][3]
				e = res2[i][4]

		if res4[i] != None:
			if maxscore4[i] > maxs + 0.2:
				maxs = maxscore4[i]
				a = res4[i][0]
				b = res4[i][1]
				c = res4[i][2]
				d = res4[i][3]
				e = res4[i][4]

		if maxs < 0.085:
			print '%d 0 0.0 0.0 0.0 0.0'%i
			continue
		if maxscore[i] < 0.14 and res2[i] == None:
			print '%d 0 0.0 0.0 0.0 0.0'%i
			continue


		print '%d %d %.6f %.6f %.6f %.6f'%(i, a, b / width, c / height, d / width, e / height)
		cv2.rectangle(img, (int(b), int(c)), (int(d), int(e)), (0, 255, 0))
	cv2.imwrite('result_mix/%d.jpg'%i, img)