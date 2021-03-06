#add your imports here
from sys import argv
from glob import glob
from scipy import misc
from skimage import io, img_as_float
from segment import label_components, segment_components_by_label
import imageutil as util
from symrec import EquationFCN
import numpy as np
import shelve
"""
add whatever you think it's essential here
"""


class SymPred():
	def __init__(self,prediction, x1, y1, x2, y2):
		"""
		<x1,y1> <x2,y2> is the top-left and bottom-right coordinates for the bounding box
		(x1,y1)
			   .--------
			   |	   	|
			   |	   	|
			    --------.
			    		 (x2,y2)
		"""
		self.prediction = prediction
		self.x1 = x1
		self.y1 = y1
		self.x2 = x2
		self.y2 = y2
	def __str__(self):
		return self.prediction + '\t' + '\t'.join([
												str(self.x1),
												str(self.y1),
												str(self.x2),
												str(self.y2)])

class ImgPred():
	def __init__(self,image_name,sym_pred_list,latex = 'LATEX_REPR'):
		"""
		sym_pred_list is list of SymPred
		latex is the latex representation of the equation 
		"""
		self.image_name = image_name
		self.latex = latex
		self.sym_pred_list = sym_pred_list
	def __str__(self):
		res = self.image_name + '\t' + str(len(self.sym_pred_list)) + '\t' + self.latex + '\n'
		for sym_pred in self.sym_pred_list:
			res += str(sym_pred) + '\n'
		return res


def predict(fcn, image_path):
	equation_image = io.imread(image_path, as_grey=True)
	image, cbs, label = label_components(equation_image)
	components = segment_components_by_label(equation_image, cbs, label)
	size = len(components)
	for i in range(size):
		components[i] = util.normalize(img_as_float(components[i]))
	categories = fcn.test_custom(components)
	# build ImgPred
	label_arr = ['A', 'tan', ')', '(', '+', '-', 'sqrt', '1', '0', '3', '2', '4', '6', 'mul', 'pi', '=', 'sin', 'pm', 'dots', 'frac', 'cos', 'delta', 'a', 'c', 'b', 'bar', 'd', 'f', 'i', 'h', 'k', 'm', 'o', 'n', 'p', 's', 't', 'y', 'x', 'div', 'bg']
	sym_pred_list = []

	for i in range(size):
		sympred = SymPred(label_arr[categories[i]], cbs[i][0], cbs[i][1], cbs[i][2], cbs[i][3])

		# check '='
		if label_arr[categories[i]] == '-':
			flag = False
			index_todelete = 0
			new_sympred = None
			for j in range(0,len(sym_pred_list)):
				item = sym_pred_list[j]
				if item.prediction == '-' and abs((item.x1 - cbs[i][0]) - (item.x2 - cbs[i][2])) < 5 and abs(item.y1 - cbs[i][1]) < 5 and abs(item.y2 - cbs[i][3]) < 5:
					#condition match
					index_todelete = j
					new_x1 = min(item.x1, cbs[i][0])
					new_y1 = min(item.y1, cbs[i][1])
					new_x2 = max(item.x2, cbs[i][2])
					new_y2 = max(item.y2, cbs[i][3])

					new_sympred = SymPred('=',new_x1, new_y1, new_x2, new_y2)
					flag = True
					break

			if flag == True:
				del sym_pred_list[index_todelete]
				sym_pred_list.append(new_sympred)
			else:
				sym_pred_list.append(sympred)

		else:
			# this is not '-'
			sym_pred_list.append(sympred)



	img_prediction = ImgPred(image_path, sym_pred_list)
	return img_prediction


if __name__ == '__main__':
	image_folder_path = argv[1]
	isWindows_flag = False
	if len(argv) == 3:
		isWindows_flag = True
	if isWindows_flag:
		image_paths = glob(image_folder_path + '\\*png')
	else:
		image_paths = glob(image_folder_path + '/*png')
	# image_paths = ["images/equations/SKMBT_36317040717260_eq8.png"]
	results = []
	fcn = EquationFCN(False)
	fcn.restore_model()
	count = 0
	for image_path in image_paths:
		impred = predict(fcn, image_path)
		results.append(impred)
		count += 1
		print(count)

	with open('predictions.txt','w') as fout:
		for res in results:
			fout.write(str(res))