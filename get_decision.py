
import pandas as pd
import numpy as np
import math


def get_decision(data):
	# calc total entropy
	pt = len(data.loc[data['rating'] == 'Positive'].index)			# total number of positive rating
	nt = len(data.loc[data['rating'] == 'Negative'].index)			# total number of negative rating
	if pt == 0 or nt == 0:							# all cases have only positive or negative rating
		entropy_total = 0
	else:									# all cases have both positive or negative rating
		entropy_total = -1 * (pt / (pt + nt) * math.log2(pt / (pt + nt)) + nt / (pt + nt) * math.log2(nt / (pt + nt)))

	# calc entropy and gain of each word
	calculations = np.zeros(shape=[7, len(data.columns) - 1])
	for col_name in data.columns:
		if col_name != 'rating':
			col_index = data.columns.get_loc(col_name)

			# calc p1, n1, p0, n0
			calculations[0, col_index] = len(data.loc[(data[col_name] == 1) & (data['rating'] == 'Positive')].index)
			calculations[1, col_index] = len(data.loc[(data[col_name] == 1) & (data['rating'] == 'Negative')].index)
			calculations[2, col_index] = len(data.loc[(data[col_name] == 0) & (data['rating'] == 'Positive')].index)
			calculations[3, col_index] = len(data.loc[(data[col_name] == 0) & (data['rating'] == 'Negative')].index)

			# calc entropy at 1
			if calculations[0, col_index] == 0 and calculations[1, col_index] == 0:			# no case with word = 1
				calculations[4, col_index] = -1
			elif calculations[0, col_index] == 0 or calculations[1, col_index] == 0:		# pure decision at 1
				calculations[4, col_index] = 0
			else:
				p = calculations[0, col_index] / (calculations[0, col_index] + calculations[1, col_index])
				n = calculations[1, col_index] / (calculations[0, col_index] + calculations[1, col_index])
				calculations[4, col_index] = -1 * (p * math.log2(p) + n * math.log2(n))

			# calc entropy at 0
			if calculations[2, col_index] == 0 and calculations[3, col_index] == 0:			# no case with word = 0
				calculations[5, col_index] = -1
			elif calculations[2, col_index] == 0 or calculations[3, col_index] == 0:		# pure decision at 0
				calculations[5, col_index] = 0
			else:
				p = calculations[2, col_index] / (calculations[2, col_index] + calculations[3, col_index])
				n = calculations[3, col_index] / (calculations[2, col_index] + calculations[3, col_index])
				calculations[5, col_index] = -1 * (p * math.log2(p) + n * math.log2(n))

			# calc gain
			p1 = (calculations[0, col_index] + calculations[1, col_index]) / (pt + nt)
			p0 = (calculations[2, col_index] + calculations[3, col_index]) / (pt + nt)
			calculations[6, col_index] = entropy_total - p1 * calculations[4, col_index] - p0 * calculations[5, col_index]

	# get index of word with max gain
	max_index = np.argmax(calculations[6, :])
	decision = -1
