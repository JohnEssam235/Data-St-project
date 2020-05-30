
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
	
	
		# probabilities of word with max gain
	if calculations[4, max_index] == -1:			# no case with word = 1
		p1 = n1 = -1
	else:
		p1 = calculations[0, max_index] / (calculations[0, max_index] + calculations[1, max_index])
		n1 = calculations[1, max_index] / (calculations[0, max_index] + calculations[1, max_index])

	if calculations[5, max_index] == -1:			# no case with word = 0
		p0 = n0 = -1
	else:
		p0 = calculations[2, max_index] / (calculations[2, max_index] + calculations[3, max_index])
		n0 = calculations[3, max_index] / (calculations[2, max_index] + calculations[3, max_index])

	# if there is no case where either word = 1 or word = 0
	if p1 == -1 or p0 == -1:
		if pt >= nt:
			decision = 4					# pos at 1 and pos at 0
		else:
			decision = 7					# neg at 1 and neg at 0

	# if there is only one word
	elif len(data.columns) == 2:
		if (p1 >= n1) and (p0 >= n0):
			decision = 4					# pos at 1 and pos at 0
		elif (p1 >= n1) and (p0 <= n0):
			decision = 5					# pos at 1 and neg at 0
		elif (p1 <= n1) and (p0 >= n0):
			decision = 6					# neg at 1 and pos at 0
		elif (p1 <= n1) and (p0 <= n0):
			decision = 7					# neg at 1 and neg at 0

	# if there is pure decisions at word == 1 and also at word == 0
	elif (calculations[4, max_index] == 0) and (calculations[5, max_index] == 0):
		if (p1 == 1) and (p0 == 1):
			decision = 4					# pos at 1 and pos at 0
		elif (p1 == 1) and (n0 == 1):
			decision = 5					# pos at 1 and neg at 0
		elif (n1 == 1) and (p0 == 1):
			decision = 6					# neg at 1 and pos at 0
		elif (n1 == 1) and (n0 == 1):
			decision = 7					# neg at 1 and neg at 0

	# if there is a pure decision at word == 1
	elif calculations[4, max_index] == 0:
		if p1 == 1:
			decision = 0					# pos at 1
		else:
			decision = 1					# neg at 1

	# if there is a pure decision at word == 0
	elif calculations[5, max_index] == 0:
		if p0 == 1:
			decision = 2					# pos at 0
		else:
			decision = 3					# neg at 0

	return max_index, decision
