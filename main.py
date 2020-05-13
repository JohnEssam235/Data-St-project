import numpy as np
import pandas as pd
import math

df = pd.read_csv('sample_train.csv')
df = df.drop(columns=['reviews.text'])


def calc_gain(data):
	# calc total entropy
	Pt = len(data.loc[data['rating'] == 'Positive'].index)
	Nt = len(data.loc[data['rating'] == 'Negative'].index)
	entropy_total = -1 * (((Pt / (Pt + Nt)) * math.log2((Pt / (Pt + Nt)))) + ((Nt / (Pt + Nt)) * math.log2((Nt / (Pt + Nt)))))
	print(entropy_total)
	print()

	# calc entropy of each attribute
	entropy_values = np.zeros(shape=[7, 25])
	for col_name in data.columns:
		if col_name != 'rating':
			col_index = data.columns.get_loc(col_name)
			entropy_values[0, col_index] = len(data.loc[(data[col_name] == 1) & (data['rating'] == 'Positive')].index)  # P1
			entropy_values[1, col_index] = len(data.loc[(data[col_name] == 1) & (data['rating'] == 'Negative')].index)  # N1
			entropy_values[2, col_index] = len(data.loc[(data[col_name] == 0) & (data['rating'] == 'Positive')].index)  # P0
			entropy_values[3, col_index] = len(data.loc[(data[col_name] == 0) & (data['rating'] == 'Negative')].index)  # N0

			x = entropy_values[0, col_index] / (entropy_values[0, col_index] + entropy_values[1, col_index])
			y = entropy_values[1, col_index] / (entropy_values[0, col_index] + entropy_values[1, col_index])
			entropy_values[4, col_index] = -1 * (x * math.log2(x) + y * math.log2(y))  # entropy1

			x = entropy_values[2, col_index] / (entropy_values[2, col_index] + entropy_values[3, col_index])
			y = entropy_values[3, col_index] / (entropy_values[2, col_index] + entropy_values[3, col_index])
			entropy_values[5, col_index] = -1 * (x * math.log2(x) + y * math.log2(y))  # entropy0

			x = entropy_values[0, col_index] + entropy_values[1, col_index]
			y = entropy_values[2, col_index] + entropy_values[3, col_index]
			entropy_values[6, col_index] = entropy_total - (x / (Pt + Nt)) * entropy_values[4, col_index] - (y / (Pt + Nt)) * entropy_values[5, col_index]  # gain

	print(entropy_values[4, :])
	print()
	print(entropy_values[5, :])
	print()
	print(entropy_values[6, :])
	# get index at max gain
	max_index = np.where(entropy_values[6, :] == np.amax(entropy_values[6, :]))

	return max_index


print(calc_gain(df))
