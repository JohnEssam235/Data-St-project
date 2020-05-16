def build_tree(data):
	root = Node()
	if data.empty:
		return root

	index, decision = calc_gain(data)
	root.column = index
	root.decision = decision
	root.childs = np.empty((1, 0), Node)

	# pure decision
	if decision != -1:
		# if decision is pure when word = 1
		if decision < 2:
			data0 = data.loc[data[data.columns[index]] == 0]		# get all rows where word == 0
			data0 = data0.drop(columns=[data.columns[index]])
			root.childs = np.append(root.childs, np.array([build_tree(data0)]))

		# if decision is pure when word = 0
		elif decision < 4:
			data1 = data.loc[data[data.columns[index]] == 1]		# get all rows where word == 1
			data1 = data1.drop(columns=[data.columns[index]])
			root.childs = np.append(root.childs, np.array([build_tree(data1)]))

		# if decision is pure at both 1 and 0
		elif decision < 8:
			return root

	else:
		data0 = data.loc[data[data.columns[index]] == 0]			# get all rows where word == 0
		data0 = data0.drop(columns=[data.columns[index]])
		root.childs = np.append(root.childs, np.array([build_tree(data0)]))

		data1 = data.loc[data[data.columns[index]] == 1]			# get all rows where word == 1
		data1 = data1.drop(columns=[data.columns[index]])
		root.childs = np.append(root.childs, np.array([build_tree(data1)]))

	return root
	
