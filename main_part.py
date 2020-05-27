# Reading train and building the decision tree
train_df = pd.read_csv('sample_train.csv')
train_df = train_df.drop(columns=['reviews.text'])
print('Building Tree...')
root = build_tree(train_df)
print('Tree was built successfully')
print()

# Evaluating the accuracy by traversing the tree using the train sample
print('Processing Train...')
count = 0
output = []
for i, row in train_df.iterrows():
	output.append(root.traverse(row))
	if train_df.iloc[i, -1] == output[-1]:
		count = count + 1
print('Accuracy = {:0.2f}%' .format((count / len(train_df.index) * 100)))
print('Train processed successfully')
print()

# Predicting the ratings of the test sample
test_df = pd.read_csv('sample_test.csv')
print('Processing Test...')
count = 0
output = []
for i, row in test_df.iterrows():
	output.append(root.traverse(row))
label = {'rating': output}
output_df = pd.DataFrame(label)
output_df.to_csv('result.csv', index=False)
print('Predictions saved to (result.csv)')

# Predicting the result of a user entered case
while True:
	print('Try new case? (Y/N)')
	choice = input()
	if choice == 'N' or choice == 'n':
		break
	else:
		print('Enter new case:')
		case = {}
		for i in range(0, 25):
			col_name = train_df.columns[i]
			case[col_name] = int(input())
		case = pd.Series(case)
		print('Prediction: ', root.traverse(case))
		print()
