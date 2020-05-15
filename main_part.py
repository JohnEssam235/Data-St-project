root = build_tree(df)
dev = pd.read_csv('sample_dev.csv')
dev = dev.drop(columns=['reviews.text'])

count = 0
output = []
for index, row in dev.iterrows():
	if root.traverse(row) == 1:
		output.append('Positive')

	else:
		output.append('Negative')

	if dev.iloc[index, -1] == output[-1]:
		count = count + 1

print('Accuracy = {:0.2f}%' .format((count / len(dev.index) * 100)))

label = {'rating': output}
output_df = pd.DataFrame(label)
output_df.to_csv('result.csv')