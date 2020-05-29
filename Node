class Node(object):
	def __init__(self):
		self.column = None
		self.decision = None
		self.children = None

	def traverse(self, data):
		if data[self.column] == 1:
			if self.decision == 0 or self.decision == 4 or self.decision == 5:
				return 'Positive'
			elif self.decision == 1 or self.decision == 6 or self.decision == 7:
				return 'Negative'
			else:
				data = data.drop(data.index[self.column])
				return self.children[-1].traverse(data)

		else:
			if self.decision == 2 or self.decision == 4 or self.decision == 6:
				return 'Positive'
			elif self.decision == 3 or self.decision == 5 or self.decision == 7:
				return 'Negative'
			else:
				data = data.drop(data.index[self.column])
				return self.children[0].traverse(data)
        
