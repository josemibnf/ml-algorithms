import sys


# T3
def read(file_name):
	ret = []
	with open(file_name, 'r') as f:

		for line in f:
			treated_line = line.rstrip('\n').split('\t')
			ret.append(treated_line)

	return ret

# T4
def unique_counts(part):
	res = {}

	for elem in part:
		if elem[-1] not in res:
			res[elem[-1]] = 1
		else:
			res[elem[-1]] += 1

	return res

# T5
def gini_impurity(part):
	total = len(part)
	results = unique_counts(part)
	imp = 0
	for v in results.values():
		imp += (v / float(total))**2
	return 1 - imp

# T6
def entropy(rows):
	from math import log
	total = len(rows)
	results = unique_counts(rows)
	ent = 0.0

	for v in results.values():
		p = v / float(total)
		ent -= p * log(p, 2)

	return ent

# T7
def divide_set(part, column, value):
	def split_fun(prot): return prot[column] == value
	if isinstance(value, int) or isinstance(value, float):
		def split_fun(prot): return prot[column] <= value

	set1, set2 = [], []
	for elem in part:
		if split_fun(elem):
			set1.append(elem)
		else:
			set2.append(elem)

	return set1, set2

# T8
class decisionnode(object):
	def __init__(self, col=-1, value=None, results=None, tb=None, fb=None):
		self.col = col
		self.value = value
		self.results = results
		self.tb = tb
		self.fb = fb

# T9 - Construccion del arbol de forma recursiva
def buildtree(part, scoref=gini_impurity):
	if len(part) == 0:
		return decisionnode()
	current_score = scoref(part)

	best_gain = 0
	best_criteria = None
	best_sets = None
	elements = len(part[0]) - 1
	for elem in range(0, elements):
		elements_values = set([elements_value[elem] for elements_value in part])

		for value in elements_values:
			set1, set2 = divide_set(part, elem, value)

			probability1 = len(set1) / len(part)
			probability2 = len(set1) / len(part)

			current_gain = current_score - (probability1 * scoref(set1)) - (probability2 * scoref(set2))

			if current_gain > best_gain and len(set1) > 0 and len(set2) > 0:
				best_gain = current_gain
				best_criteria = (elem, value)
				best_sets = set1,set2

	if best_gain > 0:
		
		true = buildtree(best_sets[0])
		false = buildtree(best_sets[1])
		return decisionnode(best_criteria[0], best_criteria[1], None, true, false)

	else:
		return decisionnode(results=unique_counts(part))

"""
# T10 - Construccion del arbol de forma iterativa
def buildtree_ite(part, scoref=gini_impurity):
	pass

# T11 - Restarting policies
def kcluster():
		pass
"""

def printtree(tree, indent=''):
	# Is this a leaf node?
	if tree.results is not None:
		print(tree.results)
	else:
		# Print the criteria
		print(indent + str(tree.col)+':'+str(tree.value)+'? ')
		# Print the branches
		print(indent+'T->')
		printtree(tree.tb, indent+'  ')
		print(indent+'F->')
		printtree(tree.fb, indent+'  ')
		
# T12 - Funcion de clasificacion
def classify(obj, tree):
	if tree.results is not None:
		return tree.results
	else:
		v = obj[tree.col]
		branch = None

		if isinstance(v, int) or isinstance(v, float):
			if v >= tree.value: branch = tree.tb
			else: branch = tree.fb
		else:
			if v == tree.value:	branch = tree.tb
			else: branch = tree.fb		

		return classify(obj, branch)	

# T13/14 - Evaluacion del arbol 
def test_performance(testset, trainingset):
	pass

# T15 -Missing data
	# Alternativas dataset a los que les faltan datos

# T16 - Poda de arbol
def prune(tree, threshold):
	pass


if __name__ == "__main__":
	dat_file = read(sys.argv[1])
	counts = unique_counts(dat_file)
	gini = gini_impurity(dat_file)
	ent = entropy(dat_file)
	tree = buildtree(dat_file)

	print("Training Set:\n", dat_file)
	print("Goal Attributes:", counts)
	print("Gini Index:", gini)
	print("Entropy:", ent)
	printtree(tree)
	print("Classify Tree: ", classify(['slashdot', 'USA', 'yes', '18'], tree))
	#print('Divide set: (Location, New Zealand)\n', divide_set(dat_file, 1, 'New Zealand'))
