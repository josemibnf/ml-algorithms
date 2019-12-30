import sys
from queue import Queue 

def read(data, file_name):
    ret = []
    with open(file_name, 'r') as f:
        for line in f:
            treated_line = line.rstrip('\n').split('\t')
            ret.append(treated_line)
    return ret


def unique_counts(part):
    res = {}
    for elem in part:
        if elem[-1] not in res:
            res[elem[-1]] = 1
        else:
            res[elem[-1]] += 1
    return res


def gini_impurity(part):
    total = len(part)
    results = unique_counts(part)
    imp = 0
    for v in results.values():
        imp += (v / float(total))**2

    return 1 - imp


def entropy(rows):
    from math import log
    total = len(rows)
    results = unique_counts(rows)
    ent = 0.0
 
    for v in results.values():
        p = v / float(total)
        ent -= p * log(p, 2)
 
    return ent
 
 
def divide_set(part, column, value):
    def split_fun(elem): return part[column[elem]] == value
    if isinstance(value, int) or isinstance(value, float):
        def split_fun(elem): return part[column[elem]] <= value

    set1, set2 = [], []
    for elem in part[column]:
        if split_fun(elem):
            set1.append(elem)
        else:
            set2.append(elem)
    return (set1, set2)

 
class decisionnode(object):
 
    def __init__(self, col=-1, value=None, results=None, tb=None, fb=None):
        self.col = col 
        self.value = value
        self.results = results
        self.tb = tb
        self.fb = fb
        
    def actualiza(self,  col=-1, value=None, results=None tb=None, fb=None):
        self.col = col 
        self.value = value
        self.results = results
        self.tb=tb
        self.fb=fb

def buildtree(part, scoref=gini_impurity, beta=0):
	if len(part) == 0:
		return decisionnode()
	current_score = scoref(part)  

	best_gain = 0
	best_criteria = None
	best_sets = None
	elements = len(part[0]) - 1
	for elem in range(0, elements):  #Recorremos todas los valores para saber cual es la columna/elemento que mayor descenso de impureza/desorden ofrece.
		element_values = set([elements_value[elem] for elements_value in part])
        colum=0
        for value in element_values:
            set1, set2 = divide_set(part, colum, value)
            colum=colum+1
            probability1 = len(set1) / len(part)
            probability2 = len(set1) / len(part)

            current_gain = current_score - (probability1 * scoref(set1)) - (probability2 * scoref(set2))

            if current_gain > best_gain and len(set1) > 0 and len(set2) > 0:
				best_gain = current_gain
				best_criteria = elem, value
				best_sets = set1,set2

	if best_gain > beta:  #¿Hemos conseguido algun split disminuya la impureza/desorden por encima de beta?
		true = buildtree(best_sets[0], scoref, beta)
		false = buildtree(best_sets[1], scoref, beta)
		return decisionnode(best_criteria[0], best_criteria[1], None, true, false)

	else:
		return decisionnode(results=unique_counts(part))
 
def buildtree_ite(part, scoref=gini_impurity, beta=0):
    rootNode=None
    fringe=Queue()
    fringe.put(([] ,part))
    while fringe.empty() == False:
        nodo = fringe.get()
        current_score = scoref(nodo[1])  

        best_gain = 0
        best_criteria = None
        best_sets = None
        elements = len(nodo[1][0]) - 1
        
        for elem in range(0, elements):  
            element_values = set([elements_value[elem] for elements_value in nodo[1]])

            for value in element_values: 
                set1, set2 = divide_set(nodo[1], elem, value)

                probability1 = len(set1) / len(nodo[1])
                probability2 = len(set2) / len(nodo[1])

                current_gain = current_score - (probability1 * scoref(set1)) - (probability2 * scoref(set2))

                if current_gain > best_gain and len(set1) > 0 and len(set2) > 0:
                    best_gain = current_gain
                    best_criteria = elem, value
                    best_sets = set1,set2

        if best_gain > beta:
            true = decisionnode()
            false = decisionnode()
            fringe.put((true ,(best_sets[0]) ))
            fringe.put((false ,(best_sets[1]) ))
            if rootNode == None:
                rootNode = decisionnode(best_criteria[0], best_criteria[1], None, true, false)
            else:
                nodo[0].actualiza(best_criteria[0], best_criteria[1], None, true, false)
        else:
            nodo[0].actualiza(results=unique_counts(part))


        return rootNode


def printtree(tree, indent=''):
    # Is this a leaf node?
    if tree.results is not None:
        print(indent+str(tree.results))
    else:
        # Print the criteria
        print(indent + str(tree.col)+':'+str(tree.value)+'? ')
        # Print the branches
        print(indent+'T->')
        printtree(tree.tb, indent+'  ')
        print(indent+'F->')
        printtree(tree.fb, indent+'  ')
       
 
def classify(obj, tree):
    if tree.results is not None:    #Es nodo hoja.
        return tree.results
    else:
        def split_fun(elem): return elem[tree.col] == tree.value
        if isinstance(tree.value, int) or isinstance(tree.value, float):
            def split_fun(elem): return elem[tree.col] <= tree.value

        if split_fun(obj):
            return classify(obj, tree.tb)
        else:
            return classify(obj, tree.fb)
        
 
def test_performance(testset, trainingset):
    tree = buildtree(part=read(trainingset))
    test = read(testset)
    elements = len(test) - 1
    for elem in range(0, elements):  
        obj = [elements_value[elem] for elements_value in test]
        if elem in classify(elem, tree):
            correct+=1
    print(correct/elements)

 
def prune(tree, threshold):
    pass

def transpostedMatrix(data):
    return [list(elem) for elem in zip(*data)]
 
if __name__ == "__main__":
    dat_file = read(sys.argv[1])
    counts = unique_counts(dat_file)
    gini = gini_impurity(dat_file)
    ent = entropy(dat_file)
    tree = buildtree(part=dat_file)
 
    print("Training Set:\n", dat_file)
    print("Goal Attributes:", counts)
    print("Gini Index:", gini)
    print("Entropy:", ent)
 
    print("Build Tree: ", tree)
    #print("Print Tree: ", printtree(tree))
    #print('Divide set: (Location, New Zealand)\n', divide_set(dat_file, 1, 'New Zealand'))
