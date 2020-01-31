#!/usr/bin/env python2
"""
Codigo fuente: treepredict.py
Grau Informatica
X8592934L Yassine El kihal
18068091G Jose Miguel Avellana
"""

class Node(object):
    def __init__(self, part, node):
        self.part = part
        self.node = node

class Stack:
     def __init__(self):
         self.items = []

     def isEmpty(self):
         return self.items == []

     def push(self, item):
         self.items.append(item)

     def pop(self):
         return self.items.pop()
"""
T3 Define a function to load the data into a bidimensional list
named data
"""
def read(file_name,spliter):
    data = []
    with open(file_name,'r') as file:
        for line in file:
            data.append(list(map(parser,line.strip('\n').split(spliter))))
    return data

def parser(element):
    try:
        return int(element)
    except ValueError:
        try:
            return float(element)
        except ValueError:
            return element
"""
T4 Create counts of possible results
(The last column of each row is the result)
"""
def unique_counts(part): 
    results = {}
    for row in part:
        r = row[len(row)-1]
        if r not in results: results[r] = 0
        results[r] += 1
    return results
"""
T5 Difine a function that computes the Gini index of a node
"""
def gini_impurity(part): 
    total = len(part)
    results = unique_counts(part)
    imp = 0
    for i in results:
        imp += pow(float(results[i])/total, 2)
    return 1-imp
"""
T6 function that computes the entropy of a node
Entropy is the sum of p(x)log(p(x))
across all the different possible results
"""
def entropy(part):
    from math import log
    log2 = lambda x:log(x)/log(2)
    results = unique_counts(part)
    # Now calculate the entropy
    imp = 0.0

    for i in results:
        p = float(results[i])/len(part)
        imp -= p*log2(p)
    return imp
"""
T7 Difine a function that partitions a previous partition.
Divide a ser on specific column. Can handle numeric or 
categorial values
"""
def divideset(part, column, value):
    isplit_function = None
    if isinstance(value, int) or isinstance(value, float):
        split_function = lambda prot: prot[column]>=value
    else:
        split_function = lambda prot: prot[column]==value
    set1, set2 = [], []
    for line in part:
        if split_function(line): set1.append(line)
        else: set2.append(line)
    return set1, set2
'''
T8 Define a new class decisionnode, which represents a node
in the tree using the constructor (def__init__(....))
'''
class decisionnode:
    def __init__(self,col=-1,value=None,results=None,tb=None,fb=None):
        self.col = col
        self.value = value
        self.results = results
        self.tb = tb
        self.fb = fb
'''
T9 Difine a new recursive function that builds a decision tree using any
of the impurity measures we have seen following the criterion.
'''
def buildtree(part,scoref=entropy, beta=0):
    if len(part)==0: return decisionnode()
    current_score=scoref(part)

    #Set up some variables to track the best criteria
    best_gain=0.0
    best_criteria=None
    best_sets=None

    column_count=len(part[0])-1
    for col in range(0,column_count):
        column_values={}
        for row in part:
            column_values[row[col]]=1
    
        for value in column_values.keys():
            (set1,set2)=divideset(part,col,value)

            probability1=float(len(set1))/len(part)
            probability2=float(len(set2))/len(part)
            gain=current_score-probability1*scoref(set1)-probability2*scoref(set2)
            
            if gain>best_gain and len(set1)>0 and len(set2)>0:
                best_gain=gain
                best_criteria=(col,value)
                best_sets=(set1,set2)
        
    if best_gain>0:
        tb=buildtree(best_sets[0])
        fb=buildtree(best_sets[1])
        return decisionnode(best_criteria[0],best_criteria[1],None,tb,fb)
    else:
        return decisionnode(results=unique_counts(part))
'''
T10 Difine a new iterative function that builds a decision tree using any
of the impurity measures we have seen following the criterion.
'''
def buildtree_ite(part, scoref=entropy, beta=0): 
    stack = []
    node = Node(part,decisionnode())
    stack.append(node)
    
    while len(stack) != 0:
        n = stack.pop()
        current_score = scoref(n.part)
        best_gain = 0.0
        best_criteria = None
        best_sets = None

        column_count = len(n.part[0]) - 1 
        for col in range(0, column_count):
            column_values={}
            for row in n.part:
                column_values[row[col]]=1
            for value in column_values.keys():
                set1, set2 = divideset(n.part, col, value)

                probability = float(len(set1)) / len(n.part) 
                gain = current_score - probability*scoref(set1) - (1-probability)*scoref(set2) 
        
                if gain > best_gain and len(set1) > 0 and len(set2) > 0:
                    best_gain = gain
                    best_criteria = (col, value)
                    best_sets = (set1, set2)

        if best_gain > beta:
            n.node.tb = decisionnode()
            n.node.fb = decisionnode()
            n.node.col = best_criteria[0]
            n.node.value = best_criteria[1]
            nodeTrueBranch = Node(best_sets[0], n.node.tb)
            nodeFalseBranch = Node(best_sets[1], n.node.fb)
            stack.append(nodeTrueBranch)
            stack.append(nodeFalseBranch)  
        else:
            n.node.results = unique_counts(n.part)
    return node.node

"""
T11 fuction for printing the trees
"""
def printtree(tree,indent=''):
    # Is this a leaf node?
    if tree.results!=None:
        print str(tree.results)
    else:
        # Print the criteria
        print str(tree.col)+':'+str(tree.value)+'? '

        # Print the branches
        print indent+'T->',
        printtree(tree.tb,indent+' ')
        print indent+'F->',
        printtree(tree.fb,indent+' ')

"""
T12 Build a function classify that allows to classify new objects.
It must return the dictionary that represents the partition of
the leave node where the object is classified.
"""
def classify(obj,tree): 
    if tree.results!=None:
        return tree.results
    else:
        v=obj[tree.col]
        branch=None
        if isinstance(v,int) or isinstance(v,float):
            if v>=tree.value: branch=tree.tb
            else: branch=tree.fb
        else:
            if v==tree.value: branch=tree.tb
            else: branch=tree.fb
    return classify(obj,branch)

"""
T13 Define a function test that takes a test set and a training
set and computes the percentage of examples correctly
classified. T14 Show the quality of the classifier increasing a 20% the
training set. You can retrieve data from the following database(Http....)

"""
def test_performance(testset, trainingset):
    tree = buildtree(trainingset)
    accuracy = 0.0
    for line in testset:
        obj = line.pop()
        result = classify(line,tree)
        if result.keys()[0] == obj:
            accuracy+=1
    return accuracy/len(testset)

"""
T16 Define a function that for every pair of leaves with a 
common father check if their union increases the entropy 
below a given threshold. If thatis the case, delete those 
leaves by joining their prototypes in the father.
This function implements the above prunning strategy.
"""
def prune(tree,threshold):
    if tree.tb.results==None: prune(tree.tb,threshold)
    if tree.fb.results==None: prune(tree.fb,threshold)
    
    if tree.tb.results!=None and tree.fb.results!=None:
        tb,fb=[],[]
        for v,c in tree.tb.results.items():
            tb+=[[v]]*c
        for v,c in tree.fb.results.items():
            fb+=[[v]]*c
 
        probability = float(len(tb) / len(tb + fb))
        d = entropy(tb+fb) - probability*entropy(tb) - (1-probability)*entropy(fb)

        if d<threshold:
            tree.tb,tree.fb=None,None
            tree.results=unique_counts(tb+fb)



if __name__ == '__main__':
    data=read('decision_tree_example.txt','\t')
    tree=buildtree(data)
    data_train1=read('poker-hand-training-true-1.data',',')
    data_train2=read('poker-hand-training-true-2.data',',')

    print gini_impurity(data)
    print entropy(data)
    print(classify(['(direct)','USA','yes',5],tree))
    
    print "----------ARBOL RECURSIVO----------"
    printtree(buildtree(data))
    print "----------ARBOL ITERATIVO----------"
    printtree(buildtree_ite(data))
    print "----------PODAR ARBOL----------"
    prune(tree,1)
    printtree(tree)
    print "----------PERFORMANCE TEST----------"
    print test_performance(read('poker-hand-testing.data',','), data_train1)
    print "----------PERFORMANCE TEST INCREASED----------"
    print test_performance(read('poker-hand-testing.data',','), data_train2)