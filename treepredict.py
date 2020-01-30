#!/usr/bin/env python2

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
"""
def unique_counts(part): 
    results = {}
    for row in part:
        r = row[len(row)-1]
        if r not in results: results[r] = 0
        results[r] += 1
    return results

"""
T5 function that computes the Gini index of a node
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
T7 function that partitions a previous partition, taking
into account the values of a given attribute (column).
column is the index of the column and value is the value of
the partition criterion.
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
T8 class decisionnode, which represents a node
in the tree.
'''
class decisionnode:
    def __init__(self,col=-1,value=None,results=None,tb=None,fb=None):
        self.col = col
        self.value = value
        self.results = results
        self.tb = tb
        self.fb = fb

'''
T9 recursive function that builds a decision tree using any
of the impurity measures.
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

            p1=float(len(set1))/len(part)
            p2=float(len(set2))/len(part)
            gain=current_score-p1*scoref(set1)-p2*scoref(set2)
            
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

                p = float(len(set1)) / len(n.part) 
                gain = current_score - p*scoref(set1) - (1-p)*scoref(set2) 
        
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
T13/14 Define a function test that takes a test set and a training
set and computes the percentage of examples correctly
classified. Show the quality of the classifier.
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
T16 Define a function that every pair of leaves with a common father check if their
union increases the entropy below a given threshold. If that
is the case, delete those leaves by joining their prototypes
in the father
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
 
        p = float(len(tb) / len(tb + fb))
        d = entropy(tb+fb) - p*entropy(tb) - (1-p)*entropy(fb)

        if d<threshold:
            tree.tb,tree.fb=None,None
            tree.results=unique_counts(tb+fb)



if __name__ == '__main__':
    data=read('decision_tree_example.txt','\t')
    tree=buildtree(data)
    tree1=read('poker-hand-training-true-1.data',',')
    tree2=read('poker-hand-training-true-2.data',',')

    #print gini_impurity(data)
    #print entropy(data)
    #print(classify(['(direct)','USA','yes',5],tree))
    
    print "----------RECURSIVE TREE----------"
    printtree(buildtree(data))
    print "----------ITERATIVE TREE----------"
    printtree(buildtree_ite(data))
    print "----------DESPRES DE PRUNE----------"
    prune(tree,1)
    printtree(tree)
    print "----------TEST PERFORMANCE----------"
    print test_performance(read('poker-hand-testing.data',','), tree1)
    print "----------TEST PERFORMANCE INCREASED----------"
    print test_performance(read('poker-hand-testing.data',','), tree2)