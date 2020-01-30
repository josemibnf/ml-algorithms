import sys
from queue import Queue 

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

def read(data, file_name):
    if file_name[-1]=="t":  #para .txt
        split='\t'
    elif file_name[-1]=="a":
        split=','    #para .data
    else:
        print("Error: extension de archivo invalida.")
        sys.exit()

    with open(file_name, 'r') as f:
        for line in f:
            treated_line = [i for i in line.rstrip('\n').split(split)]
            data.append(treated_line)


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
    if isinstance(value, int) or isinstance(value, float):
        def split_fun(elem): return elem[column] <= value
    else:
        def split_fun(elem): return elem[column] == value

    set1, set2 = [], []
    for elem in part:
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
        
    def actualiza(self,  col=-1, value=None, results=None, tb=None, fb=None):
        self.col = col 
        self.value = value
        self.results = results
        self.tb=tb
        self.fb=fb

def buildtree(part, scoref=gini_impurity, beta=0):
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
            (set1,set2)=divide_set(part,col,value)

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
 
def buildtree_ite(part, scoref=gini_impurity, beta=0):
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
                set1, set2 = divide_set(n.part, col, value)

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


def printtree(tree, indent=''):
    # Is this a leaf node?
    if tree.results is not None:
        print(indent+str(tree.results))
    else:
        # Print the criteria
        print(indent + str(tree.col)+':'+str(tree.value)+'? ')
        # Print the branches
        print(indent+'T->')
        printtree(tree.tb, indent+' ')
        print(indent+'F->')
        printtree(tree.fb, indent+' ')
       
 
def classify(obj, tree):
    if tree.results is not None:    #Es nodo hoja.
        return tree.results
    else:
        if isinstance(tree.value, int) or isinstance(tree.value, float):
            def split_fun(elem): return elem[tree.col] <= tree.value
        else:
            def split_fun(elem): return elem[tree.col] == tree.value

        if split_fun(obj):
            return classify(obj, tree.tb)
        else:
            return classify(obj, tree.fb)
        
 
def test_performance(testset, trainingset):
    tree = buildtree(part=trainingset)
    correct=0
    for elem in testset:  
        class_list = classify(elem, tree)
        if elem[-1] in class_list:
            correct+=1

        """
        else:
            print("Noooooooo:  Es")
            print(class_list)
            print("no")
            print(elem[-1])
            print("")
        """

    return (correct*100)/(len(testset))

 
def prune(tree,threshold=1.0):
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
 


if __name__ == "__main__":
    if len(sys.argv)==1:
        file_data="decision_tree_example.txt"  #por defecto
    else:
        file_data=sys.argv[1]
    print(file_data)
    dat_file=[]
    read(dat_file, file_data)
    counts = unique_counts(dat_file)
    gini = gini_impurity(dat_file)
    ent = entropy(dat_file)
    tree = buildtree(part=dat_file)
    iter_tree = buildtree_ite(part=dat_file)
 
    #print("Training Set:\n", dat_file)   #Printa todo el DataSet
    print("Goal Attributes:", counts)
    print("Gini Index:", gini)
    print("Entropy:", ent)

    print("\nBuild Recursive Tree: ")
    printtree(tree)

    print("\nBuild Iterative Tree: ")
    printtree(iter_tree)

    print("\n------------")
    bool_elem_not_ready = True
    while bool_elem_not_ready:
        print("\nNumero de elementos totales: ",len(dat_file))
        print("Introduce el numero de elementos a entrenar:")
        to_train = int(input())
        if to_train<len(dat_file):
            bool_elem_not_ready=False
        else:
            print("Demasiados elementos para entrenar.")
    print("Test Performance: ", test_performance(testset=dat_file[to_train:], trainingset=dat_file[:to_train]))

    print("\n")
    while True:
        print("¿Quieres podar el árbol? (Y/N): ")
        bool_prune = input()
        if bool_prune=='Y':
            print("\nPrune Tree: ", prune(tree))
            printtree(tree)
            exit()
        elif bool_prune=='N':
            exit()
