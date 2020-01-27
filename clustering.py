# coding=utf-8
from math import sqrt
import random
import sys
from copy import deepcopy

def readfile(filename):
	with open(filename) as file:
		lines=[line for line in file]
		# First line is the column titles
		colnames=lines[0].strip().split('\t')[1:]
		rownames=[]
		data=[]
		for line in lines[1:]:
			p=line.strip().split('\t')
			# First column in each row is the rowname
			rownames.append(p[0])
			# The data for this row is the remainder of the row
			data.append([float(x) for x in p[1:]])
		return rownames,colnames,data

def euclidean(v1, v2):
	distance = sqrt(sum([(v1 - v2) ** 2 for v1, v2 in zip(v1, v2)]))
	return distance	


def manhattan(v1, v2):
	distance = sum([abs(v1[i]-v2[i]) for i in range(len(v1))])
	return distance

def pearson(v1,v2):
	# Simple sums
	sum1 = sum(v1)
	sum2 = sum(v2)
	# Sums of the squares
	sum1Sq = sum([pow(v,2) for v in v1])
	sum2Sq = sum([pow(v,2) for v in v2])
	# Sum of the products
	pSum = sum([v1[i] * v2[i] for i in range(len(v1))])
	# Calculate r (Pearson score)
	num = pSum-(sum1 * sum2/len(v1))
	den = sqrt((sum1Sq-pow(sum1,2)/len(v1)) * (sum2Sq-pow(sum2,2)/len(v1)))
	if den==0: 
                return 0
	
	return 1.0-num/den

class bicluster:
	def __init__(self, vec, left=None, right=None, distance=0.0, id=None):
		self.left = left
		self.right = right
		self.vec = vec
		self.id = id
		self.distance = distance

def printclust(clust, labels=None, n=0):
        # indent to make a hierarchy layout
        for i in range(n):
                print (' '),
        if clust.id < 0:
        # negative id means that this is branch
                print ('-')
        else:
        # positive id means that this is an endpoint
                if labels == None:
                    print (clust.id)
                else:
                    print (labels[clust.id])

  # now print the right and left branches
        if clust.left != None:
                printclust(clust.left, labels=labels, n=n + 1)
        if clust.right != None:
                printclust(clust.right, labels=labels, n=n + 1)

def kcluster(rows, distance=pearson, k=4):
    # Determine the minimum and maximum values for each point
    ranges = [(min([row[i] for row in rows]),
     max([row[i] for row in rows])) for i in range(len(rows[0]))]

  # Create k randomly placed centroids
    clusters = [[random.random() * (ranges[i][1] - ranges[i][0]) + ranges[i][0]
                for i in range(len(rows[0]))] for j in range(k)]
    lastmatches = []

    for t in range(100):
        print("Iteration", t)

        bestmatches = [[] for i in range(k)]

        # Find which centroid is the closest for each row
        for j in range(len(rows)):
            row = rows[j]
            bestmatch = 0

            for i in range(k):
                d = distance(clusters[i], row)
                if d < distance(clusters[bestmatch], row):
                    bestmatch = i
            bestmatches[bestmatch].append(j)

        if t%3 == 0 and t != 0: #reset
            for cent in range(0,k):
                if lastmatches[cent] != bestmatches[cent]:
                    print ("RESETING CLUSTER: ", cent)

                    clusters[cent] = [random.random() * (ranges[i][1] - ranges[i][0]) + ranges[i][0]
                        for i in bestmatches[cent]]

        # If the results are the same as last time, done
        if bestmatches == lastmatches:
            break

        lastmatches = bestmatches

        # Move the centroids to the average of their members
        for i in range(k):
            avgs = [0.0] * len(rows[0])

            if len(bestmatches[i]) > 0:

                for rowid in bestmatches[i]:
                    for m in range(len(rows[rowid])):
                        avgs[m] += rows[rowid][m]

                for j in range(len(avgs)):
                    avgs[j] /= len(bestmatches[i])

                clusters[i] = avgs

    return bestmatches, clusters

if __name__ == "__main__":
    rownames, colnames, data= readfile(sys.argv[1])
    random.seed(6)
    kclust , clusters = kcluster(data)
    
    for i in range(0,len(kclust)):
        print ("CENTROID OF THE ", i+1, "ELEMENT:")
        print (kclust[i])



