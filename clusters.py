import random
import math

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

def euclideanSqr(v1,v2):
    s=0.0
    for i in range(len(v1)):
        s+=(v1[i]-v2[i])**2
    return s

def euclidean(v1,v2):
    return (euclideanSqr(v1,v2))**0.5

def pearson(v1,v2):
	# Simple sums
	sum1 = sum(v1)
	sum2 = sum(v2)

	# Sums of the squares
	sum1Sq = sum([pow(v,2) for v in v1])
	sum2Sq = sum([pow(v,2) for v in v2])

	# Sum of the products
	pSum = sum([v1[i]*v2[i] for i in range(len(v1))])

	# Calculate r (Pearson score)
	num = pSum-(sum1*sum2/len(v1))
	den = math.sqrt((sum1Sq-pow(sum1,2)/len(v1))*(sum2Sq-pow(sum2,2)/len(v1)))
	if den==0: return 0
	
	return 1.0-num/den

class bicluster:
	def __init__(self,vec,left=None,right=None,dist=0.0,id=None):
		self.left = left
		self.right = right
		self.vec = vec
		self.id = id
		self.distance = dist

def hcluster(rows,distance=euclidean):
	distances={} # stores the distances for efficiency
	currentclustid=-1 # all except the original items have a negative id
	
	# Clusters are initially just the rows
	clust = [bicluster(rows[i],id=i) for i in range(len(rows))]

	while len(clust)>1: #stop when there is only one cluster left
		lowestpair = (0,1)
		closest = distance(clust[0].vec,clust[1].vec)

		# loop through every pair looking for the smallest distance
		for i in range(len(clust)):
			for j in range(i+1,len(clust)):
				# distances is the cache of distance calculations
				if (clust[i].id,clust[j].id) not in distances:
					distances[(clust[i].id,clust[j].id)] = distance(clust[i].vec,clust[j].vec)
				d = distances[(clust[i].id,clust[j].id)]

				if d < closest:
					closest = d
					lowestpair = (i,j)

		# calculate the average of the two clusters
		mergevec=[(clust[lowestpair[0]].vec[i]+clust[lowestpair[1]].vec[i])/2.0 for i in range(len(clust[0].vec))]

		# create the new cluster
		newcluster=bicluster(mergevec,left=clust[lowestpair[0]], 
		right=clust[lowestpair[1]],
		dist=closest,id=currentclustid)
		
        # cluster ids that weren't in the original set are negative
		currentclustid-=1
		del clust[lowestpair[1]]
		del clust[lowestpair[0]]
		clust.append(newcluster)
	return clust[0]

def printclust(clust,labels=None,n=0):
    # indent to make a hierarchy layout
    for i in range(n): print(' '),
    if clust.id<0:
        # negative id means that this is branch
        print('-')
    else:
        # positive id means that this is an endpoint
        if labels==None: print(clust.id)
        else: print(labels[clust.id])

    # now print the right and left branches
    if clust.left!=None:
        printclust(clust.left,labels=labels,n=n+1)
    if clust.right!=None:
        printclust(clust.right,labels=labels,n=n+1)

def rotatematrix(data):
    newdata = []
    for i in range(len(data[0])):
        newrow = [data[j][i] for j in range(len(data))]
        newdata.append(newrow)
    return newdata

def kcluster(rows,distance=euclidean,k=4):
    it=0
    total=0

    # Determine the minimum and maximum values for each point
    ranges=[(min([row[i] for row in rows]),
    max([row[i] for row in rows])) for i in range(len(rows[0]))]
    
    # Create k randomly placed centroids
    clusters=[[random.random()*(ranges[i][1]-ranges[i][0])+ranges[i][0]
    for i in range(len(rows[0]))] for j in range(k)]
    
    lastmatches=None
    for t in range(100):
        bestmatches=[[] for i in range(k)]
        # Find which centroid is the closest for each row
        for j in range(len(rows)):
            row=rows[j]
            bestmatch=0
            for i in range(k):
                d=distance(clusters[i],row)
                if d<distance(clusters[bestmatch],row): bestmatch=i
            bestmatches[bestmatch].append(j)

        # If the results are the same as last time, done
        if bestmatches==lastmatches: break
        lastmatches=bestmatches
        it+=1

        # Move the centroids to the average of their members
        for i in range(k):
            avgs=[0.0]*len(rows[0])
            if len(bestmatches[i])>0:
                for rowid in bestmatches[i]:
                    for m in range(len(rows[rowid])):
                        avgs[m]+=rows[rowid][m]
                for j in range(len(avgs)):
                    avgs[j]/=len(bestmatches[i])
                clusters[i]=avgs
    
    # TOTAL DISTANCE
    for j in range(k): 
        for i in range(len(bestmatches[j])): 
            row=rows[bestmatches[j][i]] 
            total += euclideanSqr(clusters[j], row)  

    return total, it

def minim_total(n, nklust):
    t1, it = kcluster(data, k=nklust)
    niter = it
    for i in range(n):
        t2,it = kcluster(data, k=nklust)
        if t1 > t2:
            t1 = t2
            niter = it           
    return t1, niter


if __name__ == '__main__':
    blognames,words,data = readfile('blogdata.txt')
    total, it= minim_total(5, 3)
    print "MILLOR CONFIGURACIO:",total
    print "Nombre d'iteracions:",it
    