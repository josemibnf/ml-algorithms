import random
def kcluster(rows, distance=pearson, k=4):
    
    # Determine the minimum and maximum values for each point
    ranges=[(min([row[i] for row in rows]),
    max([row[i] for row in rows])) for i in range(len(rows[0]))]

    #Create k ramdomly placed centroids
    clusters=[[random.random()*(ranges[i][1]-ranges[i][0])+ranges[i][0]
    for i in range(len(rows[0]))] for j in range(k)]