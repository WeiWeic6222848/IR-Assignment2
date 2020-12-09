import sys
from datetime import datetime

from pyspark import SparkContext
from pyspark.sql import Row, SparkSession
from Config import teleportationProbability


def updateIteration(entry):
    # entry = ("0",[lijst outgoing])
    n = len(entry[1])
    p = int(entry[0][0])
    PR = float(entry[0][1])
    outlinks = entry[1]

    value1=[]
    for outlink in outlinks:
        value1.append((outlink,PR/n))

    value1.append((str(p),outlinks))
    return value1

def updateReduce(next):
    N = 50000000
    # entry = ("0",[lijst outgoing])
    #PR(di) := /N + (1-)j=1..m PRj(di);
    #emit(key3: [di, PR(di)], value3: outlinks(di));
    PRlist=next[1]
    p=next[0]
    Outlinks = []
    PRSum=0
    for i in PRlist:
        if(not isinstance(i,list)):
            PRSum+=i
        else:
            Outlinks=i
    PR = teleportationProbability * 1 / N + (1-teleportationProbability)*PRSum
    return ((p,PR),Outlinks)

def pageRank(data):
    N = 50000000
    data = data.map(lambda x: ((x[0], 1 / N), x[1]))

    rold= None
    rnew = data
    iter=0
    diff = sys.float_info.max
    eps=0.000001
    while diff>eps:
        timer=datetime.now()
        iter+=1
        rold=rnew
        rnew=rold.flatMap(updateIteration).groupByKey().map(updateReduce)

        #calculate difference?
        diff=rold.map(lambda x:x[0]).join(rnew.map(lambda x:x[0])).map(lambda x:abs(x[1][0]-x[1][1]))
        diff=diff.max()
        print("Iteration ",iter," just finished with diff: ",diff," approximated elapsed time = ",(datetime.now()-timer).total_seconds())

    result=rnew.map(lambda x:x[0]).sortBy(lambda x:x[1],False).collect()
    print(result)
    return True


if __name__ == '__main__':
    sc = SparkContext("local", "PageRanking")

    # filtering mini database to be of form (from node, list of to nodes)
    data = sc.textFile("./Dataset/web-Google.txt").filter(lambda l: not str(l).startswith("#")) \
        .map(lambda x: x.split("\t")).map(lambda x: (str(x[0].strip()), str(x[1].strip()))).groupBy(
        lambda x: x[0]).mapValues(lambda x: list(map(lambda y: y[1], x))).sortBy(lambda x: x[0])
    # (0,[]) <-
    # (score) <- rold

    pageRank(data)
