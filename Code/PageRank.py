import sys
from datetime import datetime

from pyspark import SparkContext, SparkConf, StorageLevel
from Config import teleportationProbability

N = 875713
deadendpool = 0

def updateIteration(entry):
    # entry = ("0",[lijst outgoing])
    p = entry[0][0]

    n = len(entry[1])
    PR = float(entry[0][1])
    outlinks = entry[1]

    value1 = []
    for outlink in outlinks:
        value1.append((outlink, PR / n))

    if (len(outlinks)==0):
        deadendaccumulator.add(PR)

    value1.append((p, outlinks))
    return value1


def updateReduce(next):
    # entry = ("0",[lijst outgoing])
    # PR(di) := /N + (1-)j=1..m PRj(di);
    # emit(key3: [di, PR(di)], value3: outlinks(di));
    p = next[0]

    PRlist = next[1]
    Outlinks = []

    PRSum = 0
    for i in PRlist:
        if (not isinstance(i, list)):
            PRSum += i
        else:
            Outlinks = i

    PR = teleportationProbability * 1 / N + (1 - teleportationProbability) * PRSum + (1-teleportationProbability) * deadendpool.value / N
    return ((p, PR), Outlinks)


def pageRank(data):
    data = data.map(lambda x: ((x[0], 1 / N), x[1]))

    rold = None
    rnew = data
    iter = 0
    diff = sys.float_info.max
    eps = 0.000001
    while diff>eps:
        timer = datetime.now()
        iter += 1
        rold = rnew
        rnew = rold.flatMap(updateIteration)

        global deadendpool
        val = deadendaccumulator.value
        deadendpool = sc.broadcast(val)
        deadendaccumulator.add(-val) #reset deadendpool


        rnew = rnew.groupByKey().map(updateReduce)
        #print(deadendpool)

        # calculate difference?
        if(iter>=0):
            mapped1=rnew.map(lambda x:x[0]).persist(storageLevel=StorageLevel.MEMORY_AND_DISK)
            mapped2=rold.map(lambda x:x[0]).persist(storageLevel=StorageLevel.MEMORY_AND_DISK)
            #print(mapped2.map(lambda x:x[1]).sum())
            diff=mapped1.join(mapped2)
            diff=diff.map(lambda x:abs(x[1][0]-x[1][1]))
            diff=diff.max()
            print("Iteration ", iter, " just finished with first order norm: ", diff, " approximated elapsed time = ",
                 (datetime.now() - timer).total_seconds())
            mapped1.unpersist()
            mapped2.unpersist()
        else:
            print("Iteration ", iter, " just finished, approximated elapsed time = ",
                 (datetime.now() - timer).total_seconds())

    # sort by descending order of page ranking values
    result = rnew.map(lambda x: x[0]).sortBy(lambda x: x[1], False)
    return result


def addEntries(entry):
    entry=entry.split("\t")
    yield (int(entry[0].strip()),[int(entry[1].strip())])
    yield (int(entry[1].strip()),[])

if __name__ == '__main__':
    conf = SparkConf()
    conf.set("spark.network.timeout", "36001s")
    conf.set("spark.executor.heartbeatInterval", "36000s")
    conf.set("spark.storage.blockManagerSlaveTimeoutMs", "36000s")
    conf.set("spark.worker.timeout", "36000s")
    conf.set("spark.sql.broadcastTimeout", "36000s")
    conf.set("spark.executor.memory", "5g")
    conf.set("spark.driver.memory", "8g")
    conf.set("spark.worker.cleanup.enabled", "true")
    sc = SparkContext("local[3]", "PageRanking",conf=conf)
    deadendaccumulator = sc.accumulator(0)


    # filtering mini database to be of form (from node, list of to nodes)
    data = sc.textFile("./Dataset/web-Google.txt").filter(lambda l: not str(l).startswith("#")) \
        .flatMap(addEntries)\
        .reduceByKey(lambda x,y:x+y)
    # data = sc.parallelize([(0, [2]), (1, [2]), (2, [0, 1, 3]), (3, [])])

    #    t=data.collect()

    result = pageRank(data)

    # write result to csv file
    result = result.map(lambda line: str(line)[1:-1])
    rankFile = open('ranking_google_'+str(teleportationProbability)+'.csv', 'w')
    rankFile.write("nodeID,pageRankScore\n")
    rankFile.write("\n".join(result.collect()))
