import sys
from datetime import datetime

from pyspark import SparkContext, SparkConf, StorageLevel
from Config import teleportationProbability

N = 875713
deadendpool = 0.0

def updateIteration(entry):
    # entry = ("0",[lijst outgoing])
    n = len(entry[1])
    p = int(entry[0][0])
    PR = float(entry[0][1])
    outlinks = entry[1]

    value1 = []
    for outlink in outlinks:
        value1.append((outlink, PR / n))

    # if outlink is empty, add PR to the deadend pool
    if (len(outlinks) == 0):
        global deadendpool
        deadendpool += PR

    value1.append((p, outlinks))
    return value1


def updateReduce(next):
    # entry = ("0",[lijst outgoing])
    # PR(di) := /N + (1-)j=1..m PRj(di);
    # emit(key3: [di, PR(di)], value3: outlinks(di));
    PRlist = next[1]
    p = next[0]
    Outlinks = []

    PRSum = 0
    for i in PRlist:
        if (not isinstance(i, list)):
            PRSum += i
        else:
            Outlinks = i

    PR = teleportationProbability * 1 / N + (1 - teleportationProbability) * PRSum + (1-teleportationProbability) * deadendpool / N
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
        rnew = rold.flatMap(updateIteration).groupByKey().map(updateReduce)

        # calculate difference?
        if(iter>=0):
            mapped1=rnew.map(lambda x:x[0]).persist(storageLevel=StorageLevel.MEMORY_AND_DISK)
            mapped2=rold.map(lambda x:x[0]).persist(storageLevel=StorageLevel.MEMORY_AND_DISK)
            diff=mapped1.join(mapped2).map(lambda x:abs(x[1][0]-x[1][1]))
            diff=diff.sum()
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
    sc = SparkContext("local[*]", "PageRanking",conf=conf)


    # filtering mini database to be of form (from node, list of to nodes)
    data = sc.textFile("./Dataset/web-Google.txt").filter(lambda l: not str(l).startswith("#")) \
        .map(lambda x: x.split("\t")).map(lambda x: (int(x[0].strip()), int(x[1].strip()))).groupBy(
        lambda x: x[0]).mapValues(lambda x: list(map(lambda y: int(y[1]), x)))

    result = pageRank(data)

    # write result to csv file
    result = result.map(lambda line: str(line)[1:-1])
    rankFile = open('ranking_google_'+str(teleportationProbability)+'.csv', 'w')
    rankFile.write("nodeID,pageRankScore\n")
    rankFile.write("\n".join(result.collect()))
