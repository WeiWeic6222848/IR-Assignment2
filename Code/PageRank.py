import sys
from datetime import datetime

from pyspark import SparkContext, SparkConf
from Config import teleportationProbability


def updateIteration(entry):
    # entry = ("0",[lijst outgoing])
    n = len(entry[1])
    p = int(entry[0][0])
    PR = float(entry[0][1])
    outlinks = entry[1]

    value1 = []
    for outlink in outlinks:
        value1.append((outlink, PR / n))

    value1.append((p, outlinks))
    return value1


def updateReduce(next):
    N = 50000000
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

    PR = teleportationProbability * 1 / N + (1 - teleportationProbability) * PRSum
    return ((p, PR), Outlinks)


def pageRank(data):
    N = 50000000
    data = data.map(lambda x: ((x[0], 1 / N), x[1]))

    rold = None
    rnew = data
    iter = 0
    diff = sys.float_info.max
    eps = 0.000001
    while iter<1:
        timer = datetime.now()
        iter += 1
        rold = rnew.persist()
        rnew = rold.flatMap(updateIteration).groupByKey().map(updateReduce).persist()

        # calculate difference?
        diff=rnew.map(lambda x:x[0]).join(rold.map(lambda x:x[0])).map(lambda x:abs(x[1][0]-x[1][1]))
        diff=diff.max()
        print(diff)
        # print("Iteration ", iter, " just finished with diff: ", diff, " approximated elapsed time = ",
        #      (datetime.now() - timer).total_seconds())
        rold.unpersist()
        rnew.unpersist()

    # sort by descending order of page ranking values
    result = rnew.map(lambda x: x[0]).sortBy(lambda x: x[1], False)
    # print(result)
    return result


if __name__ == '__main__':
    conf = SparkConf()
    conf.set("spark.network.timeout", "36000s")
    conf.set("spark.executor.heartbeatInterval","3600s")
    conf.set("spark.storage.blockManagerSlaveTimeoutMs","3600s")
    conf.set("spark.worker.timeout","3600s")
    conf.set("spark.sql.broadcastTimeout","3600s")
    sc = SparkContext("local[*]", "PageRanking",conf=conf)


    # filtering mini database to be of form (from node, list of to nodes)
    data = sc.textFile("./Dataset/ClueWeb09_WG_50m.graph-txt").zipWithIndex()
    data = data.filter(lambda x: int(x[1])!=0 and int(x[1])<=10).map(lambda x: (
        int(x[1])-1, list(map(lambda x: int(x), filter(lambda x: x != "" and int(x) <= 10, str(x[0]).split(" "))))))

    # tmp=data.count()
    # print(tmp)
    #
    # tmp=data.collect()
    # print(tmp)

    # data = sc.textFile("./Dataset/web-Google.txt").filter(lambda l: not str(l).startswith("#")) \
    #     .map(lambda x: x.split("\t")).map(lambda x: (int(x[0].strip()), int(x[1].strip()))).groupBy(
    #     lambda x: x[0]).mapValues(lambda x: list(map(lambda y: int(y[1]), x)))
    # # (0,[]) <-
    # # (score) <- rold



    result = pageRank(data)

    # write result to csv file
    result = result.map(lambda line: str(line))
    rankFile = open('ranking_clueweb_0.15.csv', 'w')
    rankFile.write("\n".join(result.collect()))
