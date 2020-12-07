from pyspark import SparkContext
from pyspark.sql import Row, SparkSession
from Config import teleportationProbability


def pageRank():
    return True


if __name__ == '__main__':
    sc = SparkContext("local", "PageRanking")

    #filtering mini database to be of form (from node, list of to nodes)
    data = sc.textFile("./Dataset/web-Google.txt").filter(lambda l: not str(l).startswith("#")) \
        .map(lambda x: x.split("\t")).map(lambda x: (str(x[0].strip()), str(x[1].strip()))).groupBy(
        lambda x: x[0]).mapValues(lambda x: list(map(lambda y:y[1],x)))

    #test
    t = data.filter(lambda x: x[0] == "0").first()
    print(t)
    #endtest


    words = sc.parallelize(
        ["scala",
         "java",
         "hadoop",
         "spark",
         "akka",
         "spark vs hadoop",
         "pyspark",
         "pyspark and spark"]
    )
    counts = words.count()
    print("Number of elements in RDD -> %i" % (counts))
