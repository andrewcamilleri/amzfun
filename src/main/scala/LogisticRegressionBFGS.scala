import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.sql.SparkSession


object LogisticRegressionBFGS {

  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder().getOrCreate()

    val trainData = spark.read.
      option("inferSchema", true).
      option("header", true).
      csv("hdfs:///tmp/train.csv")
    val test = spark.read.
      option("inferSchema", true).
      option("header", true).
      csv("hdfs:///tmp/test.csv")

    val points = trainData.map(line => {
      val label = line.getInt(0).toDouble
      val vector = line.toSeq.drop(1).map(l => l.asInstanceOf[Int].toDouble)
      LabeledPoint(label, Vectors.dense(vector.toArray))
    })

    // Build the model
    val model = new LogisticRegressionWithLBFGS().run(points.rdd)
  }
}