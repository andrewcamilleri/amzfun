import org.apache.spark.mllib.evaluation.RegressionMetrics
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.regression.{LabeledPoint, LinearRegressionWithSGD}
import org.apache.spark.sql.SparkSession
import org.apache.spark.mllib.linalg.DenseVector

object AmzPredict {

  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder().getOrCreate()

    val data = spark.read.
      option("inferSchema", true).
      option("header", false).
      csv("hdfs:///tmp/test.csv")
    val points = data.map(line => LabeledPoint(line.getDouble(0), new DenseVector(line.toSeq.drop(1).toArray[Double])))

    // Build the model
    val numIterations = 100
    val model = LinearRegressionWithSGD.train(points.rdd, numIterations)
  }
}