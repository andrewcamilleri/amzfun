import org.apache.spark.ml.classification.{BinaryLogisticRegressionSummary, LogisticRegression}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.SparkSession

/**
  * Created by Andrew on 23/09/2017.
  */
object LogisticRegression {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder().getOrCreate()

    val trainData = spark.read.
      option("inferSchema", true).
      option("header", true).
      csv("hdfs:///tmp/train.csv")
    val testData = spark.read.
      option("inferSchema", true).
      option("header", true).
      csv("hdfs:///tmp/test.csv")

    val inputCols = trainData.columns.filter(_ != "ACTION")
    val assembler = new VectorAssembler().
      setInputCols(inputCols).
      setOutputCol("featureVector")
    val featureVector = assembler.transform(trainData)

    val evaluations =
      for (elasticNet <- Seq(0.0); regParam <- (0.0 to 20 by 0.1).toSeq)
        yield {
          val mlr = new LogisticRegression()
            .setMaxIter(1000)
            .setFeaturesCol("featureVector")
            .setLabelCol("ACTION")
            .setPredictionCol("prediction")
            .setElasticNetParam(elasticNet)
            .setRegParam(regParam)

          val lrModel = mlr.fit(featureVector)
          val trainingSummary = lrModel.summary
          val objectiveHistory = trainingSummary.objectiveHistory
          val binarySummary = trainingSummary.asInstanceOf[BinaryLogisticRegressionSummary]
          (binarySummary.areaUnderROC, elasticNet, regParam, lrModel, binarySummary.roc)
        }

    val best = evaluations.sortBy(_._1).reverse(0)._4

    val testCols = testData.columns.filter(_ != "id")
    val testAssembler = new VectorAssembler().
      setInputCols(testCols).
      setOutputCol("featureVector")
    val testVector = testAssembler.transform(testData)

    val predictions = best.transform(testVector)
    val result = predictions.select("id", "prediction", "probability")

  }
}
