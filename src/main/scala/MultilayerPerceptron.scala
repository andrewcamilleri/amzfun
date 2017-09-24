package scala

import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.SparkSession

/**
  * Created by Andrew on 24/09/2017.
  */
object MultilayerPerceptron {
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

    val layers = Array[Int](9, 5, 4, 2)


    val trainer = new MultilayerPerceptronClassifier()
      .setLayers(layers)
      .setBlockSize(128)
      .setSeed(1234L)
      .setMaxIter(100)
      .setFeaturesCol("featureVector")
      .setLabelCol("ACTION")
      .setPredictionCol("prediction")

    val model = trainer.fit(featureVector)

    val testCols = testData.columns.filter(_ != "id")
    val testAssembler = new VectorAssembler().
      setInputCols(testCols).
      setOutputCol("featureVector")
    val testVector = testAssembler.transform(testData)

    val result = model.transform(testVector)
    val predictionAndLabels = result.select("prediction", "label")
    val evaluator = new MulticlassClassificationEvaluator()
      .setMetricName("accuracy")

    println("Test set accuracy = " + evaluator.evaluate(predictionAndLabels))
  }
}
