(ns clj-dl4-examples.ui
  (:import [org.deeplearning4j.api.storage StatsStorage]
           [org.deeplearning4j.optimize.listeners ScoreIterationListener]
           [org.deeplearning4j.ui.api UIServer Utils]
           [org.deeplearning4j.ui.stats StatsListener]
           [org.deeplearning4j.ui.storage InMemoryStatsStorage]
           [org.nd4j.linalg.activations Activation]
           [org.deeplearning4j.nn.multilayer MultiLayerNetwork]
           [org.nd4j.linalg.dataset.api.iterator DataSetIterator]
           [org.deeplearning4j.datasets.iterator.impl MnistDataSetIterator]
           [org.deeplearning4j.nn.api OptimizationAlgorithm]
           [org.deeplearning4j.nn.conf MultiLayerConfiguration
            Updater
            NeuralNetConfiguration
            NeuralNetConfiguration$Builder
            ]
           [org.deeplearning4j.nn.conf.layers OutputLayer
            OutputLayer$Builder
            DenseLayer$Builder
            DenseLayer
            ]
           [org.deeplearning4j.nn.weights WeightInit]
           [org.deeplearning4j.optimize.api IterationListener]
           [org.nd4j.linalg.api.ndarray INDArray]
           [org.nd4j.linalg.dataset DataSet]
           [org.nd4j.linalg.dataset.api.iterator.DataSetIterator]
           [org.nd4j.linalg.lossfunctions LossFunctions LossFunctions$LossFunction]

           [org.nd4j.linalg.dataset.api.iterator DataSetIterator]
           [org.deeplearning4j.eval Evaluation]
           [org.deeplearning4j.nn.multilayer MultiLayerNetwork]
           [org.deeplearning4j.optimize.listeners ScoreIterationListener])
  (:require [clj-dl4-examples.model-utils :as model-utils]))

(def num-rows 28)
(def num-columns 28)
(def output-num 10)
(def batch-size 128)
(def seed 123)
(def num-epochs 3)
(def listener-freq 1)

(def ^DataSetIterator iter-train (MnistDataSetIterator. batch-size true seed))
(def ^DataSetIterator iter-test (MnistDataSetIterator. batch-size false seed))

(def conf
  (-> (NeuralNetConfiguration$Builder.)
      (.seed seed)
      (.optimizationAlgo OptimizationAlgorithm/STOCHASTIC_GRADIENT_DESCENT)
      (.iterations 1)
      (.learningRate 0.006)
      (.l2 1e-4)
      (.regularization true)
      (.updater Updater/NESTEROVS)
      (.momentum 0.9)
      (.list)
      (.layer 0
              (->
               (DenseLayer$Builder.)
               (.nIn (* num-rows num-columns))
               (.nOut 1000)
               (.weightInit WeightInit/XAVIER)
               (.activation "relu")
               (.build)))
      (.layer 1
              (->
               (OutputLayer$Builder. LossFunctions$LossFunction/NEGATIVELOGLIKELIHOOD)
               (.nIn 1000)
               (.nOut output-num)
               (.activation "softmax")
               (.weightInit WeightInit/XAVIER)
               (.build)))
      (.pretrain false)
      (.backprop true)
      (.build)))

(def model (MultiLayerNetwork. conf))

(.init model)

(def server (UIServer/getInstance))

(def storage (new InMemoryStatsStorage))

(.setListeners model (list (StatsListener. storage listener-freq)))

(.attach server storage)

(defn train
  []
  (doseq [_ (range num-epochs)]
    (.fit model iter-train)))
