(ns clj-dl4-examples.mnist-two-layers
  (:import [org.nd4j.linalg.activations Activation]
           [org.nd4j.linalg.dataset.api.iterator DataSetIterator]
           [org.deeplearning4j.datasets.iterator.impl MnistDataSetIterator]
           [org.deeplearning4j.eval Evaluation]
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
           [org.deeplearning4j.nn.multilayer MultiLayerNetwork]
           [org.deeplearning4j.nn.weights WeightInit]
           [org.deeplearning4j.optimize.api IterationListener]
           [org.deeplearning4j.optimize.listeners ScoreIterationListener]
           [org.nd4j.linalg.api.ndarray INDArray]
           [org.nd4j.linalg.dataset DataSet]
           [org.nd4j.linalg.dataset.api.iterator.DataSetIterator]
           [org.nd4j.linalg.lossfunctions LossFunctions LossFunctions$LossFunction]
           [java.io]
           [java.nio.file Files]
           [java.nio.file Paths]
           [java.util Arrays]
           [java.util Random]
           [org.apache.commons.io FileUtils])
  (:require [clj-dl4-examples.model-utils :as model-utils]))

(def ^Integer num-rows 28)
(def ^Integer num-columns 28)
(def ^Integer output-num 10)
(def ^Integer batch-size 64)
(def ^Integer seed 123)
(def ^Integer num-epochs 15)
(def ^Double rate 0.0015)
(def ^Integer listener-freq 1)

(def ^DataSetIterator iter-train (MnistDataSetIterator. batch-size true seed))
(def ^DataSetIterator iter-test (MnistDataSetIterator. batch-size false seed))

;;(def split-train-num (int (* batch-size 0.8)))
;; (def iter (MnistDataSetIterator. batch-size ???))
;; (def nxt (.next iter))
;; (def test-and-train (.splitTestAndTrain nxt split-train-num (Random. (int seed))))
;; (def train (.getTrain test-and-train))
;; (def tst (.getTest test-and-train))

(def conf
  (-> (NeuralNetConfiguration$Builder.)
      (.seed seed)
      (.optimizationAlgo OptimizationAlgorithm/STOCHASTIC_GRADIENT_DESCENT)
      (.iterations 1)
      (.activation "relu")
      (.weightInit WeightInit/XAVIER)
      (.learningRate rate)
      (.updater Updater/NESTEROVS)
      (.momentum 0.98)
      (.regularization true)
      (.l2 (* rate 0.005))
      (.list)
      (.layer 0
              (->
               (DenseLayer$Builder.)
               (.nIn (* num-rows num-columns))
               (.nOut 500)
               (.build)))
      (.layer 1
              (->
               (DenseLayer$Builder.)
               (.nIn 500)
               (.nOut 100)
               (.build)))
      (.layer 2
              (->
               (OutputLayer$Builder. LossFunctions$LossFunction/NEGATIVELOGLIKELIHOOD)
               (.activation "softmax")
               (.nIn 100)
               (.nOut output-num)
               (.build)))
      (.pretrain false)
      (.backprop true)
      (.build)))

(def model (MultiLayerNetwork. conf))

(defn all-cycle []
  (model-utils/pipe-visualization  model
                                   iter-train
                                   iter-test
                                   num-epochs
                                   output-num
                                   listener-freq))
