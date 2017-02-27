(ns clj-dl4-examples.mnist-single-layer
  (:require [clj-dl4-examples.model-utils :as model-utils])
  (:import [org.nd4j.linalg.activations Activation]
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
           [org.nd4j.linalg.lossfunctions LossFunctions LossFunctions$LossFunction]))

(def num-rows 28)
(def num-columns 28)
(def output-num 10)
(def batch-size 128)
(def seed 123)
(def num-epochs 3)
(def listener-freq 1)
;; (def num-examples 50000)

(def ^DataSetIterator iter-train (MnistDataSetIterator. batch-size true seed))
(def ^DataSetIterator iter-test (MnistDataSetIterator. batch-size false seed))

;; (def iter (MnistDataSetIterator. batch-size num-examples false))
;; (def split-train-num (int (* batch-size 0.8)))
;; (def nxt (.next iter))
;; (def test-and-train (.splitTestAndTrain nxt split-train-num (java.util.Random. (int seed))))
;; (def train (.getTrain test-and-train))
;; (def tst (.getTest test-and-train))

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

(model-utils/small-pipe model
                        iter-train
                        iter-test
                        num-epochs
                        output-num
                        listener-freq)
