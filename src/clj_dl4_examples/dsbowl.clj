(ns clj-dl4-examples.dsbowl
  (:require [clj-dl4-examples.model-utils :refer [pipe-visualization]])
  (:import [org.nd4j.linalg.dataset.api.iterator DataSetIterator]
           [org.deeplearning4j.nn.api OptimizationAlgorithm]
           ;; Layers
           [org.deeplearning4j.nn.conf MultiLayerConfiguration
            Updater
            NeuralNetConfiguration
            NeuralNetConfiguration$Builder]

           [org.deeplearning4j.nn.weights WeightInit]
           [org.deeplearning4j.nn.conf.layers
            ActivationLayer ActivationLayer$Builder
            OutputLayer OutputLayer$Builder
            DenseLayer DenseLayer$Builder
            DropoutLayer DropoutLayer$Builder
            ConvolutionLayer ConvolutionLayer$Builder
            ZeroPaddingLayer ZeroPaddingLayer$Builder
            SubsamplingLayer SubsamplingLayer$Builder SubsamplingLayer$PoolingType]

           [org.nd4j.linalg.lossfunctions LossFunctions LossFunctions$LossFunction]
           [org.nd4j.linalg.learning Adam]
           [org.nd4j.linalg.activations Activation]
           [org.deeplearning4j.nn.multilayer MultiLayerNetwork]
           [org.deeplearning4j.util LayerValidation SummaryStatistics]))
(def conf
  (-> (NeuralNetConfiguration$Builder.)
      (list)
      (.layer 0
              (->
               (DenseLayer$Builder.)
               (.batchSize 30)
               (.activation Activation/TANH)
               (.build)))
      (.layer 1
              (->
               (ConvolutionLayer$Builder.)
               ()))))