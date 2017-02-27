(ns clj-dl4-examples.model-utils
  (:import [org.nd4j.linalg.dataset.api.iterator DataSetIterator]
           [org.deeplearning4j.eval Evaluation]
           [org.deeplearning4j.nn.multilayer MultiLayerNetwork]
           [org.deeplearning4j.optimize.listeners ScoreIterationListener]))

(defn init-model
  [model]
  (.init model))

(defn set-listener
  [model ^Integer listener-freq]
  (->> (ScoreIterationListener. listener-freq)
       (list)
       (.setListeners model)))

(defn train
  [model ^DataSetIterator iter-train num-epochs]
  (doseq [_ (range num-epochs)]
    (.fit model iter-train)))

(defn evaluate
  [model ^DataSetIterator iter-test output-num]
  (let [evaluation (Evaluation. output-num)]
    (while (.hasNext iter-test)
      (let [n (.next iter-test)]
        (->> (.getFeatureMatrix n)
             (.output model)
             (.eval evaluation (.getLabels n)))))
    (println (.stats evaluation))))

(defn small-pipe
  [^MultiLayerNetwork model ^DataSetIterator iter-train ^DataSetIterator iter-test
   ^Integer num-epochs ^Integer output-num ^Integer listener-freq]
  (init-model model)
  (set-listener model listener-freq)
  (train model iter-train num-epochs)
  (evaluate model iter-test output-num))
