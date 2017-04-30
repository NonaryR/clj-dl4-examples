(ns clj-dl4-examples.model-utils
  (:import [org.nd4j.linalg.dataset.api.iterator DataSetIterator]
           [org.deeplearning4j.eval Evaluation]
           [org.deeplearning4j.nn.multilayer MultiLayerNetwork]
           [org.deeplearning4j.optimize.listeners ScoreIterationListener]
           [org.deeplearning4j.ui.api UIServer Utils]
           [org.deeplearning4j.ui.stats StatsListener]
           [org.deeplearning4j.ui.storage InMemoryStatsStorage]))


(defn init-model
  [model]
  (.init model))

(defmulti set-listener
  (fn [_ _ vis]
    (if
      (= :visualise vis) :visualise
      :without-vis)))

(defmethod set-listener :without-vis
  [model ^Integer listener-freq _]
  (->> (ScoreIterationListener. listener-freq)
       (list)
       (.setListeners model)))

(defmethod set-listener :visualise
  [model ^Integer listener-freq _]
  (let [server (UIServer/getInstance)
        storage (InMemoryStatsStorage.)]
    (->> listener-freq
         (StatsListener. storage)
         (list)
         (.setListeners model))
    (.attach server storage)))

(defn train
  [model ^DataSetIterator iter-train num-epochs]
  (dotimes [_ num-epochs]
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

(defn pipe-visualization
  [^MultiLayerNetwork model ^DataSetIterator iter-train ^DataSetIterator iter-test
   ^Integer num-epochs ^Integer output-num ^Integer listener-freq]
  (init-model model)
  (set-listener model listener-freq :visualise)
  (train model iter-train num-epochs)
  (evaluate model iter-test output-num))
