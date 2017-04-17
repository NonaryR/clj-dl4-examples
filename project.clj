(defproject clj-dl4-examples "0.1.0-SNAPSHOT"
  :description "FIXME: write description"
  :dependencies [[org.clojure/clojure "1.9.0-alpha14"]
                 [commons-io/commons-io "2.4"]
                 [com.google.guava/guava "21.0"]
                 [org.slf4j/slf4j-nop "1.7.22" :exclusions [org.slf4j/slf4j-api]]
                 [org.eclipse.collections/eclipse-collections "8.0.0"]
                 [org.eclipse.collections/eclipse-collections-api "8.0.0"]
                 [org.eclipse.collections/eclipse-collections-forkjoin "8.0.0"]
                 ;; DL4J deps
                 [org.deeplearning4j/deeplearning4j-core "0.8.0"
                  :exclusions [com.google.guava/guava
                               ch.qos.logback/logback-classic
                               commons-io
                               org.apache.commons/commons-compress]]
                 [org.datavec/datavec-api "0.8.0"]
                 [org.nd4j/nd4j-native-platform "0.8.0"]
                 [org.deeplearning4j/deeplearning4j-nlp "0.8.0"]
                 [org.deeplearning4j/deeplearning4j-modelimport "0.8.0"]
                 [org.deeplearning4j/deeplearning4j-nn "0.8.0"]
                 [org.deeplearning4j/deeplearning4j-ui_2.10 "0.8.0"]
                 [org.deeplearning4j/gym-java-client "0.8.0"]
                 ]
  :target-path "target/%s"
  :profiles {:uberjar {:aot :all}})
