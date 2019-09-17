#!/bin/sh
java -Xmx4g -cp "en/stanford-corenlp-full-2018-10-05/*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9001 -timeout 30000
