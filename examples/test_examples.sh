#!/bin/bash

OPT=" -seed 1 -maxiteration 10 "

echo "Running 'toy1.lua'..."
th toy1.lua -strict -train -datadir `pwd` -tmpdir `pwd` \
  -adagrad -adagrad_learningRate 0.1 $OPT

echo "Running 'toy1.lua' with cuda..."
th toy1.lua -strict -train -datadir `pwd` -tmpdir `pwd` \
  -adagrad -adagrad_learningRate 0.1 -cuda $OPT
for i in `seq 1 3`; do
  echo "Running 'toy1_fastlstmcell.lua' (version $i)..."
  th toy1_fastlstmcell.lua -strict -train -datadir `pwd` -tmpdir `pwd` \
    -adagrad -adagrad_learningRate 0.1 \
    -version $i $OPT
  echo "Running 'toy1_fastlstmcell.lua' with cuda (version $i)..."
  th toy1_fastlstmcell.lua -strict -train -datadir `pwd` -tmpdir `pwd` \
    -adagrad -adagrad_learningRate 0.1 -cuda \
    -version $i $OPT
done
for i in `seq 0 2`; do
  echo "Running 'toy1_deeplstm.lua' (version $i)..."
  th toy1_deeplstm.lua -strict -train -datadir `pwd` -tmpdir `pwd` \
    -adagrad -adagrad_learningRate 0.1 \
    -version $i $OPT
  echo "Running 'toy1_deeplstm.lua' with cuda (version $i)..."
  th toy1_deeplstm.lua -strict -train -datadir `pwd` -tmpdir `pwd` \
    -adagrad -adagrad_learningRate 0.1 -cuda \
    -version $i $OPT
done
