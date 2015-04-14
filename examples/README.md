Sample commands to run examples:
```bash
cd examples
th toy1.lua -strict -train -datadir `pwd` -tmpdir `pwd` -adagrad -adagrad_learningRate 0.1
th toy1_fastlstmcell.lua -strict -train -datadir `pwd` -tmpdir `pwd` -adagrad -adagrad_learningRate 0.1 -cuda
```
