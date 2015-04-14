#!/bin/bash

cmake -E make_directory build && \
  cd build && \
  cmake .. && \
  make && \
  cp liboxnn_cuda.so ..
cd ..

# cd build && make install"
