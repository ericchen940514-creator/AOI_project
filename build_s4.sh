#!/bin/bash
source /home/user0514/miniconda3/etc/profile.d/conda.sh
conda activate fdtdx
CP=/home/user0514/miniconda3/envs/fdtdx

export CFLAGS="-Wno-incompatible-pointer-types -I$CP/include -I$CP/include/suitesparse"
export CXXFLAGS="-Wno-incompatible-pointer-types -I$CP/include -I$CP/include/suitesparse"
export LDFLAGS="-L$CP/lib"

cd /home/user0514/S4_src
make clean 2>/dev/null
make S4_pyext \
  BOOST_INC="-I$CP/include" \
  BOOST_LIBS="-L$CP/lib -lboost_serialization" \
  CHOLMOD_INC="-I$CP/include/suitesparse" \
  CHOLMOD_LIB="-L$CP/lib -lcholmod -lamd -lcolamd -lcamd -lccolamd" \
  2>&1
echo "Exit: $?"
