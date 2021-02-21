#! /bin/bash

maxlens="41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60"

for maxlen in ${maxlens} ; do
  bash run_sh/run_only_maxlen_gan.sh ${maxlen}

done
exit 0