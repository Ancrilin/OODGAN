#! /bin/bash

maxlens="10 11 12 13 14 15 16"

for maxlen in ${maxlens} ; do
  bash run_sh/run_only_maxlen_gan.sh ${maxlen}

done
exit 0