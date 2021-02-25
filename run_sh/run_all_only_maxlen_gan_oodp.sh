#! /bin/bash

maxlens="10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25"

for maxlen in ${maxlens} ; do
  bash run_sh/run_only_maxlen_gan_oodp.sh ${maxlen}

done
exit 0