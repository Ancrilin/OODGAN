#! /bin/bash

maxlens="18 19 20 21"

for maxlen in ${maxlens} ; do
  bash run_sh/run_only_maxlen_gan_lr2e5.sh ${maxlen}

done
exit 0