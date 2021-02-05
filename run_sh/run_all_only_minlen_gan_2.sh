#! /bin/bash

minlens="1 2 3 4 5"

for minlen in ${minlens} ; do
  bash run_sh/run_only_minlen_gan_2.sh ${minlen}

done
exit 0