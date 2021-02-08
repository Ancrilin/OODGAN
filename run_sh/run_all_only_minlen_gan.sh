#! /bin/bash

minlens="6 7 8 9 10"

for minlen in ${minlens} ; do
  bash run_sh/run_only_minlen_gan.sh ${minlen}

done
exit 0