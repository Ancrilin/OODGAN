#! /bin/bash

maxlens="17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40"

for maxlen in ${maxlens} ; do
  bash run_sh/run_only_maxlen_gan.sh ${maxlen}

done
exit 0

minlens="1 2 3 4 5"

for minlen in ${minlens} ; do
  bash run_sh/run_only_minlen_gan.sh ${minlen}

done
exit 0

for minlen in ${minlens} ; do
  bash run_sh/run_remove_entity_minlen_gan.sh ${minlen}

done
exit 0