#! /bin/bash

minlens="1 2 3 4 5"

for minlen in ${minlens} ; do
  bash run_sh/run_remove_entity_minlen_gan.sh ${minlen}

done
exit 0