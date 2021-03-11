#! /bin/bash

pseudos="0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0"

for pseudo in ${pseudos} ; do
  bash run_sh/run_gan_oodp_true_alpha0.85_remove_entity_pseudo_weight.sh ${pseudo}

done
exit 0