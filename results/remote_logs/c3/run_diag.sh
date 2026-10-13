#!/bin/bash
cd /root/c3
for c in results/campaign/c3_real/best_predictor.pt results/campaign/c3_sim/best_predictor_real.pt; do
  echo "CKPT $c"
  .venv/bin/python /root/diag_cem.py /root/c3 "$c" results/encoded/encoded_data.pt 1 4
  echo "EXIT=$?"
done
echo DIAG_DONE
