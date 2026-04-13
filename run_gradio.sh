#!/bin/bash
CUDA_VISIBLE_DEVICES=1 python app_flux.py --name flux-krea-dev --t5_quant int8 "$@"
