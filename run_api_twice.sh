#!/bin/bash

python -m sim.api_runner --sessions 3 --stagger 2 --scheduler mapreduce_events --rpm 20 --tpm 200000 --base-url https://llmgateway.app

sleep 20

python -m sim.api_runner --sessions 8 --stagger 2 --scheduler mapreduce_events --rpm 20 --tpm 200000 --base-url https://llmgateway.app
