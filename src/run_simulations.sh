#!/bin/bash
NUM_RUNS=1

for i in $(seq 1 $NUM_RUNS)
do
   echo "================================="
   echo "Launching simulation run #$i"
   echo "================================="
   
   python -m src.simulate --run_id $i
done

echo "All simulations completed."