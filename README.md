# nebius_sopertors_inferencex_benchmark
The repro has the steps to setup a Nebuis soperators GPU cluster, run two InferenceX benchmarks and process the benchmark results

1. Provision Nebuis soperator cluster by following the steps [here](https://github.com/nebius/nebius-solutions-library/tree/main/soperator)
  
3. Verify the cluster health by running nccl tests from /opt/slurm-test/quickcheck. Verify the all-reduce-perf BW is matching the Nvidia published numbers.
  
4. SSH to the login node and clone [InferenceX](https://github.com/SemiAnalysisAI/InferenceX.git) into /home/InferenceX
  
5. Down the models to /home/models folder
   hf download openai/gpt-oss-120b --local-dir /home/models/openai--gpt-oss-120b
   hf download deepseek-ai/DeepSeek-R1-0528 --local-dir /home/models/deepseek-r1
  
6. Clone [this repo's script](https://github.com/heyram74/nebius_sopertors_inferencex_benchmark/tree/main/scripts) folder into into /home/scripts
  
7. Edit the scripts as required for the cluster specific fileshare information  
  
8. Run the gpt_oss benchmark test with "SBATCH /home/scripts/run_gpt.sh" with sweep of required TP and CONC values in the script
  
9. Run the deepseekr1 benchmark test with "SBATCH /home/scripts/run_deepseekr1.sh" with sweep of required TP and CONC values in the script
   
10. Process the results files in /home/results by running
   python /home/script/analyze_benchmark.py --results-dir=/home/results/gpt_oss_sweep1
