# Goal

The goal of this demo is to experiment the below
   * Provision the Nebuis soperators cluster
   * Benchmark atleast two InferenceX benchmarks for difference configurations for throughput and latency
   * Analyze the observed benchmark results in comparision with publiched InferenceX results

# Experiment steps

The repro has the steps to setup a Nebuis soperators GPU cluster, run two InferenceX benchmarks and process the benchmark results

1. Provision Nebuis soperator cluster by following the steps [here](https://github.com/nebius/nebius-solutions-library/tree/main/soperator)
   - The [demotf folder](https://github.com/heyram74/nebuis-sopertors-inferencex-benchmark/tree/main/demotf) has the copy of the soperator terraform used for this demo.
   - It creates 2 node H200 GPU cluster with 3.6TB high speed nfs share
  
2. Verify the cluster health by running nccl tests from /opt/slurm-test/quickcheck. Verify the all-reduce-perf BW is matching the Nvidia published numbers.
   - The [nccl_results](https://github.com/heyram74/nebuis-sopertors-inferencex-benchmark/tree/main/results/nccl_results) folder has all the resuts from the demo cluster
  
3. SSH to the login node and verify the file system and its allocated size. Below is the snapshot from the demo cluster
    ~~~
    # df -h
    Filesystem                                                                         Size  Used Avail Use% Mounted on
    jail                                                                               512G  243G  270G  48% /
    ....
    nfs-server.nfs-system.svc.cluster.local:/pvc-ac42b148-5376-4c66-a286-0d148fb5f45a  3.6T  1.1T  2.6T  30% /home
    ....
    jail-submount-data                                                                 512G     0  512G   0% /mnt/data
    ~~~

    The high speed nfs should show up as one file mount as /home. This will be used to store all the artifacts of this experiments. Below is the folder structure used for this experiment.
    ~~~
        /home/InferenceX
        /home/cache
        /home/containers
        /home/models
        /home/results
        /home/scripts
    ~~~
    
5. Clone [InferenceX](https://github.com/SemiAnalysisAI/InferenceX.git) into /home/InferenceX
  
6. Download the models to /home/models folder

   ~~~
   hf download openai/gpt-oss-120b --local-dir /home/models/openai--gpt-oss-120b

   hf download deepseek-ai/DeepSeek-R1-0528 --local-dir /home/models/deepseek-r1
   ~~~
  
7. Clone [this repo's script](https://github.com/heyram74/nebius_sopertors_inferencex_benchmark/tree/main/scripts) folder into into /home/scripts
  
8. Edit the scripts as required for the cluster specific fileshare and mount. For reference below is the    
  
9. Run the gpt_oss benchmark test with "SBATCH /home/scripts/run_gpt.sh" with sweep of required TP and CONC values in the script
  
10. Run the deepseekr1 benchmark test with "SBATCH /home/scripts/run_deepseekr1.sh" with sweep of required TP and CONC values in the script
   
11. Process the results files in /home/results by running
   ~~~
   python /home/script/analyze_benchmark.py --results-dir=/home/results/gptoss_h200_vllm22.0_sweep1

   python /home/script/analyze_benchmark.py --results-dir=/home/results/deepseekr1_h200_sglang5.12_sweep1
   ~~~
   
   This creates the below in the given results-dir 
      * summary.txt with a metrics table
      * chart for throughout vs interactivity
      * charts for throughout vs latency

# Results Analysis

InferenceX publishes the benchmark results on continous frequency at https://inferencex.semianalysis.com/inference. For this analysis purpose, took the snapshot of the two benchmarks and placed it under the results folder

* Deepseek R1 benchmark numbers closely follow the baseline pattern with throughput numbers around 10% lower
* gpt_oss120B 8GPU showing very low throughput, interactivity and latency numbers - this might need another rerun of the experiments and deepdive