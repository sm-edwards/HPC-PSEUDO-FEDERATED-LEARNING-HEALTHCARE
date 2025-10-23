# HPC Project — Part 1: Data Locality and Labeled Prefetcher

This project benchmarks multiple **data locality mechanisms** for efficient data movement and training throughput in PyTorch, using ECG windows as a test workload.  
It measures the effect of cache-friendly batching, pinned memory, and asynchronous transfers — culminating in the LABL prefetcher (A4).

---

# Dependencies

Create a clean conda environment first:

```bash
conda create -n hpc-part1 python=3.9
conda activate hpc-part1

pip install torch==2.1.0 numpy pandas matplotlib wfdb tqdm
pip install seaborn

VERIFY CUDA WITH
python -m torch.utils.collect_env

```

```` Directory Layout
PART-1/
├── src/
│   ├── shard_prep.py
│   ├── labl_loader.py
│   ├── train_ecg_labl.py
│   ├── bench_locality.py
│   ├── plot_all_results.py
│   └── utils/
├── data/
│   └── shards/
├── results/
│   ├── part1_locality_results.csv
│   ├── part1_labl_results.csv
│   ├── part1_all_results.csv
│   ├── shard_prep_metrics.json
│   ├── throughput_comparison_A0_A4.png
│   └── time_breakdown_batch512_A0_A4.png
````

