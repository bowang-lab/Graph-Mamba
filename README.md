<p align="center">
<img src="/images/title_page.png"/> 
</p>


Attention mechanisms have been widely used to capture long-range dependencies among nodes in Graph Transformers. Bottlenecked by the quadratic computational cost, attention mechanisms fail to scale in large graphs. Recent improvements in computational efficiency are mainly achieved by attention sparsification with random or heuristic-based graph subsampling, which falls short in data-dependent context reasoning. State space models (SSMs), such as Mamba, have gained prominence for their effectiveness and efficiency in modeling long-range dependencies in sequential data. However, adapting SSMs to non-sequential graph data presents a notable challenge. 

In this work, we introduce Graph-Mamba, the first attempt to enhance long-range context modeling in graph networks by integrating a Mamba block with the input-dependent node selection mechanism. Specifically, we formulate graph-centric node prioritization and permutation strategies to enhance context-aware reasoning, leading to a substantial improvement in predictive performance. Extensive experiments on ten benchmark datasets demonstrate that Graph-Mamba outperforms state-of-the-art methods in long-range graph prediction tasks, with a fraction of the computational cost in both FLOPs and GPU memory consumption.

<p align="center">
<img src="/images/main_results.png"/> 
</p>

### Python environment setup with Conda

```bash
conda create --name graph-mamba --file requirements_conda.txt
conda activate graph-mamba
conda clean --all
```
To troubleshoot Mamba installation, please refer to https://github.com/state-spaces/mamba.

### Running Graph-Mamba
```bash
conda activate graph-mamba

# Running Graph-Mamba for Peptides-func dataset
python main.py --cfg configs/Mamba/peptides-func-EX.yaml  wandb.use False
```
You can also set your wandb settings and use wandb.

### Guide on configs files

Most of the configs are shared with [GraphGPS](https://github.com/rampasek/GraphGPS) and [Exphormer](https://github.com/hamed1375/Exphormer) code. You can change the following parameters in the config files for different parameters and variants of Graph-Mamba:
```
gt:
  layer_type: CustomGatedGCN+Mamba_Hybrid_Degree_Noise
  # Refer to graphgps/layer/gps_layer.py for NUM_BUCKETS
  # CustomGatedGCN+Mamba_Hybrid_Degree_Noise_Bucket - For large graph datasets that use the bucketing technique
  # CustomGatedGCN+Mamba_Hybrid_Noise - For permutation-only Graph-Mamba
  # CustomGatedGCN+Mamba - For baseline Mamba without Graph-Mamba adaptations
```
