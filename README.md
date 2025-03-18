# Towards Learning High-Precision Least Squares Algorithms with Sequence Models

![Prior work focuses on statistical least squares: Transformers approximate Bayes-optimal estimators (left, adapted from Garg et al. 2022). In this work, we focus on numerical least squares: Transformers struggle to obtain precise solutions (inset). Using a high-precision training recipe, we train two polynomial architectures to perform high-precision gradient descent iterates on least squares (right): applied iteratively, they reach $10^{-13}$ MSE.](assets/banner_fig.png)

This repository contains code for the following paper:

> **Towards Learning High-Precision Least Squares Algorithms with Sequence Models.**
>
> Jerry Liu, Jessica Grogan, Owen Dugan, Ashish Rao, Simran Arora, Atri Rudra, Chris Ré.
> ICLR 2025.
> [arXiv](https://arxiv.org/abs/2503.12295)

## Dependencies
Install dependencies with
```
conda create -n "precision-ls" python=3.10
conda activate precision-ls
pip install -r requirements.txt
```

## Code structure
The code is organized as follows:
- `notebooks/`: contains notebooks for the experiments
- `replication_scripts/`: contains scripts for running the experiments
- `src/datagen/`: contains code for data generation (tasks and samplers)
- `src/models/`: contains code for the models, including different sequence mixers
- `src/schedulers/`: contains code for the learning rate schedulers, including the adaptive one used in the paper

## Running experiments
To test the data generation, run
```
bash src/datagen/test.sh
```

To run the least squares experiments, run
```
bash replication_scripts/least_squares.sh
```

To run the experiments and ablations with linear algebra primitives, run
```
bash replication_scripts/primitives.sh
```

To run the experiments and ablations with the explicit gradient and k-iterate gradient descent tasks, run
```
bash replication_scripts/explicit_gradient.sh
```

To run the experiments with in-context ODEs, run
```
bash replication_scripts/odes.sh
```

## Citation
If you find this work useful, please cite it as follows:
```
@misc{liu2025learninghighprecisionsquaresalgorithms,
      title={Towards Learning High-Precision Least Squares Algorithms with Sequence Models}, 
      author={Jerry Liu and Jessica Grogan and Owen Dugan and Ashish Rao and Simran Arora and Atri Rudra and Christopher Ré},
      year={2025},
      eprint={2503.12295},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2503.12295}, 
}
```
