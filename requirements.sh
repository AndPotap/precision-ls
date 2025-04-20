py -m pip install -U pip
py -m pip install torch

py -m pip install einops matplotlib numpy pandas scikit-learn seaborn tqdm wandb
py -m pip install pyyaml Cerberus cytoolz funcy gin_config toposort munch

py -m pip install "python-lsp-server[all]" --no-cache-dir
py -m pip install ruff
py -m pip uninstall autopep8 -y
py -m pip install python-lsp-ruff --no-cache-dir
py -m pip install pre-commit

py -m pip install ~/gpustat/.
