# Getting started

### Requirements
1. [Docker](https://docs.docker.com/engine/install/debian/) - 20.10.10
2. [Determined AI](https://github.com/determined-ai/determined) - 0.17.2
3. [Polyxon CE](https://github.com/polyaxon/polyaxon) - 1.11.2 
4. RAM - 64 GB
5. CPU - 16 cores
5. Nvidia GPU with support for atleast CUDA 10.0 and atleast 16 GB memory (RTX A4000)

<br />

### Folder Structure
| | |
|---|---|
| `src/data` | Source code of the scripts used for generating data for experiments. |
| `src/models` | Source code of the unsupervised models described in the study. |
| `src/experiments` | Source code of the supervised models described in the study. |
| `configs` | Determined configuration files for running experiments on the determined cluster. |
| `polyaxon_configs` | Polyaxon configuration files for running experiments on the k8s cluster. |
| `main.py` | Entry point script used to run scripts locally. |

<br />

### Getting started.

1. Download and extract the data needed to run experiments
```
# download the data
wget https://research.bioinformatics.udel.edu/iptmnet_data/downloads/ikg_v2_data.tar.gz

# extract to /data/ml_data/ikg_v2_data
tar -xf ikg_v2_data.tar.gz r -C /data/ml_data/ikg_v2_data
```

2. Before starting the experiments, you need to build the docker containers.
```
# change to the docker directory
cd docker

# build the docker container
bash build.sh
```

3. Start the docker container and open an interactive shell into it.
```
# start the container
docker-compose up -d

# start an interactive shell into the container
docker exec -it ikg-dev /bin/bash
```

4. Generate an embedding using triple walk skip gram algorithm
```
det experiment create configs/triples_walk_embedder_const_mul_seeds.yaml .
```

5. Once the training is complete, retrieve the embeddings from checkpoint folder. The embeddings will be in the form `protein_embedding_{fold_number}.csv`.


6. To generate predictions use the polyaxon configs to run the prediction tasks on the kubernetes cluster.
```
# run the prediction task on the kubernetes cluster
polyaxon run -p ikg_v2 -f ./polyaxon_configs/make_predictions.yml -u

# run the prediction task locally
export POLYAXON_NO_OP=0
python src/main.py -c "make_predictions"

# After the task completes, check for a file named `predicted_edges.csv` in the root folder of this project.
``` 
