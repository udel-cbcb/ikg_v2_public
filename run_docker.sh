docker run --rm --gpus=all \
  --name=ikg-dev \
  --ipc=host \
  --shm-size=256m \
  --user $(id -u):$(id -g) \
  --mount type=bind,source="$(pwd)",target=/home/datascience/workdir \
  --mount type=bind,source="$HOME/Projects/ikg_v2_data",target=/home/datascience/data/ \
  --mount type=bind,source="$HOME/.ssh",target=/home/datascience/.ssh \
  -it \
  -d \
  sachinx0e/ikg:3.0