FROM determinedai/environments:cuda-11.1-pytorch-1.9-lightning-1.3-tf-2.4-gpu-0.16.4

# install system deps
RUN apt install nano \
    && conda install numba -y \
    && curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/install-poetry.py > /tmp/install-poetry \
    && python /tmp/install-poetry --version 1.1.9

# add poetry to path
ENV PATH=/root/.local/bin:$PATH

# Create a working directory
RUN mkdir -p /root/workdir/ && \
    mkdir -p /root/data/

WORKDIR /root/workdir

COPY pyproject.toml .

# disable venv and install deps
RUN poetry config virtualenvs.create false \
    && poetry install \
    && pip install https://github.com/Sachinx0e/torch_random_walk/raw/0.1.9/dist/torch_rw-0.1.9-cp37-cp37m-linux_x86_64.whl \
    && pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.9.0+cu111.html \
    && pip install pytorch_tabnet==3.1.1

# Set the default command to python3
CMD ["bash"]
