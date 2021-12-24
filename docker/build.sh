export DOCKER_BUILDKIT=0
cp ../pyproject.toml pyproject.toml
docker build -t sachinx0e/ikg:4.8 .
