#!/bin/bash

CONTAINER="issm_container"
echo $CONTAINER

# Stops the container in case it's open
docker stop $CONTAINER
docker rm $CONTAINER

# Launch a caontianer with a spoofed MAC for Matlab licensing
docker run --mac-address 02:42:ac:11:00:02 --name $CONTAINER -t -d ubuntu/issm

# Copy run script and model file
docker cp do_run.m $CONTAINER:/home/issm/do_run.m
docker cp model.mat $CONTAINER:/home/issm/model.mat

# Execute the script
docker exec -i $CONTAINER /usr/local/MATLAB/R2020a/bin/matlab -nojvm -nodesktop -batch "run do_run"
docker exec -i $CONTAINER ls
# Copy the results back
docker cp $CONTAINER:/home/issm/model1.mat .

docker stop $CONTAINER
docker rm $CONTAINER
