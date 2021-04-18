#! /bin/bash

# stops godot servers
proc_dir=/home/skugele/Development/simple-animat-world/scripts/daemontools

shopt -s nullglob

cd $proc_dir

for instance in */; do
        svstat $instance
done
