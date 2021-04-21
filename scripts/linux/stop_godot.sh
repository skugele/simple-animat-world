#! /bin/bash

# stops godot servers
proc_dir=/home/skugele/Development/simple-animat-world/scripts/linux/daemontools

shopt -s nullglob

cd $proc_dir

for instance in */; do
        svc -d $instance
done
