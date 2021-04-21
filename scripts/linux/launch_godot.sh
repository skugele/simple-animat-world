#! /bin/bash

# launches godot servers using supervise (see daemontools).
proc_dir=/home/skugele/Development/simple-animat-world/scripts/linux/daemontools

shopt -s nullglob

cd $proc_dir
for instance in */; do
	svok $instance
	if [ $? -eq 0 ]
	then
		# bounce running instance (if needed)
		svc -tu $instance
	else
		nohup supervise $instance 1>${instance}/supervise.log 2>&1 &
	fi
done
