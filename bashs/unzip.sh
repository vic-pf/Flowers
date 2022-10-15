#!/bin/bash

for i in *.zip 
do 
	echo "Unziping file $i"
	unzip -d ../Unicamp/PFG/code/videos/ "${i}"
done
