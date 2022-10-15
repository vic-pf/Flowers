#!/bin/bash
mkdir frames

for i in *.wmv *.mp4 
do 
	mkdir frames/${i::-4}
	duration=$(ffprobe -v quiet -of csv=p=0 -show_entries format=duration $i)
	timestamp=1
	dur=$(echo "scale=0; $duration / 1"| bc)
	if [[ $dur -le 10 ]]
	then
		inc=2
 	elif [[ $dur -gt 10 ]] && [[ $dur -le 25 ]]
	then
		inc=4
	elif [[ $dur -gt 25 ]] && [[ $dur -le 35 ]]
	then
		inc=6
	elif [[ $dur -gt 35 ]] && [[ $dur -le 60 ]]
	then
		inc=$(echo "scale=0; $dur / 15" | bc)
	elif [[ $dur -gt 60 ]] && [[ $dur -le 120 ]]
	then
		inc=$(echo "scale=0; $dur / 30" | bc)
 	else
  		inc=$(echo "scale=0; $dur / 45" | bc)
	fi
	
	while [ $timestamp -le $dur ]
	do
		frame=$(echo "scale=6; $timestamp"| bc)
		echo "Getting frame at $frame seconds"
		ffmpeg -i $i -ss $frame -f image2 -vframes 1 "frames/"${i::-4}"/"${i::-4}"_"${timestamp}".png"
		timestamp=$(( $timestamp + $inc ))
	done
done
