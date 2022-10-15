mkdir frames

for i in *.wmv *.mp4 
do 
	mkdir frames/${i::-4}
	duration=$(ffprobe -v quiet -of csv=p=0 -show_entries format=duration $i)
	timestamp=0
	dur=$(echo "scale=0; $duration / 1"| bc)
	
	while [ $timestamp -le $dur ]
	do
		frame=$(echo "scale=6; $timestamp"| bc)
		echo "Getting frame at $frame seconds"
	
		ffmpeg -i $i -ss $frame -f image2 -vframes 1 "frames/"${i::-4}"/"${i::-4}"_"${timestamp}".png"
	  	timestamp=$(( $timestamp + 1 ))
	done
done
