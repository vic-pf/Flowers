mkdir frames

for i in *.wmv *.mp4 
do 
	duration=$(ffprobe -v quiet -of csv=p=0 -show_entries format=duration $i)

	frame=$(echo "scale=6; $duration / 2"| bc)
	echo "Midle frame is at $frame seconds"
	
	ffmpeg -i $i -ss $frame -f image2 -vframes 1 frames/${i::-4}".jpg"
done
