mkdir resized_videos

for i in *.wmv
do 
	width=640
	height=480
	ffprobe -v error -select_streams v:0 -show_entries stream=$width,$height -of csv=s=x:p=0 $i
done

for i in *.mp4
do 
	width=640
	height=480
	ffprobe -v error -select_streams v:0 -show_entries stream=$width,$height -of csv=s=x:p=0 $i
done
