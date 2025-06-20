# face sample rate is 1 frame every 3 seconds

video="terry-flash.mp4"
if [ $# -gt 0 ]; then
    video=$1
fi
prefix="${video}_%03d.jpg"

rm -f "${video}*jpg"
#time ffmpeg -i $video -r 0.1 -qscale:v 5 -vf mpdecimate=hi=8000:lo=8000:frac=1 $prefix
time ffmpeg -i $video -r 0.333 -qscale:v 5 $prefix

