# duplicate frame filter is overly aggressive for testing purposes
# sample rate is 1 frame every 20 seconds for testing only
# production should use sample rate of at least 1 fps and default dup filter

video="terry.mp4"
if [ $# -gt 0 ]; then
    video=$1
fi
prefix="${video}_%02d.png"

rm -f "${video}*png"
time ffmpeg -i $video -r 0.05 -vf mpdecimate=hi=8000:lo=8000:frac=1 $prefix

