# Segment 1: frames 0-99 (high quality, copy)
ffmpeg -i input.mp4 -vf "select=between(n\,0\,50),setpts=PTS-STARTPTS" -c copy -avoid_negative_ts make_zero_duration part1.mp4

# Segment 2: frames 100-200 (low quality: scale to 256x144 ~144p)
ffmpeg -i input.mp4 -vf "select=between(n\,51\,100),scale=256:144:force_original_aspect_ratio=decrease,pad=1920:1080:-1:-1,setsar=1,setpts=PTS-STARTPTS+(100/30)*TB" -c:v libx264 -crf 23 -c:a copy part2.mp4

# Segment 3: frames 201-end (high quality, copy)
ffmpeg -i input.mp4 -vf "select=gte(n\,150),setpts=PTS-STARTPTS+(101/30)*TB" -c copy -avoid_negative_ts make_zero_duration part3.mp4

# Concat list file (file 'list.txt')
echo "file 'part1.mp4'" > list.txt
echo "file 'part2.mp4'" >> list.txt
echo "file 'part3.mp4'" >> list.txt

# Concatenate
ffmpeg -f concat -safe 0 -i list.txt -c copy output.mp4
