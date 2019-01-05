from moviepy.video.VideoClip import *
from moviepy.editor import *
t1 = '00:00:01'
t2 = '00:02:13'
clip = VideoFileClip('in.mp4')
new_clip = clip.subclip(t1, t2)
new_clip.write_videofile('out.mp4')
