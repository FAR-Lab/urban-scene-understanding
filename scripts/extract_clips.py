from moviepy.editor import VideoFileClip
import moviepy.video.io.ffmpeg as ffmpeg

import os 
import fire 

def extract_clips(video_path, periods, output_dir, FRAME_RATE=60):
    """Extracts clips from a video based on a list of periods.
    
    Args:
        video_path (str): Path to the video to extract clips from.
        periods (list): List of tuples containing the start and end times of each clip.
        output_dir (str): Directory to save the clips to.
    
    Returns:
        None
    """
    for codec in ffmpeg.list_codecs():
      print(codec)
    
    
    

    for i, (start_frame, end_frame) in enumerate(periods):
        start_time = start_frame / FRAME_RATE 
        end_time = end_frame / FRAME_RATE
        output_path = f"{output_dir}_intersection_{start_time:.2f}_{end_time:.2f}_{i+1}.mp4"
        video = VideoFileClip(video_path).subclip(start_time, end_time)
        video.write_videofile(output_path, codec='libx264')

if __name__ == '__main__':
    fire.Fire(extract_clips)
