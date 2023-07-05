import subprocess

def strip_audio(video_path): 
    ffmpeg_cmd = ['ffmpeg', '-i', video_path, '-c', 'copy', '-an', f'{video_path[:-4]}_noaud.MP4']
    # Execute the FFmpeg command
    try: 
        subprocess.run(ffmpeg_cmd, capture_output=True)
    except subprocess.CalledProcessERror as e: 
        print('error code', e.returncode, e.output)

