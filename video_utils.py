import ffmpeg
import numpy as np

class VideoUtils:
    def __init__(self, video_path):
        self.video_path = video_path
        self.width = None
        self.height = None
        self.total_frames = None
        self.fps = None
        self.process = None
        self.frame_size = None
        
        self._probe_video()
    
    def _probe_video(self):
        """Probe video to get metadata"""
        probe = ffmpeg.probe(self.video_path)
        video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
        
        self.width = int(video_info['width'])
        self.height = int(video_info['height'])
        self.total_frames = int(video_info.get('nb_frames', 0))
        self.fps = eval(video_info['r_frame_rate'])  # Convert "30/1" to 30.0
        self.frame_size = self.width * self.height
    
    def start(self, grayscale=True):
        """Start FFmpeg process to read frames"""
        pix_fmt = 'gray' if grayscale else 'rgb24'
        
        self.process = (
            ffmpeg
            .input(self.video_path)
            .output('pipe:', format='rawvideo', pix_fmt=pix_fmt)
            .run_async(pipe_stdout=True, pipe_stderr=True)
        )
        
        return self
    
    def readFrame(self, grayscale=True):
        """Read next frame, returns None when video ends"""
        if self.process is None:
            raise RuntimeError("Call start() before reading frames")
        
        frame_size = self.frame_size if grayscale else self.frame_size * 3
        in_bytes = self.process.stdout.read(frame_size)
        
        if not in_bytes:
            return None
        
        if grayscale:
            frame = np.frombuffer(in_bytes, np.uint8).reshape([self.height, self.width])
        else:
            frame = np.frombuffer(in_bytes, np.uint8).reshape([self.height, self.width, 3])
        
        return frame
    
    def __iter__(self):
        """Iterator interface for easy looping"""
        while True:
            frame = self.readFrame()
            if frame is None:
                break
            yield frame
    
    def release(self):
        """Clean up FFmpeg process"""
        if self.process:
            self.process.stdout.close()
            self.process.wait()
            self.process = None
    
    def __enter__(self):
        """Context manager support"""
        return self.start()
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager cleanup"""
        self.release()
