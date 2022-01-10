import cv2 as cv
import os

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

CROP_START = (0, 0) #y, x
CROP_REL = (1080/1080, 1520/1920) #y, x
SSIM_RESOLUTION = (32, 32) #y, x

def cropframe(frame, height, width):
    return frame[CROP_START[0]:int(CROP_START[0]+CROP_REL[0]*height),
                 CROP_START[1]:int(CROP_START[1]+CROP_REL[1]*width)]

class ReferenceFrames(object):
    def __init__(self, filepath):
        self.filepath = filepath
        self.frametypes = dict()
        for dirpath, _, filenames in os.walk(filepath):
            for filename in filenames:
                ext = os.path.splitext(filename)[1]
                if ext == '.jpg' or '.png':
                    frame = cv.imread(os.path.join(dirpath, filename), cv.IMREAD_GRAYSCALE)
                    crop = cropframe(frame, frame.shape[0], frame.shape[1])
                    small = cv.resize(crop, SSIM_RESOLUTION)
                    frametype = os.path.basename(dirpath)
                    if frametype in self.frametypes:
                        self.frametypes[frametype].append(small) 
                    else:
                        self.frametypes[frametype] = [small]

    def match(self, frame):
        crop = cropframe(frame, frame.shape[0], frame.shape[1])
        gray = cv.cvtColor(crop, cv.COLOR_BGR2GRAY)
        small = cv.resize(gray, SSIM_RESOLUTION)
        max_scores = list()
        for frametype in self.frametypes.keys():
            maxscore = 0
            for reference_frame in self.frametypes[frametype]:
                score = ssim(small, reference_frame)
                if score > maxscore:
                    maxscore = score
            max_scores.append((frametype, maxscore))
        return sorted(max_scores, key=lambda item: item[1], reverse=True)

class Video(object):
    def __init__(self, filepath, reference_frames):
        self.filepath = filepath
        self.reference_frames = reference_frames
        self.cap = cv.VideoCapture(self.filepath)
        self.frame_count = int(self.cap.get(cv.CAP_PROP_FRAME_COUNT))
        self.height = int(self.cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        self.width = int(self.cap.get(cv.CAP_PROP_FRAME_WIDTH))
        print(f"opened {self.filepath} with {self.frame_count} frames height {self.height} width {self.width}")

    def parse(self):
        frame_num = 0
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                match_result = self.reference_frames.match(frame)
            else:
                print("warning, skipping frame...")
            if frame_num % 300 == 0:
                msecs = self.cap.get(cv.CAP_PROP_POS_MSEC)
                secs = msecs / 1000
                hrs = int(secs // 3600)
                secs -= hrs * 3600
                mins = int(secs // 60)
                secs-= mins * 60
                print(f"{hrs}:{mins}:{secs}")
                print(match_result)
            frame_num += 1

def main():
    print("testing video parsing...")
    reference_frames = ReferenceFrames('scene_reference')
    test_video = Video('test2.mp4', reference_frames)
    test_video.parse()
    print("OK")

if __name__ == '__main__':
    main()
