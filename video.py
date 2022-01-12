import csv
import re
import random
import os

from PIL import Image
import cv2 as cv
import numpy as np
import pytesseract

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

DEBUG_ITER = 0

def crop(img, ry0, rx0, ry1, rx1):
    height = img.shape[0]
    width = img.shape[1]
    return img[int(ry0*height):int(ry1*height), int(rx0*width):int(rx1*width)]

def cropframe(frame, height, width):
    CROP_START = (0, 0) #ry1, rx1
    CROP_REL = (1080/1080, 1520/1920) #ry1, rx1
    return crop(frame, 0, 0, CROP_REL[0], CROP_REL[1])

def threshold(img, boundary=56):
    res = cv.threshold(cv.cvtColor(img, cv.COLOR_BGR2GRAY), boundary, 255, cv.THRESH_BINARY_INV)[1]
    return res

# currently unused
def numeric_substitutions(s):
    s = s.lower()
    s.replace('s', '5')
    s.replace('b', '8')
    s.replace('z', '2')
    s.replace('i', '1')
    s.replace('a', '4')
    s.replace('o', '0')
    return s

# currently unused
def parse_matchtext(player_text):
    # heuristic value to rank the plausibility of OCR'd MMR values, currently unused
    STANDARD_MMR = 2000
    name = None
    mmr = None
    possible_id = False
    for line in player_text:
        if line != '':
            if name is None:
                name = line
            else:
                line = numeric_substitutions(line)
                curr_mmr = re.sub('[^0-9]','', line)
                if curr_mmr != '':
                    curr_mmr = int(curr_mmr)
                    if mmr is None:
                        mmr = curr_mmr
                    elif abs(curr_mmr - STANDARD_MMR) < abs(mmr - STANDARD_MMR):
                        mmr = curr_mmr
    return name, mmr

def grab_likely_rank(player_crop):
    #A: red, B: purple, S: yellow
    RANGES = {'A': (0, 15), 'B': (140, 160), 'S': (25, 35)}
    # TODO: these can likely be further tuned
    VIBRANCE_LOW = 100
    VIBRANCE_HIGH = 200
    SATURATION_LOW = 100
    SATURATION_HIGH = 200
    player_hsv = cv.cvtColor(player_crop.copy(), cv.COLOR_BGR2HSV)
    rank_to_masks = [(rank, cv.inRange(player_hsv, (RANGES[rank][0], SATURATION_LOW, VIBRANCE_LOW),
                                       (RANGES[rank][1], SATURATION_HIGH, VIBRANCE_HIGH))) for rank in RANGES.keys()] 
    rank_to_sum = [(rank, np.sum(cv.threshold(mask, 0, 1, cv.THRESH_OTSU)[1])) for rank, mask in rank_to_masks]
    sorted_ranks = sorted(rank_to_sum, key=lambda pair:pair[1], reverse=True)
    if sorted_ranks[0][1] > 0:
        return sorted_ranks[0][0]
    # couldn't match any colors
    return None

# TODO: binary valid/invalid instead of guessing
def guess_mmr(player_mmr, player_rank):
    RANK_MSDS = {'A': 2000, 'B': 1000, 'S': 2000}
    RANK_MSD2S = {'A': 2100, 'B': 1900, 'S': 2300}
    # sanity check MMRs
    RANK_MAXES = {'A': 2350, 'B': 2200, 'S': 9999}
    RANK_MINS = {'A': 1900, 'B': 1650, 'S': 2150}
    if player_mmr is not None:
        if player_rank in RANK_MSD2S and player_mmr < 100:
            player_mmr += RANK_MSD2S[player_rank]
        elif player_rank in RANK_MSDS and player_mmr < 1000:
            player_mmr += RANK_MSDS[player_rank]
        elif player_rank in RANK_MSDS and player_mmr > RANK_MAXES[player_rank]:
            player_mmr = (player_mmr % 1000) + RANK_MSDS[player_rank]
        if player_rank in RANK_MAXES:
            if player_mmr > RANK_MAXES[player_rank]:
                player_mmr = None
            elif player_mmr < RANK_MINS[player_rank]:
                player_mmr = None
    return player_mmr

def grab_matchdata2(frame, player_a_race, player_b_race, debug=False):
    global DEBUG_ITER

    def grab_playerdata(player_name_crop, player_mmr_crop, player_race):
        NAME_THRESHOLD = 224
        MMR_THRESHOLD = 72
        global DEBUG_ITER
        player_name_threshold = threshold(player_name_crop, NAME_THRESHOLD)
        player_mmr_threshold = threshold(player_mmr_crop, MMR_THRESHOLD)
        player_name_text = pytesseract.image_to_string(player_name_threshold, config='--psm 7')
        player_mmr_text = pytesseract.image_to_string(player_mmr_threshold, config='--psm 7 -c tessedit_char_whitelist=0123456789')
        player_name = player_name_text.strip()
        player_rank = grab_likely_rank(player_mmr_crop)
        player_mmr = re.sub('[^0-9]','', player_mmr_text)
        player_mmr = int(player_mmr) if len(player_mmr) else None

        if debug:
            cv.imwrite(f'2{DEBUG_ITER}aname.png', player_name_threshold)
            cv.imwrite(f'2{DEBUG_ITER}bmmr.png', player_mmr_threshold)
            print(player_name_text.strip(), player_mmr_text.strip())
            DEBUG_ITER += 1

        player_data = (player_name, player_rank, player_mmr, player_race)
        return player_data
    
    NAME_BBOX_HEIGHT = 30
    NAME_BBOX_WIDTH = 232
    MMR_BBOX_HEIGHT = 20
    MMR_BBOX_WIDTH = 50
    MAP_BBOX_HEIGHT = 24
    MAP_BBOX_WIDTH = 463
    MAP_THRESHOLD = 72
    PLAYER_A_NAME_BBOX = (763/1080, 408/1920, (763+NAME_BBOX_HEIGHT)/1080, (408+NAME_BBOX_WIDTH)/1920)
    PLAYER_B_NAME_BBOX = (763/1080, 1293/1920, (763+NAME_BBOX_HEIGHT)/1080, (1293+NAME_BBOX_WIDTH)/1920)
    PLAYER_A_MMR_BBOX = (793/1080, 575/1920, (793+MMR_BBOX_HEIGHT)/1080, (575+MMR_BBOX_WIDTH)/1920)
    PLAYER_B_MMR_BBOX = (793/1080, 1298/1920, (793+MMR_BBOX_HEIGHT)/1080, (1298+MMR_BBOX_WIDTH)/1920)
    MAP_BBOX = (214/1080, 729/1920, (214+MAP_BBOX_HEIGHT)/1080, (729+MAP_BBOX_WIDTH)/1920)

    NAMES = ['artosis',
             'newgear',
             'valks',
             'canadadry',
             'artasis'] # lol, OCR sucks

    height = frame.shape[0]
    width = frame.shape[1]

    map_crop = crop(frame, *MAP_BBOX)
    player_a_data = grab_playerdata(crop(frame, *PLAYER_A_NAME_BBOX),
                                    crop(frame, *PLAYER_A_MMR_BBOX),
                                    player_a_race)
    player_b_data = grab_playerdata(crop(frame, *PLAYER_B_NAME_BBOX),
                                    crop(frame, *PLAYER_B_MMR_BBOX),
                                    player_b_race)
    return player_a_data, player_b_data

def grab_matchdata(frame, player_a_race, player_b_race, debug=False):
    global DEBUG_ITER
    BBOX_HEIGHT = 56
    BBOX_WIDTH = 232
    PLAYER_A_BBOX = (764/1080, 408/1920, (764+BBOX_HEIGHT)/1080, (408+BBOX_WIDTH)/1920)
    PLAYER_B_BBOX = (764/1080, 1293/1920, (764+BBOX_HEIGHT)/1080, (1293+BBOX_WIDTH)/1920)

    NAMES = ['artosis', 'newgear', 'valks', 'canadadry']
    height = frame.shape[0]
    width = frame.shape[1]
    player_a_crop = crop(frame, *PLAYER_A_BBOX)
    player_b_crop = crop(frame, *PLAYER_B_BBOX)
    player_a_threshold = threshold(player_a_crop)
    player_b_threshold = threshold(player_b_crop)
    if debug:
        cv.imwrite(f'a{DEBUG_ITER}.png', player_a_threshold)    
        cv.imwrite(f'b{DEBUG_ITER}.png', player_b_threshold)
        DEBUG_ITER += 1
    player_a_text = pytesseract.image_to_string(player_a_threshold).splitlines()
    player_b_text = pytesseract.image_to_string(player_b_threshold).splitlines()
    # guess rank based on color
    player_a_rank = grab_likely_rank(player_a_crop)
    player_b_rank = grab_likely_rank(player_b_crop)
    player_a_name, player_a_mmr = parse_matchtext(player_a_text)
    player_b_name, player_b_mmr = parse_matchtext(player_b_text)
    player_a_mmr = guess_mmr(player_a_mmr, player_a_rank)
    player_b_mmr = guess_mmr(player_b_mmr, player_b_rank)

    player_a_data = (player_a_name, player_a_rank, player_a_mmr, player_a_race)
    player_b_data = (player_b_name, player_b_rank, player_b_mmr, player_b_race)
    return player_a_data, player_b_data

def grab_pointsdata(frame):
    POINTS_BBOX_HEIGHT = 57
    POINTS_BBOX_WIDTH = 317
    POINTS_BBOX = (44/1080, 799/1920, (44+POINTS_BBOX_HEIGHT)/1080, (799+POINTS_BBOX_WIDTH)/1920)
    height = frame.shape[0]
    width = frame.shape[1]
    points_crop = crop(frame, *POINTS_BBOX)
    points_threshold = threshold(points_crop, boundary=128)
    text = pytesseract.image_to_string(points_threshold, config='--psm 7')
    text = text.lower()
    if 'victory' in text:
        return 'victory'
    elif 'defeat' in text:
        return 'defeat'
    return None

def grab_postgamedata(frame):
    POSTGAME_BBOX_HEIGHT = 46
    POSTGAME_BBOX_WIDTH = 183
    POSTGAME_BBOX = (45/1080, 412/1920, (45+POSTGAME_BBOX_HEIGHT)/1080, (412+POSTGAME_BBOX_WIDTH)/1920)
    height = frame.shape[0]
    width = frame.shape[1]
    postgame_crop = crop(frame, *POSTGAME_BBOX)
    postgame_threshold = threshold(postgame_crop)
    text = pytesseract.image_to_string(postgame_threshold, config='--psm 7')
    text = text.lower()
    if 'victory' in text:
        return 'victory'
    elif 'pending' in text:
        return 'pending'
    elif 'defeat' in text:
        return 'defeat'
    return None

def grab_turnrate(frame, debug=False):
    TURNRATE_BBOX_HEIGHT = 25
    TURNRATE_BBOX_WIDTH = 131
    TURNRATE_BBOX = (23/1080, 20/1920, (23+TURNRATE_BBOX_HEIGHT)/1080, (20+TURNRATE_BBOX_WIDTH)/1920)
    THRESHOLD = 128
    height = frame.shape[0]
    width = frame.shape[1]
    turnrate_crop = crop(frame, *TURNRATE_BBOX)
    turnrate_threshold = threshold(turnrate_crop, THRESHOLD)

    text = pytesseract.image_to_string(turnrate_threshold, config='--psm 7')
    text = text.lower()
    if debug:
        global DEBUG_ITER
        cv.imwrite(f'turnrate{DEBUG_ITER}.png', turnrate_threshold)
        DEBUG_ITER += 1

    lat = None
    if 'low' in text:
        lat = 'low'
    elif 'high' in text:
        lat = 'high'
    elif 'extra' in text:
        lat = 'extra'
    tr = None
    num = re.sub('[^0-9]','', text)
    num = int(num) if len(num) else 0
    if num in (8, 12, 14, 16, 20, 24):
        tr = num
    return lat, tr

def frametypetoraces(frametype):
    if 'tvz' in frametype:
        return 'T', 'Z'
    elif 'zvt' in frametype:
        return 'Z', 'T'
    elif 'tvp' in frametype:
        return 'T', 'P'
    elif 'pvt' in frametype:
        return 'P', 'T'
    elif 'tvt' in frametype:
        return 'T', 'T'
    else:
        assert False, f"unsupported frametype ({frametype}) to race"

class VideoParser(object):
    FRAMESKIP = 30 # frameskip for generic parts
    POI_FRAMESKIP = 4 # frameskip for points of interest
    TURNRATE_FRAMESKIP = 1800 # frameskip for turnrate
    UNKNOWN_THRESHOLD = 0.2
    # heuristic value to separate match screens
    MATCH_TIMEOUT = 30000 # msec
    # time to dwell on a point of interest for frameskip
    POI_INTERVAL = 1000 #msec
    MAX_ANALYSIS_FRAMES = 100

    def __init__(self, reference_frames, debug_dump=True):
        self.reference_frames = reference_frames
        self.cleargame()
        self.games = list()
        self.date = None
        self.debug_dump = debug_dump
        self.framecounter = 0
        self.unknown_count = 0
        self.last_match_time = None
        # whether current frame is interesting
        self.poi = False

    def pruneframes(self):
        random.shuffle(self.matchframes)
        random.shuffle(self.postgameframes)
        random.shuffle(self.pointsframes)
        random.shuffle(self.gameframes)
        self.matchframes = self.matchframes[:self.MAX_ANALYSIS_FRAMES]
        self.postgameframes = self.postgameframes[:self.MAX_ANALYSIS_FRAMES]
        self.pointsframes = self.pointsframes[:self.MAX_ANALYSIS_FRAMES]
        self.gameframes =   self.gameframes[:self.MAX_ANALYSIS_FRAMES]

    def setdate(self, date):
        self.date = date

    def cleargame(self):
        self.matchframes = list()
        self.postgameframes = list()
        self.pointsframes = list()
        self.gameframes = list()
        self.last_match_time = None
        self.gameframe = False

    def savegame(self):
        if self.last_match_time is None:
            print("no game to save, done!")
            assert len(self.matchframes) == 0
            assert len(self.postgameframes) == 0
        else:
            if not self.gameframe:
                print("WARNING, trying to save game without any gameframes")
            assert len(self.postgameframes) > 0 or len(self.pointsframes) > 0
        print(f"saving postgameframes {len(self.postgameframes)} pointsframes {len(self.pointsframes)} {len(self.matchframes)} matchframes")
        self.pruneframes()
        print("grabbing postgame results...")
        postgame_results = [grab_postgamedata(postgame_frame) for postgame_frame in self.postgameframes]
        print("grabbing points results...")
        points_results = [grab_pointsdata(points_frame) for points_frame in self.pointsframes]
        outcome_results = postgame_results + points_results
        print("grabbing match...")
        match_results = [grab_matchdata2(match_frame[1], *(frametypetoraces(match_frame[0]))) for match_frame in self.matchframes]
        player_a_results, player_b_results = zip(*match_results)
        print(len(match_results), len(self.matchframes))
        print(player_a_results)
        print(player_b_results)
        turnrate_results = [grab_turnrate(game_frame) for game_frame in self.gameframes]
        trs, lats = zip(*turnrate_results)
        print(outcome_results)
        print(turnrate_results)
        self.cleargame()

    def step(self, frame, time):
        self.framecounter += 1
        if self.poi and (time - self.poi < self.POI_INTERVAL) and self.framecounter % self.POI_FRAMESKIP == 0:
            pass # continue processing
        elif self.framecounter % self.FRAMESKIP != 0:
            return
        frametype, sim = self.reference_frames.match(frame)
        if sim < self.UNKNOWN_THRESHOLD:
            self.unknown_count += 1
        elif 'match' in frametype:
            self.poi = time
            print(frametype, time)
            if self.last_match_time is None:
                self.last_match_time = time
            elif (time - self.last_match_time) > self.MATCH_TIMEOUT:
                self.savegame()
                self.matchframes = [(frametype, frame)]
                self.last_match_time = time
            else:
                self.matchframes.append((frametype, frame))
        elif 'postgame' in frametype:
            self.poi = time
            if self.last_match_time is None:
                print("WARNING: postgame without match, skipping...", time)
            else:
                self.postgameframes.append(frame)
        elif 'points' in frametype:
            self.poi = time
            if self.last_match_time is None:
                print("WARNING: points without match, skipping...", time)
            else:
                self.pointsframes.append(frame)
        elif 'game' in frametype:
            self.gameframe = True
            if self.framecounter % self.TURNRATE_FRAMESKIP == 0:
                self.gameframes.append(frame)

    def report(self):
        self.savegame()

class ReferenceFrames(object):
    SSIM_RESOLUTION = (32, 32) #y, x

    def __init__(self, filepath):
        self.filepath = filepath
        self.frametypes = dict()
        for dirpath, _, filenames in os.walk(filepath):
            for filename in filenames:
                ext = os.path.splitext(filename)[1]
                if ext == '.jpg' or ext == '.png':
                    frame = cv.imread(os.path.join(dirpath, filename), cv.IMREAD_GRAYSCALE)
                    crop = cropframe(frame, frame.shape[0], frame.shape[1])
                    small = cv.resize(crop, self.SSIM_RESOLUTION)
                    frametype = os.path.basename(dirpath)
                    if frametype in self.frametypes:
                        self.frametypes[frametype].append(small) 
                    else:
                        self.frametypes[frametype] = [small]

    def match(self, frame):
        crop = cropframe(frame, frame.shape[0], frame.shape[1])
        gray = cv.cvtColor(crop, cv.COLOR_BGR2GRAY)
        small = cv.resize(gray, self.SSIM_RESOLUTION)
        max_scores = list()
        for frametype in self.frametypes.keys():
            maxscore = -1
            for reference_frame in self.frametypes[frametype]:
                score = psnr(small, reference_frame)
                if score > maxscore:
                    maxscore = score
            max_scores.append((frametype, maxscore))
        return sorted(max_scores, key=lambda item: item[1], reverse=True)[0]

class Video(object):
    def __init__(self, filepath, video_parser):
        self.filepath = filepath
        self.video_parser = video_parser
        date = os.path.getmtime(filepath)
        self.video_parser.setdate(date)
        self.cap = cv.VideoCapture(self.filepath)
        self.frame_count = int(self.cap.get(cv.CAP_PROP_FRAME_COUNT))
        self.height = int(self.cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        self.width = int(self.cap.get(cv.CAP_PROP_FRAME_WIDTH))
        print(f"opened {self.filepath} with {self.frame_count} frames height {self.height} width {self.width}")

    def parse(self):
        frame_num = 0
        while self.cap.isOpened():
            if frame_num >= self.frame_count:
                break
            ret, frame = self.cap.read()
            if ret:
                self.video_parser.step(frame, self.cap.get(cv.CAP_PROP_POS_MSEC))
            else:
                print("warning, skipping frame...")
            frame_num += 1

def main():
    print("testing grab_matchdata")
    matchdata_frame = cv.imread('scene_reference/match_tvp/1.png')
    print(grab_matchdata2(matchdata_frame, 'T', 'P', debug=True))
    matchdata_frame = cv.imread('scene_reference/match_tvp/2.png')
    print(grab_matchdata2(matchdata_frame, 'T', 'P', debug=True))
    matchdata_frame = cv.imread('scene_reference/match_pvt/1.png')
    print(grab_matchdata2(matchdata_frame, 'P', 'T', debug=True))
    matchdata_frame = cv.imread('scene_reference/match_pvt/2.png')
    print(grab_matchdata2(matchdata_frame, 'P', 'T', debug=True))
    matchdata_frame = cv.imread('scene_reference/match_tvz/1.png')
    print(grab_matchdata2(matchdata_frame, 'T', 'Z', debug=True))
    matchdata_frame = cv.imread('scene_reference/match_tvz/2.png')
    print(grab_matchdata2(matchdata_frame, 'T', 'Z', debug=True))
    matchdata_frame = cv.imread('scene_reference/match_tvz/3.png')
    print(grab_matchdata2(matchdata_frame, 'T', 'Z', debug=True))
    matchdata_frame = cv.imread('scene_reference/match_zvt/1.png')
    print(grab_matchdata2(matchdata_frame, 'Z', 'T', debug=True))
    matchdata_frame = cv.imread('scene_reference/match_tvt/1.png')
    print(grab_matchdata2(matchdata_frame, 'T', 'T', debug=True))
    print("testing grab_pointsdata")
    pointsdata_frame = cv.imread('scene_reference/points_victory/1.png')
    print(grab_pointsdata(pointsdata_frame))
    pointsdata_frame = cv.imread('scene_reference/points_defeat/1.png')
    print(grab_pointsdata(pointsdata_frame))
    pointsdata_frame = cv.imread('scene_reference/points_pending/1.png')
    print(grab_pointsdata(pointsdata_frame))
    print("testing grab_postgamedata")
    postgamedata_frame = cv.imread('scene_reference/postgame_victory/1.png')
    print(grab_postgamedata(postgamedata_frame))
    postgamedata_frame = cv.imread('scene_reference/postgame_defeat/1.png')
    print(grab_postgamedata(postgamedata_frame))
    postgamedata_frame = cv.imread('scene_reference/postgame_pending/1.png')
    print(grab_postgamedata(postgamedata_frame))
    print("testing grab_turnrate")
    turnratedata_frame = cv.imread('scene_reference/game_terran/1.png')
    print(grab_turnrate(turnratedata_frame, debug=True))
    turnratedata_frame = cv.imread('scene_reference/game_terran/2.png')
    print(grab_turnrate(turnratedata_frame, debug=True))
    turnratedata_frame = cv.imread('scene_reference/game_terran/3.png')
    print(grab_turnrate(turnratedata_frame, debug=True))
    turnratedata_frame = cv.imread('scene_reference/game_terran/4.png')
    print(grab_turnrate(turnratedata_frame, debug=True))
    print("testing video parsing...")
    reference_frames = ReferenceFrames('scene_reference')
    video_parser = VideoParser(reference_frames)
    test_video = Video('test2.mp4', video_parser)
    test_video.parse()
    print("OK")

if __name__ == '__main__':
    main()
