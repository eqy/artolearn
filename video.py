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
    BASEHEIGHT = 128
    factor = int(BASEHEIGHT/img.shape[0])
    width = int(factor * img.shape[1])
    img = Image.fromarray(img)
    img = img.resize((width, BASEHEIGHT), Image.ANTIALIAS)
    img = np.array(img)
    res = cv.threshold(cv.cvtColor(img, cv.COLOR_BGR2GRAY), boundary, 255, cv.THRESH_BINARY_INV)[1]
    return res

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

def plausible_mmr(player_mmr, player_rank):
    RANK_RANGES = {'A': (2050, 2300), 'B': (1700, 2100), 'S': (2200, 3000)}
    if player_mmr is None:
        return False
    if player_rank is None: # LOL
        return player_mmr > 1600 and player_mmr < 3001
    return player_mmr >= RANK_RANGES[player_rank][0] and player_mmr <= RANK_RANGES[player_rank][1]

class Canonicalizer(object):
    def __init__(self, patterns):
        self.patterns = patterns
        self.values = set()
        for key in self.patterns:
            self.values.add(self.patterns[key])

    def canonicalize(self, string):
        lower = string.lower()
        for key in self.patterns:
            if key in lower:
                return self.patterns[key]
        return string

    def matched(self, string):
        return string in self.values
        

NAME_PATTERNS = {'artosis': 'artosis',
                 'arto': 'artosis',
                 'valks': 'artosis',
                 'canadadry': 'artosis',
                 'artasis': 'artosis'}

MAP_PATTERNS = {'polypoid': 'polypoid',
                'potypoid': 'polypoid',
                'poly': 'polypoid',

                'eclipse': 'eclipse',
                'clipse': 'eclipse',
                'good night': 'goodnight',
                'good': 'goodnight',
                'largo': 'largo',
                'larg': 'largo'}

name_canonicalizer = Canonicalizer(NAME_PATTERNS)
map_canonicalizer = Canonicalizer(MAP_PATTERNS)

def grab_matchdata2(frame, player_a_race, player_b_race, debug=False, online_debug=True):
    global DEBUG_ITER
    
    def grab_mapdata(map_name_crop):
        MAP_THRESHOLD = 48
        VIBRANCE_LOW = 0
        VIBRANCE_HIGH = 50
        SATURATION_LOW = 70
        SATURATION_HIGH = 160
        BACKGROUND_COLOR_HUE_LOW = 45
        BACKGROUND_COLOR_HUE_HIGH = 70
        global DEBUG_ITER
        mask = cv.inRange(cv.cvtColor(map_name_crop.copy(), cv.COLOR_BGR2HSV),
                          (BACKGROUND_COLOR_HUE_LOW, SATURATION_LOW, VIBRANCE_LOW),
                          (BACKGROUND_COLOR_HUE_HIGH, SATURATION_HIGH, VIBRANCE_HIGH))
        mask = cv.bitwise_not(mask)
        mask = mask.reshape(mask.shape[0], mask.shape[1], 1)
        map_name_crop = map_name_crop * (mask == 255)
        map_name_threshold = threshold(map_name_crop, MAP_THRESHOLD)
        map_text = pytesseract.image_to_string(map_name_threshold, config='--psm 7').strip()
        map_text = map_canonicalizer.canonicalize(map_text)
        if online_debug and not map_canonicalizer.matched(map_text):
            cv.imwrite(f'map_unmatched{DEBUG_ITER}.png', frame)
            DEBUG_ITER += 1
        if debug:
            cv.imwrite(f'2{DEBUG_ITER}mask.png', mask)
            cv.imwrite(f'2{DEBUG_ITER}mapcrop.png', map_name_crop)
            cv.imwrite(f'2{DEBUG_ITER}map.png', map_name_threshold)
        return map_text

    def grab_playerdata(player_name_crop, player_mmr_crop, player_race):
        NAME_THRESHOLD = 128
        MMR_THRESHOLD = 50
        global DEBUG_ITER
        player_name_threshold = threshold(player_name_crop, NAME_THRESHOLD)
        player_mmr_threshold = threshold(player_mmr_crop, MMR_THRESHOLD)
        player_name_text = pytesseract.image_to_string(player_name_threshold, config='--psm 7')
        player_mmr_text = pytesseract.image_to_string(player_mmr_threshold, config='--psm 7 -c tessedit_char_whitelist=0123456789')
        player_name = name_canonicalizer.canonicalize(player_name_text.strip())
        player_rank = grab_likely_rank(player_mmr_crop)
        player_mmr = re.sub('[^0-9]','', player_mmr_text)
        player_mmr = int(player_mmr) if len(player_mmr) else None
        player_mmr = player_mmr if plausible_mmr(player_mmr, player_rank) else None
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

    map_name_crop = crop(frame, *MAP_BBOX)
    map_text = grab_mapdata(map_name_crop)
    player_a_data = grab_playerdata(crop(frame, *PLAYER_A_NAME_BBOX),
                                    crop(frame, *PLAYER_A_MMR_BBOX),
                                    player_a_race)
    player_b_data = grab_playerdata(crop(frame, *PLAYER_B_NAME_BBOX),
                                    crop(frame, *PLAYER_B_MMR_BBOX),
                                    player_b_race)
    return map_text, player_a_data, player_b_data

def grab_pointsdata(frame):
    POINTS_BBOX_HEIGHT = 57
    POINTS_BBOX_WIDTH = 317
    POINTS_BBOX = (44/1080, 799/1920, (44+POINTS_BBOX_HEIGHT)/1080, (799+POINTS_BBOX_WIDTH)/1920)
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
    POI_FRAMESKIP = 1 # frameskip for points of interest
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
        def aggregate(values, canonical=None):
            def score(item):
                # None values have lowest priority
                if item is None:
                    return -1
                # Non-canonical values have max priority 1
                elif canonical is not None and not canonical(item):
                    return values.count(item)/len(values)
                # Canonical values have priority equal to count
                return values.count(item)
            temp = max(set(values), key=score)
            return temp
            
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
        map_results = [match_result[0] for match_result in match_results]
        player_results = [(match_result[1], match_result[2]) for match_result in match_results]
        player_a_results, player_b_results = zip(*player_results)
        print(len(match_results), len(self.matchframes))
        map_result = aggregate(map_results, map_canonicalizer.matched)
        player_a_names, player_a_ranks, player_a_mmrs, player_a_races = zip(*player_a_results)
        player_b_names, player_b_ranks, player_b_mmrs, player_b_races = zip(*player_b_results)
        player_a_results = (player_a_names, player_a_ranks, player_a_mmrs, player_a_races)
        player_b_results = (player_b_names, player_b_ranks, player_b_mmrs, player_b_races)
        player_a_result = tuple(aggregate(result, name_canonicalizer.matched) for result in player_a_results)
        player_b_result = tuple(aggregate(result) for result in player_b_results)
        player_a_artosis = name_canonicalizer.matched(player_a_result[0])
        player_b_artosis = name_canonicalizer.matched(player_b_result[0])
        turnrate_results = [grab_turnrate(game_frame) for game_frame in self.gameframes]
        trs, lats = zip(*turnrate_results)
        tr_result = aggregate(trs)
        lat_result = aggregate(lats)
        outcome_result = aggregate(outcome_results)
        print(tr_result, lat_result)


        if not(player_a_artosis) and not(player_b_artosis):
            print(f"WARNING: unable to find artosis in: {player_a_result[0]}, {player_b_result[0]}, skipping...")
        elif player_a_artosis and player_b_artosis:
            print("WARNING: double artosis, skipping...")
        else:
            if not map_canonicalizer.matched(map_result):
                print(f"WARNING: unable to match map: {map_results}")
            if player_a_artosis:
                player_results = player_a_result + player_b_result
            else:
                player_results = player_b_result + player_a_result
            if player_results[3] != 'T':
                print("WARNING: artosis was not terran...")
            game_data = player_results + (map_result, tr_result, lat_result, self.last_match_time/1000, outcome_result)
            print(game_data) 
        self.cleargame()

    def step(self, capture):
        def maybe_append_maybe_evict(l, item):
            if len(l) > self.MAX_ANALYSIS_FRAMES:
                if random.random() > 0.5:
                   random.shuffle(l)
                   l[-1] = item
            else:
                l.append(item) 
        time = capture.get(cv.CAP_PROP_POS_MSEC)
        if self.poi and (time - self.poi < self.POI_INTERVAL) and self.framecounter % self.POI_FRAMESKIP == 0:
            ret, frame = capture.read()
            assert ret
        elif self.framecounter % self.FRAMESKIP != 0: 
            capture.grab()
            self.framecounter += 1
            return
        else:
            ret, frame = capture.read()
            assert ret

        frametype, sim = self.reference_frames.match(frame)
        if sim < self.UNKNOWN_THRESHOLD:
            self.unknown_count += 1
        elif 'match' in frametype:
            if not self.poi or (time - self.poi > self.POI_INTERVAL):
                print(frametype, time)
            self.poi = time
            if self.last_match_time is None:
                self.last_match_time = time
            elif (time - self.last_match_time) > self.MATCH_TIMEOUT:
                self.savegame()
                self.matchframes = [(frametype, frame)]
                self.last_match_time = time
            else:
                maybe_append_maybe_evict(self.matchframes, (frametype, frame))
        elif 'postgame' in frametype:
            self.poi = time
            if self.last_match_time is None:
                print("WARNING: postgame without match, skipping...", time)
            else:
                maybe_append_maybe_evict(self.postgameframes, frame)
        elif 'points' in frametype:
            self.poi = time
            if self.last_match_time is None:
                print("WARNING: points without match, skipping...", time)
            else:
                maybe_append_maybe_evict(self.pointsframes, frame)
        elif 'game' in frametype:
            self.gameframe = True
            if self.framecounter % self.TURNRATE_FRAMESKIP == 0:
                maybe_append_maybe_evict(self.gameframes, frame)

        self.framecounter += 1

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
            self.video_parser.step(self.cap)
            frame_num += 1

def main():
    print("testing grab_matchdata")
    matchdata_frame = cv.imread('scene_reference/match_tvp/1.png')
    print(grab_matchdata2(matchdata_frame, 'T', 'P', debug=True))
    matchdata_frame = cv.imread('scene_reference/match_tvp/2.png')
    print(grab_matchdata2(matchdata_frame, 'T', 'P', debug=True))
    matchdata_frame = cv.imread('scene_reference/match_tvp/3.png')
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
