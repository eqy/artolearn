import argparse
import csv

from video import ReferenceFrames, VideoParser, Video

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='input file', required=True)
    parser.add_argument('-o', '--output', help='output file', required=True)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    reference_frames = ReferenceFrames('scene_reference')
    video_parser = VideoParser(reference_frames, debug=args.debug)
    input_video = Video(args.input, video_parser)
    input_video.parse()
    results = input_video.report()
    with open(args.output, 'w') as f:
        csv_writer = csv.writer(f)
        for result in results:
            csv_writer.writerow(result)

if __name__ == '__main__':
    main()
