from os import path, walk, makedirs, rename
from time import time
from imagehash import average_hash
from PIL import Image
from cv2 import VideoCapture, CAP_PROP_FRAME_COUNT, CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_HEIGHT, CAP_PROP_FPS
from json import dump, load
from multiprocessing import Pool, cpu_count
import sys

input_vid_dir = 'clipped/'
json_dir = 'db/'
analyzed_dir = 'analyzed/'
duplicate_dir = 'duplicate/'

if not path.exists(json_dir):
    makedirs(json_dir)

if not path.exists(analyzed_dir):
    makedirs(analyzed_dir)

if not path.exists(duplicate_dir):
    makedirs(duplicate_dir)


def write_to_json(filename, data):
    file_full_path = json_dir + filename + ".json"
    with open(file_full_path, 'w') as file_pointer:
        dump(data, file_pointer)
    return


def video_to_json(filename):
    file_json_path = json_dir + filename + ".json"
    if not path.exists(file_json_path):
        file_full_path = input_vid_dir + filename
        start = time()
        size = round(path.getsize(file_full_path) / 1024 / 1024, 2)
        video_pointer = VideoCapture(file_full_path)
        frame_count = int(VideoCapture.get(
            video_pointer, int(CAP_PROP_FRAME_COUNT)))
        width = int(VideoCapture.get(video_pointer, int(CAP_PROP_FRAME_WIDTH)))
        height = int(VideoCapture.get(
            video_pointer, int(CAP_PROP_FRAME_HEIGHT)))
        fps = int(VideoCapture.get(video_pointer, int(CAP_PROP_FPS)))
        success, image = video_pointer.read()
        video_hash = {}
        while success:
            frame_hash = average_hash(Image.fromarray(image))
            video_hash[str(frame_hash)] = filename
            success, image = video_pointer.read()
        stop = time()
        time_taken = stop - start
        print("Time taken for ", file_full_path, " is : ", time_taken)
        data_dict = dict()
        data_dict['size'] = size
        data_dict['time_taken'] = time_taken
        data_dict['fps'] = fps
        data_dict['height'] = height
        data_dict['width'] = width
        data_dict['frame_count'] = frame_count
        data_dict['filename'] = filename
        data_dict['video_hash'] = video_hash
        write_to_json(filename, data_dict)
        return


def multiprocess_video_to_json():
    files = next(walk(input_vid_dir))[2]
    processes = cpu_count()
    print(processes)
    pool = Pool(processes)
    start = time()
    pool.starmap_async(video_to_json, zip(files))
    pool.close()
    pool.join()
    stop = time()
    print("Time Taken : ", stop - start)


def key_with_max_val(d):
    max_value = 0
    required_key = ""
    for k in d:
        if d[k] > max_value:
            max_value = d[k]
            required_key = k
    return required_key


def duplicate_analyzer(rate):
    files = next(walk(json_dir))[2]
    data_dict = {}
    for file in files:
        if (path.isfile(analyzed_dir+file[:-5])):
            filename = json_dir + file
            with open(filename) as f:
                data = load(f)
            video_hash = data['video_hash']
            count = 0
            duplicate_file_dict = dict()
            for key in video_hash:
                count += 1
                if key in data_dict:
                    if data_dict[key] in duplicate_file_dict:
                        duplicate_file_dict[data_dict[key]
                                            ] = duplicate_file_dict[data_dict[key]] + 1
                    else:
                        duplicate_file_dict[data_dict[key]] = 1
                else:
                    data_dict[key] = video_hash[key]
            if duplicate_file_dict:
                duplicate_file = key_with_max_val(duplicate_file_dict)
                if (path.isfile(analyzed_dir+duplicate_file)):
                    duplicate_percentage = (
                        (duplicate_file_dict[duplicate_file] / count) * 100)
                    if duplicate_percentage > rate:
                        file = file[:-5]
                        origin_file = duplicate_file[:-4]
                        src = analyzed_dir + file
                        tgt = duplicate_dir + origin_file + '/'
                        if not path.isdir(duplicate_dir+file[:-4]):
                            print(file, " is dup of ", duplicate_file)
                            if not path.exists(tgt):
                                makedirs(tgt)
                            tgt += file
                            if path.exists(src):
                                rename(src, tgt)


def mv_analyzed_file():
    files = next(walk(json_dir))[2]
    for filename in files:
        filename = filename[:-5]
        src = input_vid_dir + filename
        tgt = analyzed_dir + filename
        if path.exists(src):
            rename(src, tgt)


if __name__ == '__main__':
    mv_analyzed_file()
    multiprocess_video_to_json()
    mv_analyzed_file()
    if(sys.argv[1] == '-d'):
        duplicate_analyzer(50)
        duplicate_analyzer(25)
        duplicate_analyzer(13)
    elif(sys.argv[1] == '-a'):
        duplicate_analyzer(40)
        duplicate_analyzer(20)
        duplicate_analyzer(10)
    elif(sys.argv[1] == '-t'):
        duplicate_analyzer(60)
        duplicate_analyzer(30)
        duplicate_analyzer(25)
        duplicate_analyzer(20)
    else:
        duplicate_analyzer(int(sys.argv[1]))
