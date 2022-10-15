import os
from typing import Dict
import cv2
import sys
import glob
import math
import datetime
import pandas as pd


def translate_code(code, sheet):
    for i in range(0, sheet.shape[0]):
        cod = sheet['Cod'][i]
        data_field = sheet['Date'][i]
        if code == int(cod):
            data = data_field.strftime('%d-%m-%Y')
            data_code = data.replace('-', '')[0:4]
            return data, data_code
    return None, None


def get_video_time_string(string):
    time = string[string.rfind("/")+1:-4]
    time.upper()
    if time.count('_') > 0:
        start, _ = time.split("_")
        start = start.replace('H', ':')
        start = start.upper().replace('H', ':')
        return start.replace(' ', '')
    else:
        time = time.upper().replace('H', ':')
        return time.replace(' ', '')


def get_video_information(start, end, where, log):
    if(os.path.isdir("videos/" + where)):
        string = "videos/" + where + "/*.mp4"
    elif(os.path.isdir("videos/" + where[0:5])):
        string = "videos/" + where[0:5] + "/*.mp4"
    else:
        message = 'Wrong naming pattern at video diretory for data' + \
            where + '\nCaution all videos from this date will be jumped'
        print(message)
        log.write("%s\n" % message)
        return None, None, None, where
    for file in glob.glob(string):
        try:
            d_start = datetime.datetime.strptime(
                get_video_time_string(file), '%H:%M').time()
            video = cv2.VideoCapture(file)
            fps = video.get(cv2.CAP_PROP_FPS)
            frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            seconds = frame_count/fps
            minutes, seconds = divmod(seconds, 60)
            minutes += d_start.minute
            hour, minutes = divmod(minutes, 60)
            hour += d_start.hour
            d_end = datetime.time(hour=math.floor(hour), minute=math.floor(
                minutes), second=math.floor(seconds))
            if (d_start <= start) and (d_end >= end):
                return file, d_start, d_end, None
        except ValueError:
            message = 'Wrong naming pattern at file ' + file
            log.write("%s\n" % message)
        except:
            message = "Unexpected error:", sys.exc_info()[0]
            log.write("%s\n" % message)
    return None, None, None, None


def isTheSameVideo(d_start, d_end, start, end):
    if(start == None or end == None or d_start == None or d_end == None):
        return False
    else:
        if (d_start <= start) and (d_end >= end):
            return True
        else:
            return False


if __name__ == '__main__':

    file_name = sys.argv[1]
    sheet_name = sys.argv[2]
    sheet_data = sys.argv[3]

    resize = ''
    width = -1
    height = -1

    if(len(sys.argv) == 7):
        resize = sys.argv[3]
        width = int(sys.argv[4])
        height = int(sys.argv[5])

    print(file_name, sheet_name, resize, width, height)

    df = pd.read_excel(file_name, sheet_name=sheet_name)
    data = pd.read_excel(file_name, sheet_name=sheet_data)

    if not os.path.exists("clipped"):
        os.makedirs("clipped")

    f = open("log.txt", "w")

    date_code = translate_code(df['Daterecord'][0], data)
    d_start, d_end = None, None
    label: Dict[int, int] = {}
    where_error = None

    for i in range(0, df.shape[0]):
        start = df['Adstart'][i]
        end = df['Adend'][i]
        channel = df['Channel'][i]
        date_number = int(df['Daterecord'][i])
        date, date_code = translate_code(date_number, data)

        if(start == None or end == None or date == None or channel == None):
            message = 'ERROR: Null information at table in line ' + str(i+2)
            f.write("%s\n" % message)
            continue

        if(start > end):
            message = 'ERROR: Start time is bigger then the end time in line ' + \
                str(i+2)
            f.write("%s\n" % message)
            continue

        if not (date_number in label.keys()):
            label[date_number] = 0

        if(where_error == date):
            message = "ERROR: couldn't clip video for the " + \
                str(i+2) + " record"
            f.write("%s\n" % message)
            continue

        if(not isTheSameVideo(d_start, d_end, start, end)):
            video_file, d_start, d_end, where_error = get_video_information(
                start, end, date, f)

        if(video_file is None or d_start is None or d_end is None):
            message = "ERROR: couldn't clip video for the " + \
                str(i+2) + " record"
            f.write("%s\n" % message)
            continue

        try:
            start = start.replace(hour=(start.hour-d_start.hour))
            start = start.replace(minute=(start.minute-d_start.minute))
            start = start.replace(second=(start.second-d_start.second))
            start = start.replace(microsecond=(start.microsecond+500000))
            end = end.replace(hour=(end.hour-d_start.hour))
            end = end.replace(minute=(end.minute-d_start.minute))
            end = end.replace(second=(end.second-d_start.second))
        except:
            message = 'ERROR: Problem in compute the timestamps in line ' + \
                str(i+2)
            f.write("%s\n" % message)
            continue

        cut = "clipped/" + str(int(channel)) + \
            date_code + str(label[date_number]) + "_" + date[-4:] + ".mp4"
        if(resize == '-r'):
            if(isinstance(width, int) and isinstance(height, int)):
                cut_aux = "clipped/aux" + \
                    str(channel) + date_code + \
                    str(label[date_number]) + "_" + date[-4:] + ".mp4"

                command = "ffmpeg -i " + video_file + " -ss " + \
                    str(start) + " -to " + str(end) + \
                    " -c:v libx264 -crf 30 " + cut_aux
                os.system(command)

                command = "ffmpeg -i " + cut_aux + " -vf scale=" + \
                    str(width) + ":" + str(height) + ",setsar=1:1 " + cut
                os.system(command)
                os.remove(cut_aux)
                label[date_number] += 1
            else:
                print('Enter valid type for width and height')
                break
        else:
            command = "ffmpeg -i " + video_file + " -ss " + \
                str(start) + " -to " + str(end) + \
                " -c:v libx264 -crf 30 " + cut
            os.system(command)
            label[date_number] += 1
    f.close
