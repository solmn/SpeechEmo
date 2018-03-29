import wave
import os


def count():
    total = 0
    path = "../datasets/corpus/IEMOCAP_angry_sad_excited_neutral_test_train/"
    # path = "../datasets/corpus/i/"
    classes = os.listdir(path)
    for i, c in enumerate(classes):
        t = 0
        n_files = os.listdir(path + c + "/train/")
        for k, f in enumerate(n_files):
            total += get_duration(path + c + "/train/" + f)
            t += get_duration(path + c + "/train/" + f)
        print(c + ":" + get_time(t))
    return total
def get_duration(path):
    wav = wave.open(path, mode='r')
    (nchannels, sampwidth, framerate, nframes, comptype, compname) = wav.getparams()
    duration = nframes / float(framerate)
    return duration
def get_time(total):
    
    min = total //60
    # print(min)
    sec = total % 60
    hour = min // 60
    m = min % 60
    return str(int(hour)) + "hr," + str(int(m)) + "min," + str(int(sec)) + "sec"
total = count()
print("Total:"+get_time(total))
