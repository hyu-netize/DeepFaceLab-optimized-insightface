# 2023 - Modified by Dmitry Kalashnik.

import math
import multiprocessing
import operator
import os
import sys
import tempfile
from functools import cmp_to_key
from pathlib import Path

import cv2
import numpy as np
from numpy import linalg as npla

from core import imagelib, mathlib, pathex
from core.cv2ex import *
from core.imagelib import estimate_sharpness
from core.interact import interact as io
from core.joblib import Subprocessor, MultiprocessWorker
from core.leras import nn
from DFLIMG import *
from facelib import LandmarksProcessor


def get_face_blur(path, estimate_motion_blur=False):
    filepath = Path(path)
    dflimg = DFLIMG.load(filepath)

    if dflimg is None or not dflimg.has_data():
        io.log_err(f"{filepath.name} is not a dfl image file")
        return [False, [str(filepath), 0]]
    else:
        image = cv2_imread(str(filepath))
        face_mask = LandmarksProcessor.get_image_hull_mask(image.shape, dflimg.get_landmarks())
        image = (image * face_mask).astype(np.uint8)
        if estimate_motion_blur:
            value = cv2.Laplacian(image, cv2.CV_64F, ksize=11).var()
        else:
            value = estimate_sharpness(image)

        return [True, [str(filepath), value]]

def sort_by_blur(worker, input_path):
    io.log_info ("Sorting by blur...")
    worker.set_progress_bar("Loading")
    result_processed, result_error = worker.run_sorting(get_face_blur, pathex.get_image_paths(input_path))

    io.log_info ("Sorting...")
    result_processed = sorted(result_processed, key=operator.itemgetter(1), reverse=True)
    return result_processed, result_error

def sort_by_motion_blur(worker, input_path):
    io.log_info ("Sorting by motion blur...")
    worker.set_progress_bar("Loading")
    result_processed, result_error = worker.run_sorting(get_face_blur, pathex.get_image_paths(input_path), True)

    io.log_info ("Sorting...")
    result_processed = sorted(result_processed, key=operator.itemgetter(1), reverse=True)
    return result_processed, result_error

def get_face_yaw(path):
    filepath = Path(path)
    dflimg = DFLIMG.load(filepath)
    if dflimg is None or not dflimg.has_data():
        io.log_err(f"{filepath.name} is not a dfl image file")
        return [False, [str(filepath)]]

    pitch, yaw, roll = LandmarksProcessor.estimate_pitch_yaw_roll(dflimg.get_landmarks(), size=dflimg.get_shape()[1])
    return [True, [str(filepath), yaw]]

def sort_by_face_yaw(worker, input_path):
    io.log_info ("Sorting by face yaw...")
    worker.set_progress_bar("Loading")
    result_processed, result_error = worker.run_sorting(get_face_yaw, pathex.get_image_paths(input_path))

    io.log_info ("Sorting...")
    result_processed = sorted(result_processed, key=operator.itemgetter(1), reverse=True)
    return result_processed, result_error

def get_face_pitch(path):
    filepath = Path(path)
    dflimg = DFLIMG.load(filepath)
    if dflimg is None or not dflimg.has_data():
        io.log_err(f"{filepath.name} is not a dfl image file")
        return [False, [str(filepath)]]

    pitch, yaw, roll = LandmarksProcessor.estimate_pitch_yaw_roll(dflimg.get_landmarks(), size=dflimg.get_shape()[1])
    return [True, [str(filepath), pitch]]

def sort_by_face_pitch(worker, input_path):
    io.log_info ("Sorting by face pitch...")
    worker, worker.set_progress_bar("Loading")
    result_processed, result_error = worker.run_sorting(get_face_pitch, pathex.get_image_paths(input_path))

    io.log_info ("Sorting...")
    result_processed = sorted(result_processed, key=operator.itemgetter(1), reverse=True)
    return result_processed, result_error

def get_face_rect_size(path):
    filepath = Path(path)
    dflimg = DFLIMG.load(filepath)

    if dflimg is None or not dflimg.has_data():
        io.log_err(f"{filepath.name} is not a dfl image file")
        return [False, [str(filepath)]]

    source_rect = dflimg.get_source_rect()
    rect_area = mathlib.polygon_area(np.array(source_rect[[0, 2, 2, 0]]).astype(np.float32),
                                     np.array(source_rect[[1, 1, 3, 3]]).astype(np.float32))

    return [True, [str(filepath), rect_area]]

def sort_by_face_source_rect_size(worker, input_path):
    io.log_info ("Sorting by face rect size...")
    worker.set_progress_bar("Loading")
    result_processed, result_error = worker.run_sorting(get_face_rect_size, pathex.get_image_paths(input_path))

    io.log_info ("Sorting...")
    result_processed = sorted(result_processed, key=operator.itemgetter(1), reverse=True)
    return result_processed, result_error

def get_image_histogram(path):
    img = cv2_imread(path)
    return [path,
            cv2.calcHist([img], [0], None, [256], [0, 256]),
            cv2.calcHist([img], [1], None, [256], [0, 256]),
            cv2.calcHist([img], [2], None, [256], [0, 256])]

def find_min_hist_score(img_list, start_index):
    img = img_list[start_index]
    min_score_index = start_index + 1
    min_score = float("inf")
    for j in range(start_index + 1, len(img_list)):
        other_img = img_list[j]
        score = compare_histograms(img, other_img)
        if score < min_score:
            min_score = score
            min_score_index = j
    return min_score_index

def compare_histograms(img, other_img):
    return cv2.compareHist(img[1], other_img[1], cv2.HISTCMP_BHATTACHARYYA) + \
           cv2.compareHist(img[2], other_img[2], cv2.HISTCMP_BHATTACHARYYA) + \
           cv2.compareHist(img[3], other_img[3], cv2.HISTCMP_BHATTACHARYYA)


class HistSsimSubprocessor(Subprocessor):
    class Cli(Subprocessor.Cli):
        #override
        def process_data(self, img_list):
            img_list_len = len(img_list)
            for i in range(img_list_len-1):
                min_score_index = find_min_hist_score(img_list, i)
                img_list[i+1], img_list[min_score_index] = img_list[min_score_index], img_list[i+1]
                self.progress_bar_inc(1)

            return img_list

        #override
        def get_data_name (self, data):
            return "Bunch of images"

    #override
    def __init__(self, img_list):
        self.img_list = img_list
        self.img_list_len = len(img_list)

        slice_count = 20000
        sliced_count = self.img_list_len // slice_count

        if sliced_count > 12:
            sliced_count = 11.9
            slice_count = int(self.img_list_len / sliced_count)
            sliced_count = self.img_list_len // slice_count

        self.img_chunks_list = [self.img_list[i*slice_count: (i+1)*slice_count] for i in range(sliced_count)] + \
                               [self.img_list[sliced_count*slice_count:]]

        self.result = []
        super().__init__('HistSsim', HistSsimSubprocessor.Cli, 0)

    #override
    def process_info_generator(self):
        cpu_count = len(self.img_chunks_list)
        io.log_info(f'Running on {cpu_count} threads')
        for i in range(cpu_count):
            yield 'CPU%d' % (i), {'i':i}, {}

    #override
    def on_clients_initialized(self):
        io.progress_bar ("Sorting", len(self.img_list))
        io.progress_bar_inc(len(self.img_chunks_list))

    #override
    def on_clients_finalized(self):
        io.progress_bar_close()

    #override
    def get_data(self, host_dict):
        if len (self.img_chunks_list) > 0:
            return self.img_chunks_list.pop(0)
        return None

    #override
    def on_data_return (self, host_dict, data):
        raise Exception("Fail to process data. Decrease number of images and try again.")

    #override
    def on_result (self, host_dict, data, result):
        self.result += result
        return 0

    #override
    def get_result(self):
        return self.result

class HistSsimSorter:
    ALLOWED_DIFF = 1.2
    SLICE_COUNT = 10000

    def __init__(self, worker, hist_list):
        self.worker = worker
        self.hist_list = hist_list

    @staticmethod
    def sort_histogram_classic(img_list):
        img_list_len = len(img_list)
        i = 0
        while i < img_list_len - 1:
            min_score_index = find_min_hist_score(img_list, i)
            img_list[i + 1], img_list[min_score_index] = img_list[min_score_index], img_list[i + 1]
            i += 1

        return img_list

    def sort_chunk(self, img_list):
        groups_list = [[img_list[0]]]
        img_list_len = len(img_list)

        i = 1
        while i < img_list_len:
            img = img_list[i]
            group_index = -1
            min_diff = float("inf")
            for j in range(len(groups_list)):
                first_img = groups_list[j][0]
                diff_to_first = compare_histograms(first_img, img)

                if diff_to_first < HistSsimSorter.ALLOWED_DIFF and diff_to_first < min_diff:
                    min_diff = diff_to_first
                    group_index = j

            if group_index == -1:
                groups_list.append([img])
                print(f"Gathering image groups: {len(groups_list)}", end="\r")
            else:
                groups_list[group_index].append(img)
            i += 1

        i = 0
        joined_small_groups = []
        while i < len(groups_list):
            group = groups_list[i]
            if len(group) / img_list_len < 0.01:
                joined_small_groups += group
                groups_list.pop(i)
                continue
            i += 1
        groups_list.insert(0, joined_small_groups)
        groups_list.sort(key=lambda x: len(x), reverse=True)

        self.worker.set_progress_bar("Sorting")
        sorted_reference_list = self.worker.run(HistSsimSorter.sort_histogram_classic, groups_list)

        i = 0
        while i < len(sorted_reference_list):
            min_score_index = find_min_hist_score(img_list, i)
            img_list[i + 1], img_list[min_score_index] = img_list[min_score_index], img_list[i + 1]
            i += 1

        result = []
        for group in sorted_reference_list:
            result += group

        return result

    def fast_sort(self):
        slice_count = HistSsimSorter.SLICE_COUNT
        chunks_fraction = math.modf(len(self.hist_list) / slice_count)
        sliced_count = int(chunks_fraction[1])
        if chunks_fraction[0] < 0.3:    # slice 22k to 11k + 11k instead of 10k + 10k + 2k
            sliced_count -= 1

        hist_chunks_list = [self.hist_list[i * slice_count: (i + 1) * slice_count] for i in range(sliced_count)] + \
                          [self.hist_list[sliced_count * slice_count:]]

        result = []
        for i in range(len(hist_chunks_list)):
            print(f"Sorting chunk {i + 1} of {len(hist_chunks_list)}")
            result += self.sort_chunk(hist_chunks_list[i])
        return result


def sort_by_hist(worker, input_path):
    is_fast_mode = io.input_bool ("Use alternative sorting method?", False, help_message="There will be a decrease in accuracy and an increase in speed.")

    io.log_info ("Sorting by histogram similarity...")
    worker.set_progress_bar("Loading histograms")
    hist_list = worker.run(get_image_histogram, pathex.get_image_paths(input_path))
    if is_fast_mode:
        img_list = HistSsimSorter(worker, hist_list).fast_sort()
    else:
        img_list = HistSsimSubprocessor(hist_list).run()

    return img_list, []

def get_face_simple_histogram(path):
    filepath = Path(path)
    dflimg = DFLIMG.load(filepath)
    image = cv2_imread(str(filepath))
    if dflimg is not None and dflimg.has_data():
        face_mask = LandmarksProcessor.get_image_hull_mask(image.shape, dflimg.get_landmarks())
        image = (image * face_mask).astype(np.uint8)

    return [str(filepath), cv2.calcHist([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)], [0], None, [256], [0, 256]), 0]

def calc_face_histogram_difference(job):
    chunk, data_list = job
    for data in chunk:
        current_histogram = data[1]
        score_total = 0
        for other_data in data_list:
            score_total += cv2.compareHist(current_histogram, other_data[1], cv2.HISTCMP_BHATTACHARYYA)
        data[2] = score_total

    return chunk

def sort_by_hist_dissim(worker, input_path):
    io.log_info ("Sorting by histogram dissimilarity...")
    worker.set_progress_bar("Loading")
    img_list = worker.run(get_face_simple_histogram, pathex.get_image_paths(input_path))

    worker.set_progress_bar("Sorting")
    chunk_length = 100
    chunk_start = 0
    jobs = []
    while chunk_start < len(img_list):
        chunk_end = min(chunk_start + chunk_length, len(img_list))
        jobs.append((img_list[chunk_start:chunk_end], img_list))
        chunk_start = chunk_end

    chunks = worker.run(calc_face_histogram_difference, jobs)
    img_list = []
    for chunk in chunks:
        img_list += chunk

    img_list = sorted(img_list, key=operator.itemgetter(2), reverse=True)
    return img_list, []

def get_face_brightness(path):
    brightness = np.mean(cv2.cvtColor(cv2_imread(path), cv2.COLOR_BGR2HSV)[..., 2].flatten())
    return [path, brightness]

def sort_by_brightness(worker, input_path):
    io.log_info ("Sorting by brightness...")
    worker.set_progress_bar("Loading")
    img_list = worker.run(get_face_brightness, pathex.get_image_paths(input_path))
    io.log_info ("Sorting...")
    img_list = sorted(img_list, key=operator.itemgetter(1), reverse=True)
    return img_list, []

def get_face_hue(path):
    hue = np.mean(cv2.cvtColor(cv2_imread(path), cv2.COLOR_BGR2HSV)[..., 0].flatten())
    return [path, hue]

def sort_by_hue(worker, input_path):
    io.log_info ("Sorting by hue...")
    worker.set_progress_bar("Loading")
    img_list = worker.run(get_face_hue, pathex.get_image_paths(input_path))

    io.log_info ("Sorting...")
    img_list = sorted(img_list, key=operator.itemgetter(1), reverse=True)
    return img_list, []

def get_face_black_pixels_count(path):
    img = cv2_imread(path)
    return [path, img[(img == 0)].size]

def sort_by_black(worker, input_path):
    io.log_info("Sorting by amount of black pixels...")
    worker.set_progress_bar("Loading")
    img_list = worker.run(get_face_black_pixels_count, pathex.get_image_paths(input_path))

    io.log_info("Sorting...")
    img_list = sorted(img_list, key=operator.itemgetter(1), reverse=False)
    return img_list, []

def parse_face_original_name(path):
    filepath = Path(path)
    dflimg = DFLIMG.load(filepath)

    if dflimg is None or not dflimg.has_data():
        io.log_err(f"{filepath.name} is not a dfl image file")
        return False, [str(filepath)]

    return True, [str(filepath), dflimg.get_source_filename()]

def sort_by_origname(worker, input_path):
    io.log_info ("Sort by original filename...")
    worker.set_progress_bar("Loading")
    result_processed, result_error = worker.run_sorting(parse_face_original_name, pathex.get_image_paths(input_path))

    io.log_info ("Sorting...")
    result_processed = sorted(result_processed, key=operator.itemgetter(1))
    return result_processed, result_error

def sort_by_oneface_in_image(worker, input_path):
    io.log_info ("Sort by one face in images...")
    image_paths = pathex.get_image_paths(input_path)
    a = np.array ([ ( int(x[0]), int(x[1]) ) \
                      for x in [ Path(filepath).stem.split('_') for filepath in image_paths ] if len(x) == 2
                  ])
    if len(a) > 0:
        idxs = np.ndarray.flatten ( np.argwhere ( a[:,1] != 0 ) )
        idxs = np.unique ( a[idxs][:,0] )
        idxs = np.ndarray.flatten ( np.argwhere ( np.array([ x[0] in idxs for x in a ]) == True ) )
        if len(idxs) > 0:
            io.log_info ("Found %d images." % (len(idxs)) )
            img_list = [ (path,) for i,path in enumerate(image_paths) if i not in idxs ]
            trash_img_list = [ (image_paths[x],) for x in idxs ]
            return img_list, trash_img_list

    io.log_info ("Nothing found. Possible recover original filenames first.")
    return [], []

class FinalHistDissimSubprocessor(Subprocessor):
    class Cli(Subprocessor.Cli):
        #override
        def process_data(self, data):
            idx, pitch_yaw_img_list = data

            for p in range ( len(pitch_yaw_img_list) ):

                img_list = pitch_yaw_img_list[p]
                if img_list is not None:
                    for i in range( len(img_list) ):
                        score_total = 0
                        for j in range( len(img_list) ):
                            if i == j:
                                continue
                            score_total += cv2.compareHist(img_list[i][2], img_list[j][2], cv2.HISTCMP_BHATTACHARYYA)
                        img_list[i][3] = score_total

                    pitch_yaw_img_list[p] = sorted(img_list, key=operator.itemgetter(3), reverse=True)

            return idx, pitch_yaw_img_list

        #override
        def get_data_name (self, data):
            return "Bunch of images"

    #override
    def __init__(self, pitch_yaw_sample_list ):
        self.pitch_yaw_sample_list = pitch_yaw_sample_list
        self.pitch_yaw_sample_list_len = len(pitch_yaw_sample_list)

        self.pitch_yaw_sample_list_idxs = [ i for i in range(self.pitch_yaw_sample_list_len) if self.pitch_yaw_sample_list[i] is not None ]
        self.result = [ None for _ in range(self.pitch_yaw_sample_list_len) ]
        super().__init__('FinalHistDissimSubprocessor', FinalHistDissimSubprocessor.Cli)

    #override
    def process_info_generator(self):
        cpu_count = min(multiprocessing.cpu_count(), 8)
        io.log_info(f'Running on {cpu_count} CPUs')
        for i in range(cpu_count):
            yield 'CPU%d' % (i), {}, {}

    #override
    def on_clients_initialized(self):
        io.progress_bar ("Sort by hist-dissim", len(self.pitch_yaw_sample_list_idxs) )

    #override
    def on_clients_finalized(self):
        io.progress_bar_close()

    #override
    def get_data(self, host_dict):
        if len (self.pitch_yaw_sample_list_idxs) > 0:
            idx = self.pitch_yaw_sample_list_idxs.pop(0)

            return idx, self.pitch_yaw_sample_list[idx]
        return None

    #override
    def on_data_return (self, host_dict, data):
        self.pitch_yaw_sample_list_idxs.insert(0, data[0])

    #override
    def on_result (self, host_dict, data, result):
        idx, yaws_sample_list = data
        self.result[idx] = yaws_sample_list
        io.progress_bar_inc(1)

    #override
    def get_result(self):
        return self.result


def get_face_metrics(path, fast_mode):
    filepath = Path(path)
    try:
        dflimg = DFLIMG.load(filepath)
        if dflimg is None or not dflimg.has_data():
            io.log_err(f"{filepath.name} is not a dfl image file")
            return [False, [str(filepath)]]

        bgr = cv2_imread(str(filepath))
        if bgr is None:
            raise Exception("Unable to load %s" % (filepath.name))

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        if fast_mode:
            source_rect = dflimg.get_source_rect()
            sharpness = mathlib.polygon_area(np.array(source_rect[[0, 2, 2, 0]]).astype(np.float32),
                                             np.array(source_rect[[1, 1, 3, 3]]).astype(np.float32))
        else:
            face_mask = LandmarksProcessor.get_image_hull_mask(gray.shape, dflimg.get_landmarks())
            sharpness = estimate_sharpness((gray[..., None] * face_mask).astype(np.uint8))

        pitch, yaw, roll = LandmarksProcessor.estimate_pitch_yaw_roll(dflimg.get_landmarks(),
                                                                      size=dflimg.get_shape()[1])

        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    except Exception as e:
        io.log_err(e)
        return [False, [str(filepath)]]

    return [True, [str(filepath), sharpness, hist, yaw, pitch]]

def sort_best_faster(worker, input_path):
    return sort_best(worker, input_path, faster=True)

def sort_best(worker, input_path, faster=False):
    target_count = io.input_int ("Target number of faces?", 2000)

    io.log_info ("Performing sort by best faces.")
    if faster:
        io.log_info("Using faster algorithm. Faces will be sorted by source-rect-area instead of blur.")

    worker.set_progress_bar("Loading")
    img_list, trash_img_list = worker.run_sorting(get_face_metrics, pathex.get_image_paths(input_path), faster)
    final_img_list = []

    grads = 128
    imgs_per_grad = round (target_count / grads)

    #instead of math.pi / 2, using -1.2,+1.2 because actually maximum yaw for 2DFAN landmarks are -1.2+1.2
    grads_space = np.linspace (-1.2, 1.2,grads)

    yaws_sample_list = [None]*grads
    for g in io.progress_bar_generator ( range(grads), "Sort by yaw"):
        yaw = grads_space[g]
        next_yaw = grads_space[g+1] if g < grads-1 else yaw

        yaw_samples = []
        for img in img_list:
            s_yaw = -img[3]
            if (g == 0          and s_yaw < next_yaw) or \
               (g < grads-1     and s_yaw >= yaw and s_yaw < next_yaw) or \
               (g == grads-1    and s_yaw >= yaw):
                yaw_samples += [ img ]
        if len(yaw_samples) > 0:
            yaws_sample_list[g] = yaw_samples

    total_lack = 0
    for g in io.progress_bar_generator ( range(grads), ""):
        img_list = yaws_sample_list[g]
        img_list_len = len(img_list) if img_list is not None else 0

        lack = imgs_per_grad - img_list_len
        total_lack += max(lack, 0)

    imgs_per_grad += total_lack // grads


    sharpned_imgs_per_grad = imgs_per_grad*10
    for g in io.progress_bar_generator ( range (grads), "Sort by blur"):
        img_list = yaws_sample_list[g]
        if img_list is None:
            continue

        img_list = sorted(img_list, key=operator.itemgetter(1), reverse=True)

        if len(img_list) > sharpned_imgs_per_grad:
            trash_img_list += img_list[sharpned_imgs_per_grad:]
            img_list = img_list[0:sharpned_imgs_per_grad]

        yaws_sample_list[g] = img_list


    yaw_pitch_sample_list = [None]*grads
    pitch_grads = imgs_per_grad

    for g in io.progress_bar_generator ( range (grads), "Sort by pitch"):
        img_list = yaws_sample_list[g]
        if img_list is None:
            continue

        pitch_sample_list = [None]*pitch_grads

        grads_space = np.linspace (-math.pi / 2,math.pi / 2, pitch_grads )

        for pg in range (pitch_grads):

            pitch = grads_space[pg]
            next_pitch = grads_space[pg+1] if pg < pitch_grads-1 else pitch

            pitch_samples = []
            for img in img_list:
                s_pitch = img[4]
                if (pg == 0                and s_pitch < next_pitch) or \
                   (pg < pitch_grads-1     and s_pitch >= pitch and s_pitch < next_pitch) or \
                   (pg == pitch_grads-1    and s_pitch >= pitch):
                    pitch_samples += [ img ]

            if len(pitch_samples) > 0:
                pitch_sample_list[pg] = pitch_samples
        yaw_pitch_sample_list[g] = pitch_sample_list

    yaw_pitch_sample_list = FinalHistDissimSubprocessor(yaw_pitch_sample_list).run()

    for g in io.progress_bar_generator (range (grads), "Fetching the best"):
        pitch_sample_list = yaw_pitch_sample_list[g]
        if pitch_sample_list is None:
            continue

        n = imgs_per_grad

        while n > 0:
            n_prev = n
            for pg in range(pitch_grads):
                img_list = pitch_sample_list[pg]
                if img_list is None:
                    continue
                final_img_list += [ img_list.pop(0) ]
                if len(img_list) == 0:
                    pitch_sample_list[pg] = None
                n -= 1
                if n == 0:
                    break
            if n_prev == n:
                break

        for pg in range(pitch_grads):
            img_list = pitch_sample_list[pg]
            if img_list is None:
                continue
            trash_img_list += img_list

    return final_img_list, trash_img_list

"""
def sort_by_vggface(worker, input_path):
    io.log_info ("Sorting by face similarity using VGGFace model...")

    model = VGGFace()

    final_img_list = []
    trash_img_list = []

    image_paths = pathex.get_image_paths(input_path)
    img_list = [ (x,) for x in image_paths ]
    img_list_len = len(img_list)
    img_list_range = [*range(img_list_len)]

    feats = [None]*img_list_len
    for i in io.progress_bar_generator(img_list_range, "Loading"):
        img = cv2_imread( img_list[i][0] ).astype(np.float32)
        img = imagelib.normalize_channels (img, 3)
        img = cv2.resize (img, (224,224) )
        img = img[..., ::-1]
        img[..., 0] -= 93.5940
        img[..., 1] -= 104.7624
        img[..., 2] -= 129.1863
        feats[i] = model.predict( img[None,...] )[0]

    tmp = np.zeros( (img_list_len,) )
    float_inf = float("inf")
    for i in io.progress_bar_generator ( range(img_list_len-1), "Sorting" ):
        i_feat = feats[i]

        for j in img_list_range:
            tmp[j] = npla.norm(i_feat-feats[j]) if j >= i+1 else float_inf

        idx = np.argmin(tmp)

        img_list[i+1], img_list[idx] = img_list[idx], img_list[i+1]
        feats[i+1], feats[idx] = feats[idx], feats[i+1]

    return img_list, trash_img_list
"""

def sort_by_absdiff(worker, input_path):
    io.log_info ("Sorting by absolute difference...")

    is_sim = io.input_bool ("Sort by similar?", True, help_message="Otherwise sort by dissimilar.")

    from core.leras import nn

    device_config = nn.DeviceConfig.ask_choose_device(choose_only_one=True)
    nn.initialize( device_config=device_config, data_format="NHWC" )
    tf = nn.tf

    image_paths = pathex.get_image_paths(input_path)
    image_paths_len = len(image_paths)

    batch_size = 512
    batch_size_remain = image_paths_len % batch_size

    i_t = tf.placeholder (tf.float32, (None,None,None,None) )
    j_t = tf.placeholder (tf.float32, (None,None,None,None) )

    outputs_full = []
    outputs_remain = []

    for i in range(batch_size):
        diff_t = tf.reduce_sum( tf.abs(i_t-j_t[i]), axis=[1,2,3] )
        outputs_full.append(diff_t)
        if i < batch_size_remain:
            outputs_remain.append(diff_t)

    def func_bs_full(i,j):
        return nn.tf_sess.run (outputs_full, feed_dict={i_t:i,j_t:j})

    def func_bs_remain(i,j):
        return nn.tf_sess.run (outputs_remain, feed_dict={i_t:i,j_t:j})

    import h5py
    db_file_path = Path(tempfile.gettempdir()) / 'sort_cache.hdf5'
    db_file = h5py.File( str(db_file_path), "w")
    db = db_file.create_dataset("results", (image_paths_len,image_paths_len), compression="gzip")

    pg_len = image_paths_len // batch_size
    if batch_size_remain != 0:
        pg_len += 1

    pg_len = int( (  pg_len*pg_len - pg_len ) / 2 + pg_len )

    io.progress_bar ("Computing", pg_len)
    j=0
    while j < image_paths_len:
        j_images = [ cv2_imread(x) for x in image_paths[j:j+batch_size] ]
        j_images_len = len(j_images)

        func = func_bs_remain if image_paths_len-j < batch_size else func_bs_full

        i=0
        while i < image_paths_len:
            if i >= j:
                i_images = [ cv2_imread(x) for x in image_paths[i:i+batch_size] ]
                i_images_len = len(i_images)
                result = func (i_images,j_images)
                db[j:j+j_images_len,i:i+i_images_len] = np.array(result)
                io.progress_bar_inc(1)

            i += batch_size
        db_file.flush()
        j += batch_size

    io.progress_bar_close()

    next_id = 0
    sorted = [next_id]
    for i in io.progress_bar_generator ( range(image_paths_len-1), "Sorting" ):
        id_ar = np.concatenate ( [ db[:next_id,next_id], db[next_id,next_id:] ] )
        id_ar = np.argsort(id_ar)


        next_id = np.setdiff1d(id_ar, sorted, True)[ 0 if is_sim else -1]
        sorted += [next_id]
    db_file.close()
    db_file_path.unlink()

    img_list = [ (image_paths[x],) for x in sorted]
    return img_list, []

def rename(chunk):
    for data in chunk:
        src, dst = data
        try:
            src.rename(dst)
        except:
            io.log_info('fail to rename %s' % src.name)

def final_process(worker, input_path, img_list, trash_img_list):
    if len(trash_img_list) != 0:
        parent_input_path = input_path.parent
        trash_path = parent_input_path / (input_path.stem + '_trash')
        trash_path.mkdir (exist_ok=True)

        io.log_info("Trashing %d items to %s" % (len(trash_img_list), str(trash_path)))

        for filename in pathex.get_image_paths(trash_path):
            Path(filename).unlink()

        for i in io.progress_bar_generator( range(len(trash_img_list)), "Moving trash", leave=False):
            src = Path (trash_img_list[i][0])
            dst = trash_path / src.name
            try:
                src.rename (dst)
            except:
                io.log_info('fail to trashing %s' % src.name)

        io.log_info("")

    if len(img_list) != 0:
        tmp_list = []
        for i in range(len(img_list)):
            src = Path (img_list[i][0])
            dst = input_path / ('%.5d_%s' % (i, src.name))
            tmp_list.append((src, dst))

        chunk_length = 1000
        chunk_start = 0
        jobs = []
        while chunk_start < len(tmp_list):
            chunk_end = min(chunk_start + chunk_length, len(tmp_list))
            jobs.append((tmp_list[chunk_start:chunk_end]))
            chunk_start = chunk_end
        io.log_info("Renaming...")
        worker.run(rename, jobs)

        tmp_list = []
        for i in range(len(img_list)):
            src = Path(img_list[i][0])
            src = input_path / ('%.5d_%s' % (i, src.name))
            dst = input_path / ('%.5d%s' % (i, src.suffix))
            tmp_list.append((src, dst))

        chunk_length = 1000
        chunk_start = 0
        jobs = []
        while chunk_start < len(tmp_list):
            chunk_end = min(chunk_start + chunk_length, len(tmp_list))
            jobs.append((tmp_list[chunk_start:chunk_end]))
            chunk_start = chunk_end
        worker.run(rename, jobs)

sort_func_methods = {
    'blur':        ("blur", sort_by_blur),
    'motion-blur': ("motion_blur", sort_by_motion_blur),
    'face-yaw':    ("face yaw direction", sort_by_face_yaw),
    'face-pitch':  ("face pitch direction", sort_by_face_pitch),
    'face-source-rect-size' : ("face rect size in source image", sort_by_face_source_rect_size),
    'hist':        ("histogram similarity", sort_by_hist),
    'hist-dissim': ("histogram dissimilarity", sort_by_hist_dissim),
    'brightness':  ("brightness", sort_by_brightness),
    'hue':         ("hue", sort_by_hue),
    'black':       ("amount of black pixels", sort_by_black),
    'origname':    ("original filename", sort_by_origname),
    'oneface':     ("one face in image", sort_by_oneface_in_image),
    'absdiff':     ("absolute pixel difference", sort_by_absdiff),
    'final':       ("best faces", sort_best),
    'final-fast':  ("best faces faster", sort_best_faster),
}

def main (input_path, sort_by_method=None):
    io.log_info ("Running sort tool.\r\n")

    if sort_by_method is None:
        io.log_info(f"Choose sorting method:")

        key_list = list(sort_func_methods.keys())
        for i, key in enumerate(key_list):
            desc, func = sort_func_methods[key]
            io.log_info(f"[{i}] {desc}")

        io.log_info("")
        id = io.input_int("", 5, valid_list=[*range(len(key_list))] )

        sort_by_method = key_list[id]
    else:
        sort_by_method = sort_by_method.lower()

    worker = MultiprocessWorker(preserve_pool=True)
    desc, func = sort_func_methods[sort_by_method]
    img_list, trash_img_list = func(worker, input_path)

    final_process (worker, input_path, img_list, trash_img_list)
    worker.close_pool()
