import multiprocessing
import operator
import pickle
import time
import traceback
from pathlib import Path

import samplelib.PackedFaceset
from core import pathex
from core.mplib import MPSharedList
from core.interact import interact as io
from core.joblib import Subprocessor
from DFLIMG import *
from facelib import FaceType, LandmarksProcessor

from .Sample import Sample, SampleType


class SampleLoader:
    samples_cache = dict()
    @staticmethod
    def get_person_id_max_count(samples_path):
        samples = None
        try:
            samples = samplelib.PackedFaceset.load(samples_path)
        except:
            io.log_err(f"Error occured while loading samplelib.PackedFaceset.load {str(samples_path)}, {traceback.format_exc()}")

        if samples is None:
            raise ValueError("packed faceset not found.")
        persons_name_idxs = {}
        for sample in samples:
            persons_name_idxs[sample.person_name] = 0
        return len(list(persons_name_idxs.keys()))

    @staticmethod
    def load(sample_type, samples_path, subdirs=False):
        """
        Return MPSharedList of samples
        """
        samples_cache = SampleLoader.samples_cache

        if str(samples_path) not in samples_cache.keys():
            samples_cache[str(samples_path)] = [None]*SampleType.QTY

        samples = samples_cache[str(samples_path)]

        if            sample_type == SampleType.IMAGE:
            if  samples[sample_type] is None:
                samples[sample_type] = [ Sample(filename=filename) for filename in io.progress_bar_generator( pathex.get_image_paths(samples_path, subdirs=subdirs), "Loading") ]

        elif          sample_type == SampleType.FACE:
            if  samples[sample_type] is None:
                try:
                    result = samplelib.PackedFaceset.load(samples_path)
                except:
                    io.log_err(f"Error occured while loading samplelib.PackedFaceset.load {str(samples_dat_path)}, {traceback.format_exc()}")

                if result is not None:
                    io.log_info (f"Loaded {len(result)} packed faces from {samples_path}")

                if result is None:
                    result = SampleLoader.load_face_samples( pathex.get_image_paths(samples_path, subdirs=subdirs) )

                samples[sample_type] = MPSharedList(result)
        elif          sample_type == SampleType.FACE_TEMPORAL_SORTED:
                result = SampleLoader.load (SampleType.FACE, samples_path)
                result = SampleLoader.upgradeToFaceTemporalSortedSamples(result)
                samples[sample_type] = MPSharedList(result)

        return samples[sample_type]

    @staticmethod
    def load_face_samples(image_paths):
        io.progress_bar("Loading samples", len(image_paths))
        result = []

        with multiprocessing.Pool() as pool:
            futures = []
            for path in image_paths:
                futures.append(pool.apply_async(SampleLoader.load_face_sample, args=(path,)))

            counter = 0
            for future in futures:
                result.append(future.get())
                counter += 1
                if counter == 7:
                    io.progress_bar_inc(7)
                    counter = 0
            io.progress_bar_inc(counter)

        io.progress_bar_close()
        sample_list = []

        for filename, data in result:
            if data is None:
                continue
            ( face_type,
              shape,
              landmarks,
              seg_ie_polys,
              xseg_mask_compressed,
              eyebrows_expand_mod,
              source_filename ) = data
              
            sample_list.append( Sample(filename=filename,
                                        sample_type=SampleType.FACE,
                                        face_type=FaceType.fromString (face_type),
                                        shape=shape,
                                        landmarks=landmarks,
                                        seg_ie_polys=seg_ie_polys,
                                        xseg_mask_compressed=xseg_mask_compressed,
                                        eyebrows_expand_mod=eyebrows_expand_mod,
                                        source_filename=source_filename,
                                    ))
        return sample_list

    @staticmethod
    def load_face_sample(image_path):
        dflimg = DFLIMG.load(Path(image_path))

        if dflimg is None or not dflimg.has_data():
            print(f"FaceSamplesLoader: {image_path} is not a dfl image file.")
            data = None
        else:
            data = (dflimg.get_face_type(),
                    dflimg.get_shape(),
                    dflimg.get_landmarks(),
                    dflimg.get_seg_ie_polys(),
                    dflimg.get_xseg_mask_compressed(),
                    dflimg.get_eyebrows_expand_mod(),
                    dflimg.get_source_filename())

        return image_path, data

    @staticmethod
    def upgradeToFaceTemporalSortedSamples( samples ):
        new_s = [ (s, s.source_filename) for s in samples]
        new_s = sorted(new_s, key=operator.itemgetter(1))

        return [ s[0] for s in new_s]
