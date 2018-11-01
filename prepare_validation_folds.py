import os
import shutil
import argparse
import common
import numpy as np
from sklearn.model_selection import StratifiedKFold

# pylint: disable=no-member


def _copy_video_segments(fold_number, fold_data, output_path):
    # create the output folder for this fold
    fold_output_path = os.path.join(output_path, str(fold_number))
    if not os.path.exists(fold_output_path):
        os.makedirs(fold_output_path)

    # copy the video segments
    for x in fold_data:
        shutil.copyfile(x, os.path.join(fold_output_path, os.path.basename(x)))


def prepare_validation_folds(input_path, output_path, folds):
    X, y = [], []

    # get all the video segments
    video_segment_files = os.listdir(input_path)
    video_segment_files.sort(key=common.natural_sort_key)

    # determine to which class each video segment belongs to
    print("Processing %d video segments..." % len(video_segment_files))    
    for seg in video_segment_files:
        with open(os.path.join(input_path, seg), 'r') as f:
            seg_data = f.readlines()

            # do we have a consistent class label for the frames of this video segment?
            seg_label = {}
            for sd in seg_data:
                sdl = sd.rstrip().split(' ')[1]
                seg_label[sdl] = True

            if len(seg_label) == 1:
                seg_label = list(seg_label.keys())[0]   # the one and only

                # save the segment and its class label
                X.append(os.path.join(input_path, seg))
                y.append(seg_label)
    print("%d video segments have consistent labelling" % len(X))

    # perform stratified K-fold splitting so that the number of entries per class is balanced across folds
    skf = StratifiedKFold(n_splits=folds)
    print(skf)

    X = np.array(X)     # so that we can access multiple elements via an array of indices
    y = np.array(y)
    for fold_i, (_, ndx) in enumerate(skf.split(X, y)):      # we take the test split as this is a single fold
        fold_X, fold_y = X[ndx], y[ndx]
        print(np.unique(fold_y, return_counts=True))
        _copy_video_segments(fold_i+1, fold_X, output_path)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--input", help="Path to the input folder containing the video segments", default="")
    argparser.add_argument("--output", help="Path to the output file containing the video segments split according to validation folds", default="")
    argparser.add_argument("--folds", help="Number of folds to split the data into.", default=5)
    args = argparser.parse_args()

    if not args.input or not args.output or not args.folds:
        argparser.print_help()
        exit()

    prepare_validation_folds(input_path=args.input, output_path=args.output, folds=int(args.folds))
