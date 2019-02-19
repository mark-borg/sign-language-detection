"""
    This script extracts CNN features for the given video frames,
    MHI images, or frame differencing images.

    Note that for optical flow data, the script generate_CNN_features_from_flow_data.py
    should be used instead.
"""
import os
import glob
import msvcrt   # Microsoft specific
import argparse
import pytictoc
import numpy as np
from keras import preprocessing
from keras.applications.vgg16 import preprocess_input
import common
from create_neural_net_model import create_cnn_model



def generate_CNN_features(input_path, input_file_mask, cnn_model, output_path, groundtruth_file=""):
    # groundtruth data?
    gt = {}
    have_groundtruth_data = False
    if len(groundtruth_file) > 0:
        try:
            # open and load the groundtruth data
            print('Loading groundtruth data...')
            with open(groundtruth_file, 'r') as gt_file:
                gt_lines = gt_file.readlines()
            for gtl in gt_lines:
                gtf = gtl.rstrip().split(' ')
                if len(gtf) == 3:                   # our groundtruth file has 3 items per line (video ID, frame ID, class label)
                    gt[(gtf[0], int(gtf[1]))] = gtf[2]
            print('ok\n')
            have_groundtruth_data = True
        except:
            pass

    tt = pytictoc.TicToc()

    # get all the video folders
    video_folders = os.listdir(input_path)
    video_folders.sort(key=common.natural_sort_key)

    for (i, video_i) in enumerate(video_folders):
        print('processing video %d of %d:  %s' % (i+1, len(video_folders), video_i))

        # create the output folder; if it exists, then CNN features have already been produced - skip the video
        video_i_output_folder = os.path.join(output_path, video_i)
        if not os.path.exists(video_i_output_folder):
            tt.tic()
            os.makedirs(video_i_output_folder)

            # get the list of extracted frames for this video
            video_i_images = glob.glob(os.path.join(input_path, video_i, input_file_mask))
            video_i_images.sort(key=common.natural_sort_key)       # ensure images are in the correct order to preserve temporal sequence 
            assert(len(video_i_images) > 0), "video %s has no frames!!!" % video_i

            # for each video frame...
            for image_j in video_i_images:
                frame_id = int(os.path.splitext(os.path.basename(image_j))[0])
                skip_frame = False
                try:
                    skip_frame = True if have_groundtruth_data and gt[(video_i, frame_id)] == '?' else False
                except:
                    pass    # safest option is not to skip the frame

                if skip_frame:
                    print("x", end='', flush=True)
                else:
                    # load the image and convert to numpy 3D array
                    img = np.array(preprocessing.image.load_img(image_j))

                    # Note that we don't scale the pixel values because VGG16 was not trained with normalised pixel values!
                    # Instead we use the pre-processing function that comes specifically with the VGG16
                    X = preprocess_input(img)

                    X = np.expand_dims(X, axis=0)       # package as a batch of size 1, by adding an extra dimension

                    # generate the CNN features for this batch
                    print(".", end='', flush=True)
                    X_cnn = cnn_model.predict_on_batch(X)

                    # save to disk
                    output_file = os.path.join(video_i_output_folder, os.path.splitext(os.path.basename(image_j))[0] + '.npy')
                    np.savez(open(output_file, 'wb'), X=X_cnn)
            tt.toc()
        print('\n\n')

        if msvcrt.kbhit():  # if a key is pressed 
            key = msvcrt.getch()
            if key == b'q' or key == b'Q':
                print('User termination')
                return

    print('\n\nReady')


if __name__ == "__main__":
    argparser  =argparse.ArgumentParser()
    argparser.add_argument("--input", help="Path to the input parent folder containing the video frames of the downloaded YouTube videos", default="")
    argparser.add_argument("--mask", help="The file mask to use for the video frames of the downloaded YouTube videos", default="*.jpg")
    argparser.add_argument("--output", help="Path to the output folder where the the CNN features will be extracted to", default="")
    argparser.add_argument("--imwidth", help="Video frame width (in pixels)", default=224)
    argparser.add_argument("--imheight", help="Video frame height (in pixels)", default=224)
    argparser.add_argument("--fc1_layer", help="Include the first fully-connected layer (fc1) of the CNN", default=True)
    argparser.add_argument("--groundtruth", help="If groundtruth is available, then we load the file in order to only process video frames which have been labelled.", default="")
    args = argparser.parse_args()

    if not args.input or not args.output:
        argparser.print_help()
        exit()

    image_data_shape = (args.imwidth, args.imheight, 3)   # width, height, channels
    model = create_cnn_model(image_data_shape, include_fc1_layer=args.fc1_layer)

    generate_CNN_features(input_path=args.input, input_file_mask=args.mask, cnn_model=model, output_path=args.output, groundtruth_file=args.groundtruth)
