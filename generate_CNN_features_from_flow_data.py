"""
    This script extracts CNN features for optical flow data.
"""
import os
import glob
import msvcrt   # Microsoft specific
import argparse
import pytictoc
import cv2
import numpy as np
from keras.models import Model, Sequential
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from keras.applications.vgg16 import VGG16
from keras.utils.vis_utils import plot_model
import common


def _create_cnn_model_for_flow_data(flow_data_shape, K, include_fc1_layer):
    if include_fc1_layer:
        orig_model = VGG16(weights='imagenet', include_top=True)

        # remove last layers, keeping only the layers till fc1
        orig_model.layers.pop()
        orig_model.layers.pop()
    else:
        orig_model = VGG16(weights='imagenet', include_top=False)

    # create VGG16 network with modified input
    flow_data_input = Input(shape=flow_data_shape)

    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(flow_data_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    if include_fc1_layer:
        x = Flatten(name='flatten')(x)
        x = Dense(4096, activation='relu', name='fc1')(x)

    # Create model
    cnn_model = Model(flow_data_input, x, name='modified_vgg16')

    # duplicate the weights of block1_conv1 layer
    assert orig_model.layers[1].name == 'block1_conv1' and cnn_model.layers[1].name == 'block1_conv1', "Error in Network structure!!"
    ws = orig_model.layers[1].get_weights()
    ws[0] = np.dstack((ws[0] for i in range(K)))
    cnn_model.layers[1].set_weights(ws)

    # copy the other weights from the original VGG16 network
    for i in range(2, len(cnn_model.layers)):
        cnn_model.layers[i].set_weights(orig_model.layers[i].get_weights())
    orig_model = None

    # freeze the network
    for lyr in cnn_model.layers:
        lyr.trainable = False

    print('Convolutional base:')
    print(cnn_model.summary())
    plot_model(cnn_model, to_file='CNN_flow.png', show_shapes=True, show_layer_names=True)    
    return cnn_model


def generate_CNN_features_from_flow_data(input_path, input_file_mask, stack_size_K, cnn_model, output_path, groundtruth_file=""):
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
                    # load the stacked flow data from disk
                    stacked_flow_data = np.load(image_j)
                    if len(stacked_flow_data) < stack_size_K:
                        print("!", end='', flush=True)
                        continue

                    # extract the flow data (encoded as images)
                    X = None
                    for fd in stacked_flow_data:
                        # each element in the array of stacked flow data is encoded as a JPEG-compressed RGB image
                        flow_img = cv2.imdecode(fd, cv2.IMREAD_UNCHANGED)
                        X = np.dstack((X, flow_img)) if X is not None else flow_img

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
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--input", help="Path to the input parent folder containing the flow data of the downloaded YouTube videos", default="")
    argparser.add_argument("--mask", help="The file mask to use for the flow data files of the downloaded YouTube videos", default="*.npy")
    argparser.add_argument("--output", help="Path to the output folder where the the CNN features will be extracted to", default="")
    argparser.add_argument("--imwidth", help="Video frame width (in pixels)", default=224)
    argparser.add_argument("--imheight", help="Video frame height (in pixels)", default=224)
    argparser.add_argument("--K", help="Flow data of K consecutive frames are stacked together", default=2)
    argparser.add_argument("--fc1_layer", help="Include the first fully-connected layer (fc1) of the CNN", default=True)
    argparser.add_argument("--groundtruth", help="If groundtruth is available, then we load the file in order to only process video frames which have been labelled.", default="")
    args = argparser.parse_args()

    if not args.input or not args.output:
        argparser.print_help()
        exit()

    args.K = int(args.K)

    input_data_shape = (args.imwidth, args.imheight, args.K*3)   # width, height, channels of stacked flow data
    model = _create_cnn_model_for_flow_data(input_data_shape, args.K, include_fc1_layer=args.fc1_layer)

    generate_CNN_features_from_flow_data(input_path=args.input, input_file_mask=args.mask, stack_size_K=args.K,
                                        cnn_model=model, output_path=args.output, groundtruth_file=args.groundtruth)
