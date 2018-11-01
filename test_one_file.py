import os
import glob
import argparse
import common
import pytictoc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from itertools import chain
from create_neural_net_model import create_neural_net_model

# pylint: disable=no-member


# 0 maps to 'P' (speaking), 1 maps to 'S' (signing), 2 maps to 'n' (other) 
CLASS_MAP = ['P', 'S', 'n']


def visualise_labels(gt_list, title_str=""):
    fig = plt.figure()
    currentAxis = plt.gca()
    for (i, gt) in enumerate(gt_list):
        col = 'gray'
        if gt == 'P':
            col = 'green'
        if gt == 'S':
            col = 'red'
        if gt == 'n':
            col = 'yellow'
        currentAxis.add_patch(Rectangle((i, 0), 1, 4, alpha=1, fill=True, color=col))
    plt.xlim(0, len(gt_list))
    plt.ylim(0, 4)
    plt.title(title_str)
    plt.tight_layout()
    plt.show()
    return fig


def test_one_file(path_to_videos, video_id, groundtruth_file, timesteps, image_data_shape, video_data_shape, rnn_input_shape, include_cnn_fc1_layer, model_weights_file, output_path):

    # load the top RNN part of the model, without the convolutional base
    model = create_neural_net_model(image_data_shape, video_data_shape, rnn_input_shape,
            include_convolutional_base=False, include_cnn_fc1_layer=include_cnn_fc1_layer, include_top_layers=True, rnn_model_weights_file=model_weights_file)

    # open and load the groundtruth data
    print('Loading groundtruth data...')
    gt = pd.read_csv(groundtruth_file, delim_whitespace=True, header=None, names=['video_id', 'frame_id', 'gt'])
    print('ok\n')

    # the video file to be processed
    video_folder = os.path.join(path_to_videos, video_id)

    # select the groundtruth rows for this video
    print('Processing video {} ...'.format(video_id))
    gts = gt.loc[gt['video_id'] == video_id]

    # get all the frames for this video
    frame_list = os.listdir(video_folder)
    frame_list.sort(key=common.natural_sort_key)

    cnn_files = []
    gt_labels = []
    pred_labels = []
    frame_numbers = []

    # go through the sampled video frames for which we have CNN features...
    for frame_file in frame_list:
        frame_num = int(os.path.splitext(frame_file)[0])
        
        # get groundtruth value
        gt_label = '?'
        if not gts.loc[gts['frame_id'] == frame_num].empty:
            rec = gts.loc[gts['frame_id'] == frame_num]
            gt_label = rec['gt'].values[0]
        print(gt_label, end='', flush=True)

        frame_numbers.append(frame_num)
        cnn_files.append(os.path.join(video_folder, frame_file))
        gt_labels.append(gt_label)
    print('\n\n')
    assert len(cnn_files) == len(gt_labels) == len(frame_numbers), 'logical error!!'

    pred_prob = np.zeros(len(frame_numbers))          # probability of the maximal class
    pred_probs = np.zeros((len(frame_numbers), 3))    # probabilities of all classes

    # scan the video with a sliding window of T timesteps
    left_win_size = int(timesteps / 2)
    right_win_size = timesteps -1 - left_win_size
    for (i, fr) in enumerate(frame_numbers):
        window_ndx = chain(range(i - left_win_size, i), range(i, i + right_win_size + 1))
        window_ndx = list(window_ndx)
        window_ndx = [0 if i < 0 else i for i in window_ndx]        # take care of loop boundary conditions
        window_ndx = [len(frame_numbers) - 1 if i >= len(frame_numbers) else i for i in window_ndx]  # take care of loop boundary conditions

        # get the CNN features 
        X = []
        for (j, ndx) in enumerate(window_ndx):
            dt = np.load(cnn_files[ndx])
            dt = np.array(dt['X'])          # has shape (timesteps, CNN feature vector length)
            X.append(dt[0, ...])
        X = np.array(X)

        # package the input data as a batch of size 1
        X = np.expand_dims(X, axis=0)       # a batch of 1, adding an extra dimension

        # process...
        answer = model.predict(X)

        # find the maximum of the predictions (& decode from one-hot-encoding for groundthruth labels)
        for batch_i in range(0, len(answer)):     # we have an answer for each batch (1 answer in this case)
            predicted_class = np.argmax(answer[batch_i])
            predicted_label = CLASS_MAP[predicted_class]
            pred_labels.append(predicted_label)
            pred_prob[i] = answer[batch_i, np.argmax(answer[batch_i])]   # probability for the predicted class
            pred_probs[i, 0:3] = answer                                  # probabilities for all the classes
        print(pred_labels[-1], end='', flush=True)
    print('\n\n')
    assert len(pred_labels) == len(gt_labels) == len(pred_prob) == len(pred_probs), 'logical error during prediction stage!!'
  
    # visualise the groundtruth & the predictions
    fig1 = visualise_labels(gt_labels, 'groundtruth for video {}'.format(video_id))
    if output_path:
        fig1.savefig(os.path.join(output_path, video_id + '_gt.eps'))
    fig2 = visualise_labels(pred_labels, 'predictions for video {}'.format(video_id))
    if output_path:
        fig2.savefig(os.path.join(output_path, video_id + '_pred.eps'))

    # plot the probabilities
    fig3 = plt.figure()
    plt.plot(pred_probs[:, 0], color='green')
    plt.plot(pred_probs[:, 1], color='red')
    plt.plot(pred_probs[:, 2], color='orange')
    plt.show()
    if output_path:
        fig3.savefig(os.path.join(output_path, video_id + '_prob.eps'))    

    # save results to file
    if output_path:
        results_file = open(os.path.join(output_path, video_id+'.txt'), 'w')
        results_file.write('video_id,frame_number,groundtruth_label,predicted_label,predicted_label_probability,probability_P,probability_S,probability_n,mismatch_flag\n')
        for k in range(len(frame_numbers)):
            results_file.write('%s,%d,%s,%s,%f,%f,%f,%f,%s\n' % (video_id, frame_numbers[k], gt_labels[k], pred_labels[k], pred_prob[k], 
                                pred_probs[k,0], pred_probs[k,1], pred_probs[k,2], ' ' if gt_labels[k] == pred_labels[k] else '*WRONG*'))
        results_file.close()

    print('\nready')

           

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--videos", help="Path to the folder containing the video sub-folders with CNN feature data", default="")
    argparser.add_argument("--video_id", help="The ID of the video to process. Should be a sub-folder of the path given by parameter 'videos'", default="")
    argparser.add_argument("--gt", help="Path to the groundtruth file", default="")
    argparser.add_argument("--timesteps", help="Timesteps used in the RNN model. Will depend on the timesteps of the trained RNN model.", default=20)
    argparser.add_argument("--model", help="File path and filename for the trained RNN model weights. File name should be *.h5", default="")
    argparser.add_argument("--imwidth", help="Video frame width (in pixels)", default=224)
    argparser.add_argument("--imheight", help="Video frame height (in pixels)", default=224)
    argparser.add_argument("--fc1_layer", help="Include the first fully-connected layer (fc1) of the CNN", default=True)
    argparser.add_argument("--output", help="Output path where the results and figures will be saved to", default="")
    args = argparser.parse_args()

    if not args.videos or not args.video_id or not args.gt or not args.model:
        argparser.print_help()
        exit()

    args.timesteps = int(args.timesteps)

    image_data_shape = (args.imwidth, args.imheight, 3)                         # image width, image height, channels
    video_clip_data_shape = (args.timesteps, args.imwidth, args.imheight, 3)    # timesteps, image width, image height, channels
    rnn_input_shape = (args.timesteps, 4096) if args.fc1_layer else (args.timesteps, 7, 7, 512)    # timesteps, CNN features width, CNN features height, CNN features channels

    t = pytictoc.TicToc()
    t.tic()

    test_one_file(path_to_videos=args.videos, video_id=args.video_id, groundtruth_file=args.gt, timesteps=int(args.timesteps),
                image_data_shape=image_data_shape, video_data_shape=video_clip_data_shape, rnn_input_shape=rnn_input_shape, include_cnn_fc1_layer=args.fc1_layer,
                model_weights_file=args.model, output_path=args.output)

    t.toc()
