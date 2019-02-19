"""
    Tests and evaluates an RNN on the test fold of the dataset.
"""
import argparse
import gc
import pytictoc
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import common
from create_neural_net_model import create_neural_net_model
from generators import VideoSegmentDataGenerator


def test_RNN_model(test_data_path, input_file_mask, model_weights_file, output_results_file, batch_size, image_data_shape, video_data_shape, rnn_input_shape, include_cnn_fc1_layer):

    gc.disable()

    # load the top RNN part of the model, without the convolutional base
    model = create_neural_net_model(image_data_shape, video_data_shape, rnn_input_shape,
            include_convolutional_base=False, include_cnn_fc1_layer=include_cnn_fc1_layer, include_top_layers=True, rnn_model_weights_file=model_weights_file)

    # generator for loading the CNN features of the test dataset
    test_gen = VideoSegmentDataGenerator(test_data_path, input_file_mask, output_batch_size=batch_size, return_sample_id=True, do_timings=True)
    num_test_batches = test_gen.number_of_batches()
    gen = test_gen.generator()

    # run predictions on the test dataset
    y_pred = []
    y_true = []
    y_prob = []
    y_id = []
    correct = 0
    for k in range(0, num_test_batches):
        print('testing batch %d of %d' % (k+1, num_test_batches))
        X, Y, idb = next(gen)
        answer = model.predict(X)

        # find the maximum of the predictions (& decode from one-hot-encoding for groundthruth labels)
        for i in range(0,len(answer)):
            y_pred.append(np.argmax(answer[i]))
            y_true.append(np.argmax(Y[i]))
            y_prob.append(answer[i])
            y_id.append(idb[i])
            if y_pred[i] == y_true[i]:
                correct += 1

    # display performance results
    print("correct predictions %d of %d (%f percent)" % (correct, len(y_pred), 100.0 * correct / len(y_pred)))
    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred, target_names=test_gen.get_class_names()))

    # save results to file
    results_file = open(output_results_file, 'w') 
    results_file.write('video clip, probability class 0 (%s), probability class 1 (%s), probability class 2 (%s), predicted, groundtruth, correct?\n' % tuple(test_gen.get_class_names()))
    lines = []
    for k in range(len(y_id)):
        lines.append('%s, %f, %f, %f, %d, %d, %s\n' % (y_id[k], y_prob[k][0], y_prob[k][1], y_prob[k][2], y_pred[k], y_true[k], 'ok' if y_pred[k] == y_true[k] else '*WRONG*'))
    lines.sort(key=common.natural_sort_key)
    for s in lines:
        results_file.write(s)
    results_file.write('\n')
    results_file.write('correct predictions %d of %d (%f percent)\n\n' % (correct, len(y_pred), 100.0 * correct / len(y_pred)))
    results_file.write('confusion matrix:\n')
    results_file.write(str(confusion_matrix(y_true, y_pred)))
    results_file.write('\n\n')
    results_file.write(classification_report(y_true, y_pred, target_names=test_gen.get_class_names()))
    results_file.close()

    gc.enable()


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--test", help="Path to the folder(s) containing the CNN data to be used for validation. If more than one folder is available, then separate with a semi-colon.", default="")
    argparser.add_argument("--mask", help="The file mask to use for the CNN data files in the test folder", default="*.txt")
    argparser.add_argument("--model", help="File path to the trained RNN model weights", default="")
    argparser.add_argument("--results", help="Path and name of file to use for saving the test results. Extension can be .txt", default="./results.txt")
    argparser.add_argument("--imwidth", help="Video frame width (in pixels)", default=224)
    argparser.add_argument("--imheight", help="Video frame height (in pixels)", default=224)
    argparser.add_argument("--fc1_layer", help="Include the first fully-connected layer (fc1) of the CNN", default=True)
    argparser.add_argument("--timesteps", help="Timesteps used in the RNN model. Should be set to the length in frames of the video segments", default=20)
    argparser.add_argument("--batch", help="Batch size for the RNN network", default=1024)
    args = argparser.parse_args()

    if not args.test or not args.model or not args.results:
        argparser.print_help()
        exit()

    args.imwidth = int(args.imwidth)
    args.imheight = int(args.imheight)
    args.timesteps = int(args.timesteps)

    image_data_shape = (args.imwidth, args.imheight, 3)                         # image width, image height, channels
    video_clip_data_shape = (args.timesteps, args.imwidth, args.imheight, 3)    # timesteps, image width, image height, channels
    rnn_input_shape = (args.timesteps, 4096) if args.fc1_layer else (args.timesteps, 7, 7, 512)    # timesteps, CNN features width, CNN features height, CNN features channels

    t = pytictoc.TicToc()
    t.tic()
    
    test_RNN_model(test_data_path=args.test, input_file_mask=args.mask, model_weights_file=args.model, output_results_file=args.results,
        batch_size=int(args.batch), image_data_shape=image_data_shape, video_data_shape=video_clip_data_shape, rnn_input_shape=rnn_input_shape,
        include_cnn_fc1_layer=args.fc1_layer)
    
    t.toc('RNN Testing')
