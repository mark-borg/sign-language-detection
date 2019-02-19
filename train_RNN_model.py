"""
    This script is used for training our RNN-based network.
    For computational efficiency reasons, the extraction of CNN features
    need to have been done before calling this script. Use one of the
    generate_CNN_features*.py scripts for this.
    Multiple RNNs are trained for different types of input data: raw
    video frames, frame differencing images, MHIs, and optical flow data.
"""
import argparse
import gc
import pytictoc
import msvcrt   # Microsoft specific
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from keras import backend as K
from create_neural_net_model import create_neural_net_model
from generators import VideoSegmentDataGenerator


class _myTrainingCallback(Callback):
    def on_batch_end(self, batch, logs={}):
        if msvcrt.kbhit():  # if a key is pressed
            key = msvcrt.getch()
            if key == b'q' or key == b'Q':
                print('User termination')
                self.model.stop_training = True

    def on_epoch_begin(self, epoch, logs={}):
        lr = float(K.get_value(self.model.optimizer.lr))
        print(" epoch={:02d}, lr={:.5f}".format(epoch+1, lr))

    def on_epoch_end(self, epoch, logs={}):
        gc.collect()


def train_RNN_model(training_data_path, validation_data_path, input_file_mask, batch_size, image_data_shape, video_data_shape, rnn_input_shape, include_cnn_fc1_layer, model_weights_file, learning_rate):

    gc.disable()

    # load the top RNN part of the model, without the convolutional base
    model = create_neural_net_model(image_data_shape, video_data_shape, rnn_input_shape,
            include_convolutional_base=False, include_cnn_fc1_layer=include_cnn_fc1_layer, include_top_layers=True, rnn_model_weights_file=model_weights_file, learning_rate=learning_rate)
            
    # generators for loading the CNN features
    print('Preparing the data generators...')
    print('Training data generator:')
    train_gen = VideoSegmentDataGenerator(training_data_path, input_file_mask, output_batch_size=batch_size)
    print('Validation data generator:')
    validation_gen = VideoSegmentDataGenerator(validation_data_path, input_file_mask, output_batch_size=batch_size)

    # determine class imbalance
    print('Checking for class imbalance')
    class_weights, _ = train_gen.get_class_weights()

    # train the model (saving the model weights after each epoch)
    print('\nCan press (q) to quit the training process...\n\n')
    callbacks = [EarlyStopping(monitor='val_loss', patience=5, verbose=1),
                 ModelCheckpoint(model_weights_file, monitor='val_loss', save_best_only=True, verbose=1),
                 _myTrainingCallback()]
    history = model.fit_generator(generator=train_gen.generator(), steps_per_epoch=train_gen.number_of_batches(),
                    validation_data=validation_gen.generator(), validation_steps=validation_gen.number_of_batches(),
                    epochs=50, callbacks=callbacks, shuffle=False, verbose=1, class_weight=class_weights)
    #model.save_weights(model_weights_file)

    try:
        acc = history.history['acc']
        val_acc = history.history['val_acc']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(1, len(acc) + 1)

        plt.plot(epochs, acc, 'bo-', label='Training acc')
        plt.plot(epochs, val_acc, 'ro-', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.legend()
        plt.figure()

        plt.plot(epochs, loss, 'bo-', label='Training loss')
        plt.plot(epochs, val_loss, 'ro-', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()
        plt.show()
        #plt.savefig('model_training_loss.png')
    except:
        pass    # if user presses 'q'uit during training, we might not have validation accuracy and loss

    gc.enable()
    
    train_gen.stop()
    validation_gen.stop()


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--train", help="Path to the folder(s) containing the CNN data to be used for training. If more than one folder is available, then separate with a semi-colon.", default="")
    argparser.add_argument("--validate", help="Path to the folder(s) containing the CNN data to be used for validation. If more than one folder is available, then separate with a semi-colon.", default="")
    argparser.add_argument("--mask", help="The file mask to use for the CNN data files in the training and validation folders", default="*.txt")
    argparser.add_argument("--model", help="File path and filename for the trained RNN model weights. File name should be *.h5", default="")
    argparser.add_argument("--imwidth", help="Video frame width (in pixels)", default=224)
    argparser.add_argument("--imheight", help="Video frame height (in pixels)", default=224)
    argparser.add_argument("--fc1_layer", help="Include the first fully-connected layer (fc1) of the CNN", default=True)
    argparser.add_argument("--timesteps", help="Timesteps used in the RNN model. Should be set to the length in frames of the video segments", default=20)
    argparser.add_argument("--batch", help="Batch size for the RNN network", default=32)
    argparser.add_argument("--lr", help="Learning rate for the Adam optimiser", default=0.001)
    args = argparser.parse_args()

    if not args.train or not args.validate or not args.model:
        argparser.print_help()
        exit()

    args.imwidth = int(args.imwidth)
    args.imheight = int(args.imheight)
    args.timesteps = int(args.timesteps)
    args.fc1_layer = (args.fc1_layer.lower() == 'true')
    args.batch = int(args.batch)

    image_data_shape = (args.imwidth, args.imheight, 3)                         # image width, image height, channels
    video_clip_data_shape = (args.timesteps, args.imwidth, args.imheight, 3)    # timesteps, image width, image height, channels
    rnn_input_shape = (args.timesteps, 4096) if args.fc1_layer else (args.timesteps, 7, 7, 512)    # timesteps, CNN features width, CNN features height, CNN features channels

    t = pytictoc.TicToc()
    t.tic()

    train_RNN_model(training_data_path=args.train, validation_data_path=args.validate, input_file_mask=args.mask, batch_size=args.batch,
        image_data_shape=image_data_shape, video_data_shape=video_clip_data_shape, rnn_input_shape=rnn_input_shape, include_cnn_fc1_layer=args.fc1_layer,
        model_weights_file=args.model, learning_rate=float(args.lr))

    t.toc('RNN Training')
    