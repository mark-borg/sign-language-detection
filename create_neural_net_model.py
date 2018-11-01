import os
from keras.models import Model, Sequential
from keras.layers import Dropout, Flatten, Dense, LSTM, GRU, BatchNormalization, TimeDistributed, Bidirectional, MaxPool2D
from keras.optimizers import SGD, Adam, RMSprop
from keras.applications.vgg16 import VGG16
from keras.utils.vis_utils import plot_model


def create_cnn_model(image_data_shape, include_fc1_layer):
    if include_fc1_layer:
        orig_model = VGG16(weights='imagenet', include_top=True, input_shape=image_data_shape)

        # remove last layers, keeping only the layers till fc1
        orig_model.layers.pop()
        orig_model.layers.pop()
        cnn_model = Model(orig_model.input, orig_model.layers[-1].output)
    else:
        cnn_model = VGG16(weights='imagenet', include_top=False, input_shape=image_data_shape)

    print('Convolutional base:')
    print(cnn_model.summary())
    plot_model(cnn_model, to_file='CNN.png', show_shapes=True, show_layer_names=True)    
    return cnn_model


def create_neural_net_model(image_data_shape, video_clip_data_shape, rnn_input_shape,
                            include_convolutional_base=True, include_cnn_fc1_layer=True,
                            include_top_layers=True, rnn_model_weights_file=None, learning_rate=0.001):

    if include_convolutional_base:
        # load the convolutional base 
        vgg16_model = create_cnn_model(image_data_shape, include_cnn_fc1_layer)

        cnn_model = Sequential()
        # handle temporal dimension & re-freeze the CNN layers
        cnn_model.add(TimeDistributed(vgg16_model, input_shape=video_clip_data_shape))
        cnn_model.layers[0].trainable = False

    if include_top_layers:
        # add the recurrent layers
        rnn_model = Sequential()
        if not include_cnn_fc1_layer:
            rnn_model.add(TimeDistributed(Flatten(), input_shape=rnn_input_shape))
        rnn_model.add(GRU(256, activation='tanh', recurrent_activation='hard_sigmoid', dropout=0.3, return_sequences=True, input_shape=rnn_input_shape))
        rnn_model.add(GRU(256, activation='tanh', recurrent_activation='hard_sigmoid', dropout=0.3))
        rnn_model.add(BatchNormalization())
        rnn_model.add(Dense(64, activation='relu'))
        rnn_model.add(Dropout(0.4))
        rnn_model.add(Dense(3, activation='softmax'))
    
        # optimiser and learning rate
        #opt = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        opt = Adam(lr=learning_rate, decay=1e-6)
        #opt = RMSprop(lr=0.001)
        
        rnn_model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

        # if model has already been trained or is undergoing training, then load the trained weights from file
        if rnn_model_weights_file is not None:
            if os.path.exists(rnn_model_weights_file):
                print('(***) Loading model weights from file %s' % rnn_model_weights_file)
                rnn_model.load_weights(rnn_model_weights_file)

    # are we returning a full model?
    if include_convolutional_base and include_top_layers:
        # now that we have loaded the weights for the recurrent part, we can combine the CNN and RNN together
        model = Sequential()
        for lyr in cnn_model.layers:
            model.add(lyr)         # here we are probably adding a reference, but it doesn't matter because model will not be returned
        model.layers[0].trainable = False

        for lyr in rnn_model.layers:
            model.add(lyr)         # here we are probably adding a reference, but it doesn't matter because model will not be returned
        print(model.summary())
        plot_model(model, to_file='td(CNN)_GRU.png', show_shapes=True, show_layer_names=True)
        return model

    # else we return whatever part of the model the user requested
    model = cnn_model if include_convolutional_base else rnn_model
    print(model.summary())
    diagram_name = 'td(CNN)_' if include_convolutional_base else '_GRU'
    plot_model(model, to_file=diagram_name+'.png', show_shapes=True, show_layer_names=True)

    return model


if __name__ == "__main__":  

    IMAGE_DATA_SHAPE = (224, 224, 3)            # image width, image height, channels
    VIDEO_CLIP_DATA_SHAPE = (20, 224, 224, 3)   # timesteps, image width, image height, channels
    RNN_INPUT_SHAPE = (20, 7, 7, 512)           # timesteps, CNN features width, CNN features height, CNN features channels

    create_neural_net_model(image_data_shape=IMAGE_DATA_SHAPE, video_clip_data_shape=VIDEO_CLIP_DATA_SHAPE, rnn_input_shape=RNN_INPUT_SHAPE)

    create_neural_net_model(image_data_shape=IMAGE_DATA_SHAPE, video_clip_data_shape=VIDEO_CLIP_DATA_SHAPE, rnn_input_shape=RNN_INPUT_SHAPE,
        include_convolutional_base=True, include_top_layers=False)

    create_neural_net_model(image_data_shape=IMAGE_DATA_SHAPE, video_clip_data_shape=VIDEO_CLIP_DATA_SHAPE, rnn_input_shape=RNN_INPUT_SHAPE,
        include_convolutional_base=False, include_top_layers=True)
