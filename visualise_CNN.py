"""
    Experiments in visualising the activations of the CNN base of the network
"""
import argparse
import math
import skimage.io
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras import preprocessing
from keras.applications.vgg16 import preprocess_input
from keras import activations
from keras import backend as K
from vis.visualization import visualize_cam, overlay
from vis.utils import utils
from create_neural_net_model import create_cnn_model

# most of the code used here is based on:
#   https://github.com/waleedka/cnn-visualization/blob/master/cnn_visualization.ipynb


def _tensor_summary(tensor):
    """Display shape, min, and max values of a tensor."""
    print("shape: {}  min: {}  max: {}".format(tensor.shape, tensor.min(), tensor.max()))

    
def _normalize(image):
    """Takes a tensor of 3 dimensions (height, width, colors) and normalizes it's values
    to be between 0 and 1 so it's suitable for displaying as an image."""
    image = image.astype(np.float32)
    return (image - image.min()) / (image.max() - image.min() + 1e-5)


def _display_images(images, titles=None, figure_title=None, cols=5, interpolation=None, cmap="Greys_r"):
    """
    images: A list of images. I can be either:
        - A list of Numpy arrays. Each array represents an image.
        - A list of lists of Numpy arrays. In this case, the images in
          the inner lists are concatentated to make one image.
    """
    titles = titles or [""] * len(images)
    rows = math.ceil(len(images) / cols)
    height_ratio = 1.2 * (rows/cols) * (0.5 if type(images[0]) is not np.ndarray else 1)
    plt.figure(figsize=(15, 15 * height_ratio))
    if figure_title:
        plt.suptitle(figure_title)
    i = 1
    for image, title in zip(images, titles):
        plt.subplot(rows, cols, i)
        plt.axis("off")
        # Is image a list? If so, merge them into one image.
        if type(image) is not np.ndarray:
            image = [_normalize(g) for g in image]
            image = np.concatenate(image, axis=1)
        else:
            image = _normalize(image)
        plt.title(title, fontsize=9)
        plt.imshow(image, cmap=cmap, interpolation=interpolation)
        i += 1
    plt.show()


def view_filters(model, layer_name):
    lyr = model.get_layer(layer_name)
    wts = lyr.get_weights()
    if not wts:
        print('layer {} has no weights!'.format(layer_name))
    else:
        wts = wts[0]    # first entry is the filter weights; second entry is the bias values
        print("filter weights of layer {} have shape {}".format(layer_name, wts.shape))
        if wts.shape[2] <= 3:
            _display_images([wts[:, :, ::-1, i] for i in range(wts.shape[3])], cols=16, interpolation="none",
                            figure_title="{} filters of layer {} with shape {}".format(wts.shape[3], layer_name, wts.shape[0:3]))
        else:
            for j in range(wts.shape[3]):
                _display_images([wts[:, :, i, j] for i in range(wts.shape[2])], cols=16, interpolation="none",
                                figure_title="{} channels of filter #{} of layer {} with shape {}".format(wts.shape[2], j, layer_name, wts.shape[0:3]))


def read_feature_map(model, x, layer_name):
    """Return the activation values for the specific layer"""
    # Create Keras function to read the output of a specific layer
    _get_layer_output = K.function([model.layers[0].input], [model.get_layer(layer_name).output])
    outputs = _get_layer_output([x])[0]
    _tensor_summary(outputs)
    return outputs[0]
    

def view_feature_map(model, x, layer_name, cols=None):
    outputs = read_feature_map(model, x, layer_name)
    if outputs.ndim < 3:
        plt.figure()
        plt.plot(outputs)
        plt.suptitle('output of layer {}, shape: {}'.format(layer_name, outputs.shape))
    else:
        cols = cols if cols is not None else int(math.sqrt(outputs.shape[-1]))
        _display_images([outputs[:, :, i] for i in range(outputs.shape[-1])], cols=cols, figure_title='activations (feature maps) of layer {}, shape: {}'.format(layer_name, outputs.shape))


def _build_backprop(model, loss):
    # Gradient of the input image with respect to the loss function
    gradients = K.gradients(loss, model.input)[0]
    # Normalize the gradients
    gradients /= (K.sqrt(K.mean(K.square(gradients))) + 1e-5)
    # Keras function to calculate the gradients and loss
    return K.function([model.input], [loss, gradients])


def generate_maximising_image_for_class(model, layer_name, X):
    # generate the CNN features for this image
    X_cnn = model.predict_on_batch(X)

    label_index = np.argmax(X_cnn)
    loss_function = K.mean(model.get_layer(layer_name).output[:, label_index])

    # start with a random image
    random_image = np.random.random(X.shape)

    # backprop function
    backprop = _build_backprop(model, loss_function)

    # iteratively apply gradient ascent
    for i in range(50):
        loss, grads = backprop([random_image])
        
        # multiply gradients by the learning rate and add to the image
        # Optionally, apply a gaussian filter to the gradients to smooth out the generated image. This gives better results.
        # The first line, which is commented out, is the native method and the following line uses the filter. 
        #
        # random_image += grads * .1
        random_image += skimage.filters.gaussian(np.clip(grads, -1, 1), 2)

        # Print loss value
        if i % 10 == 0:
            print('Loss:', loss)

    _tensor_summary(random_image)
    _display_images(random_image[..., ::-1], cols=2, figure_title='maximising image for class/feature with index {} for layer {}'.format(label_index, layer_name))


def view_gradients_on_image(model, layer_name, X):
    # generate the CNN features for this image
    X_cnn = model.predict_on_batch(X)

    # Visualise the gradients on the input image
    # loss function that optimizes one fully-connected feature/class
    label_index = np.argmax(X_cnn)
    loss_function = K.mean(model.get_layer(layer_name).output[:, label_index])

    # backprop function
    backprop = _build_backprop(model, loss_function)

    # calculate gradients on the input image
    _, grads = backprop([X])
    _tensor_summary(grads)

    # visualize the gradients
    grad_image = _normalize(grads)
    _display_images([np.sum(grad_image[0], axis=2)], cols=2, figure_title='gradients on input image for layer {} and index {}'.format(layer_name, label_index))


def view_grad_cam(model, X, prediction_layer_name, cnn_layer_name, filters=None):
    pred_layer_idx = utils.find_layer_idx(cnn_model, prediction_layer_name)
    last_cnn_layer_idx = utils.find_layer_idx(cnn_model, cnn_layer_name)
    grads = visualize_cam(cnn_model, layer_idx=pred_layer_idx, 
                            filter_indices=filters,        # filter/class indices within the layer to be maximized (None = all filters/classes are visualized).
                            penultimate_layer_idx=last_cnn_layer_idx,
                            seed_input=X,
                            backprop_modifier=None)

    # the channel-wise mean of the resulting feature map is the heatmap of the class activation
    heatmap = np.mean(grads, axis=-1)

    # heatmap post processing for visualisation purposes
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    plt.matshow(heatmap)
    plt.show()

    img2 = cv2.imread(args.test_image)
    heatmap = cv2.resize(heatmap, (img2.shape[1], img2.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = np.uint8(heatmap * 0.6 + img2 * 0.4)
    cv2.imshow('heatmap superimposed on image', superimposed_img)
    cv2.waitKey(100)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--test_image", help="Path to the test image", default="")
    argparser.add_argument("--imwidth", help="Video frame width (in pixels)", default=224)
    argparser.add_argument("--imheight", help="Video frame height (in pixels)", default=224)
    argparser.add_argument("--fc1_layer", help="Include the first fully-connected layer (fc1) of the CNN", default=True)
    args = argparser.parse_args()

    args.test_image = 'e://sld//frames//sl//mX4HBzSySM8//1212.jpg'

    if not args.test_image:
        argparser.print_help()
        exit()

    args.imwidth = int(args.imwidth)
    args.imheight = int(args.imheight)
    image_data_shape = (args.imwidth, args.imheight, 3)

    # load the image and convert to numpy 3D array
    print('Generating CNN features for {}...'.format(args.test_image))
    img = np.array(preprocessing.image.load_img(args.test_image))
    cv2.imshow('input image', cv2.imread(args.test_image))
    cv2.waitKey(100)

    # Note that we don't scale the pixel values because VGG16 was not trained with normalised pixel values!
    # Instead we use the pre-processing function that comes specifically with the VGG16
    X = preprocess_input(img)
    X = np.expand_dims(X, axis=0)       # package as a batch of size 1, by adding an extra dimension

    # load the CNN model
    cnn_model = create_cnn_model(image_data_shape, include_fc1_layer=args.fc1_layer)
    

    # (1)----- view some of the filters ------
    view_filters(cnn_model, "block1_conv1")
    #view_filters(cnn_model, "block2_conv1")

    # (2)----- view the feature maps of some of the convolutional layers -----
    view_feature_map(cnn_model, X, 'block1_conv1')
    #outputs = read_feature_map(cnn_model, X, 'block1_conv1')
    view_feature_map(cnn_model, X, 'block5_conv2')
    view_feature_map(cnn_model, X, 'fc1')
    
    # (3)----- visualise the gradients on the input image ------
    view_gradients_on_image(cnn_model, 'fc1', X)
    
    # (4)----- generate an image that maximises a class ------
    generate_maximising_image_for_class(cnn_model, 'fc1', X)

    # (5)----- visualise a class activation map via grad-CAM ------
    view_grad_cam(cnn_model, X, 'fc1', 'block5_pool', filters=None)

    print('Ready')
