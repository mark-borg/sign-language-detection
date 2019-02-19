"""
    This Python script is used to train an SVM, with the input data to the SVM being the CNN features
    extracted using one of the scripts generate_CNN_features*.py.

    The SVM, in contrast to the RNN, considers the full set of CNN features as a single vector (single
    data point), ignoring the temporal dimension. In other words, when using the CNN features extracted
    from layer fc1, and with a timestep of 4 seconds sampled at 5fps (20 frames), we have a set of 20
    input vectors each of length 4096: [20 x 4096]. We flatten this vector into a single vector of
    length [81920] and feed this to the SVM. If PCA is enabled, then we take the 50 principal components
    of this vector.
"""
import argparse
import pickle
import pytictoc
from sklearn import svm
from sklearn.decomposition import IncrementalPCA
from generators import VideoSegmentDataGenerator


def train_SVM_model(training_data_path, input_file_mask, output_model_file, num_train_samples=3000, do_pca=False):

    # generator for loading the CNN features of the training videos
    print('Training data generator:')
    train_gen = VideoSegmentDataGenerator(training_data_path, input_file_mask, output_batch_size=1)
    gen = train_gen.generator()

    # determine class imbalance
    print('Checking for class imbalance')
    class_weights, _ = train_gen.get_class_weights()

    X, Y = [], []
    num_samples = min(num_train_samples, train_gen.number_of_batches())
    print('Total number of training samples', num_samples)
    for k in range(0, num_samples):
        x, y = next(gen)
        # flatten vectors
        x = x.flatten()         # remove time dimension
        y = y.flatten().tolist().index(1.0)     # convert from one-hot-encoding to numeric class label
        if len(x) == 20*4096 and y in range(0, 3):
            X.append(x)
            Y.append(y)
        if k % 100 == 0:
            print('read %d training samples of %d' % (k, num_samples))

    train_gen.stop()

    pca = IncrementalPCA(n_components=50, batch_size=500)
    if do_pca:
        print('Applying PCA...')
        X = pca.fit_transform(X)
        print(pca.explained_variance_ratio_)

    print('Training the SVM...')
    svmclf = svm.SVC(gamma='scale', decision_function_shape='ovo', class_weight={a: b for a, b in zip(range(0, 3), class_weights)})
    svmclf.fit(X, Y)
    print(svmclf)

    print('Saving trained SVM model to disk [%s]...' % output_model_file)
    f = open(output_model_file, 'wb')
    pickle.dump(svmclf, f)
    if do_pca:
        pickle.dump(pca, f)
    f.close()

    print('ready')


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--train", help="Path to the folder(s) containing the CNN data to be used for training. If more than one folder is available, then separate with a semi-colon.", default="")
    argparser.add_argument("--mask", help="The file mask to use for the CNN data files in the training folders", default="*.txt")
    argparser.add_argument("--model", help="File path and filename where to save the SVM model once it has been trained", default="")
    args = argparser.parse_args()

    if not args.train or not args.model:
        argparser.print_help()
        exit()

    t = pytictoc.TicToc()
    t.tic()

    train_SVM_model(training_data_path=args.train, input_file_mask=args.mask, output_model_file=args.model)

    t.toc('SVM Training')
    