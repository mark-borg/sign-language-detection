"""
    This Python script is used to test the SVM classifier created by train_CNN_SVM_model.py.
    The SVM takes as input the CNN features extracted using one of the scripts generate_CNN_features*.py.
"""
import argparse
import pickle
import pytictoc
from sklearn import svm
from sklearn.decomposition import IncrementalPCA
from sklearn.metrics import confusion_matrix, classification_report
from generators import VideoSegmentDataGenerator


def test_SVM_model(validation_data_path, input_file_mask, svm_model_file, num_validation_samples=8000, do_pca=False):

    print('Loading the trained SVM model from disk [%s]...' % svm_model_file)
    f = open(svm_model_file, 'rb')
    svmclf = pickle.load(f)
    pca = None
    if do_pca:
        pca = pickle.load(f)
    f.close()

    # generator for loading the CNN features of the validation fold
    print('Validation data generator:')
    validation_gen = VideoSegmentDataGenerator(validation_data_path, input_file_mask, output_batch_size=1)
    class_names = validation_gen.get_class_names()
    gen = validation_gen.generator()

    X, Y = [], []
    num_samples = min(num_validation_samples, validation_gen.number_of_batches())
    print('Total number of validation samples', num_samples)
    for k in range(0, num_samples):
        x, y = next(gen)
        X.append(x.flatten())
        Y.append(y.flatten().tolist().index(1.0))
        if k % 100 == 0:
            print('read %d validation samples of %d' % (k, num_samples))

    validation_gen.stop()

    if do_pca:
        print('Applying PCA...')
        X = pca.transform(X)

    print('Evaluating the SVM...')
    Ypred = svmclf.predict(X)

    correct = 0
    for i in range(0, len(Y)):
        if Ypred[i] == Y[i]:
            correct += 1

    # display performance results
    print("correct predictions %d of %d (%f percent)" % (correct, len(Y), 100.0 * correct / len(Y)))
    print(confusion_matrix(Y, Ypred))
    print(classification_report(Y, Ypred, target_names=class_names))

    print('ready')


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--validate", help="Path to the folder(s) containing the CNN data to be used for validation. If more than one folder is available, then separate with a semi-colon.", default="")
    argparser.add_argument("--mask", help="The file mask to use for the CNN data files in the validation folders", default="*.txt")
    argparser.add_argument("--model", help="File path and filename from where to load the trained SVM model", default="")
    args = argparser.parse_args()

    if not args.validate or not args.model:
        argparser.print_help()
        exit()

    t = pytictoc.TicToc()
    t.tic()

    test_SVM_model(validation_data_path=args.validate, input_file_mask=args.mask, svm_model_file=args.model)

    t.toc('SVM Evaluation')
    