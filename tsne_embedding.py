"""
    Support script to generate the t-SNE embeddings shown in the additional file.
    We use the t-SNE embeddings to analyse the CNN features extracted at the fc1 or 
    the fc2 layer of the VGG CNN.
"""
import os
import random
import glob
import argparse
import common
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def _read_files(file_list):
    print('reading %d files...' % len(file_list))
    data = []
    for (i, fp) in enumerate(file_list):
        #print('(%d) loading %s  ' % (i, fp))
        dt = np.load(fp)
        dt = np.array(dt['X'])
        dt = np.ndarray.flatten(dt)
        data.append(dt)  
    print('  read %d files' % len(file_list))
    print('  having shape ', data[0].shape)
    print()
    return data


def tsne_embedding(input_path, input_file_mask, groundtruth_file, max_pts_per_class, max_runs=1):
    # open and load the groundtruth data
    print('Loading groundtruth data...')
    gt = {}
    with open(groundtruth_file, 'r') as gt_file:
        gt_lines = gt_file.readlines()
    for gtl in gt_lines:
        gtf = gtl.rstrip().split(' ')
        if len(gtf) == 3:                   # our groundtruth file has 3 items per line (video ID, frame ID, class label)
            gt[(gtf[0], int(gtf[1]))] = gtf[2]
    print('ok\n')

    # get a list of all the data files...
    print('Traversing folder(s) for data files...')
    video_image_files = glob.glob(os.path.join(input_path, '**', input_file_mask), recursive=True)
    random.shuffle(video_image_files)

    # select a sample of data files per class
    files_S = []
    files_P = []
    files_n = []
    for (j, image_j) in enumerate(video_image_files):
        print('(%d) processing %s  ' % (j, image_j), end='')
       
        video_id = os.path.basename(os.path.dirname(image_j))
        frame_id = int(os.path.splitext(os.path.basename(image_j))[0])

        # groundtruth available?
        gt_label = '?'
        try:
            gt_label = gt[(video_id, frame_id)]
        except:
            pass
        print(gt_label)

        if gt_label == 'S':
            files_S.append(image_j)
        elif gt_label == 'P':
            files_P.append(image_j)
        elif gt_label == 'n':
            files_n.append(image_j)

        if len(files_S) > max_pts_per_class and len(files_P) > max_pts_per_class and len(files_n) > max_pts_per_class:
            print("Reached maximum number of data points allowed per class!")
            break
    print()

    # read the actual data from disk
    data_S = _read_files(files_S[0:max_pts_per_class])
    data_P = _read_files(files_P[0:max_pts_per_class])
    data_n = _read_files(files_n[0:max_pts_per_class])   

    # prepare the data
    X = np.vstack((data_S, data_P, data_n))
    y = np.concatenate([np.full(len(data_S), 'S'), 
                        np.full(len(data_P), 'P'), 
                        np.full(len(data_n), 'n')])
    data_S, data_P, data_n = None, None, None

    # dimensionality reduction with PCA
    print('PCA dimensionality reduction...')
    print('before PCA: {}'.format(X.shape))
    pca = PCA(n_components=min(1000, max_pts_per_class*3))
    pca_result = pca.fit_transform(X)
    print('after PCA: {}'.format(pca_result.shape))
    #print('explained variation by PCA components: {}'.format(pca.explained_variance_ratio_))
    print('cumulative explained variation by PCA components: {}\n'.format(np.sum(pca.explained_variance_ratio_)))
    X = pca_result

    # preparing data frame
    feature_cols = ['f' + str(i) for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_cols)
    df['label'] = pd.Categorical(y)
    print('Dataframe  shape {}\n'.format(df.shape))

    for run_i in range(max_runs):
        print('\n\n---- run {} of {} ----'.format(run_i+1, max_runs))

        # t-SNE
        tsne = TSNE(n_components=2, verbose=2, perplexity=40, n_iter=1000)
        tsne_results = tsne.fit_transform(df[feature_cols].values)
        #df['x_tsne'] = tsne_results[:, 0]
        #df['y_tsne'] = tsne_results[:, 1]
        
        print('final t-SNE: KL divergence is {} after {} iterations'.format(tsne.kl_divergence_, tsne.n_iter_))

        # plot t-SNE results
        color_map = {'S' : 'r', 'P' : 'g', 'n' : 'k'}
        pt_labels = {'S':'signing', 'P':'speaking', 'n':'other'}
        plt.scatter(x=tsne_results[:, 0], y=tsne_results[:, 1], alpha=0.5, s=15, c=df.label.map(color_map), edgecolors='none', label=df.label.map(pt_labels))
        plt.title('t-SNE plot of CNN features extracted from individual video frames (red=signing, green=speaking, black=other)')
        plt.show()

    print('Ready')


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--input", help="Path to the extracted CNN features", default="")
    argparser.add_argument("--mask", help="The file mask to use for the files containing the CNN features", default="*.npy")    
    argparser.add_argument("--gt", help="Path to the groundtruth file", default="")
    argparser.add_argument("--max-pts", help="Maximum number of data points per class", default=10000)
    argparser.add_argument("--runs", help="The number of t-SNE runs performed on the selected data (default is 1)", default=1)
    args = argparser.parse_args()

    if not args.input or not args.gt:
        argparser.print_help()
        exit()

    tsne_embedding(input_path=args.input, input_file_mask=args.mask, groundtruth_file=args.gt, max_pts_per_class=int(args.max_pts), max_runs=int(args.runs))

