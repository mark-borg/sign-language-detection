import os, glob
import math
import random
import threading
import queue
import pytictoc
import numpy as np
import common
from sklearn.utils import class_weight


class VideoSegmentReader(threading.Thread):
    def __init__(self, batch_queue, event, segments, output_batch_size, classes, num_classes, do_shuffle, return_sample_id, do_timings=False):
        threading.Thread.__init__(self)
        self.queue = batch_queue
        self.stopped = event
        self.segments = segments
        self.output_batch_size = output_batch_size
        self.classes = classes
        self.num_classes = num_classes
        self.do_shuffle = do_shuffle
        self.return_sample_id = return_sample_id
        self.do_timings = do_timings

    def run(self):
        sample_no = 0

        X_batch = []
        Y_batch = np.zeros((self.output_batch_size, self.num_classes))
        id_batch = []

        t = pytictoc.TicToc()
        if self.do_timings:
            t.tic()

        while True:
            for seg in self.segments:
                # should the thread terminate?
                if self.stopped.is_set():
                    print('VideoSegmentReader thread exiting')
                    return

                # load the data for this video segment
                Y = self.classes.index(seg[1])          # transform from a class label to a class index
                idb = seg[2] if self.return_sample_id else []
                X = []
                for pth in seg[0]:
                    dt = np.load(pth)
                    dt = np.array(dt['X'])          # has shape (1, width, height, channels)
                    X.append(dt[0, ...])
                X = np.array(X)
                sample_no += 1

                # append to our batch          
                X_batch.append(X)
                Y_batch[sample_no-1, Y] = 1         # one-hot encoding of the class label
                id_batch.append(idb)

                assert sum(sum(Y_batch)) == sample_no, "Class labels for current batch are not properly one-hot-encoded!"

                # do we have a complete batch?
                if sample_no == self.output_batch_size:
                    try:
                        X_batch = np.array(X_batch)
                        if self.do_timings:
                            t.toc('batch construction')
                        
                        placed_on_queue = False
                        while not placed_on_queue:
                            try:
                                self.queue.put((X_batch, Y_batch, id_batch) if self.return_sample_id else (X_batch, Y_batch), block=True, timeout=10)
                                placed_on_queue = True
                            except queue.Full:
                                placed_on_queue = False

                            if self.stopped.is_set():
                                print('VideoSegmentReader thread exiting')
                                return
                    except:
                        print('ERROR ENCOUNTERED WITH BATCH')
                        pass    # something wrong with this batch. Skip!

                    # reset
                    X_batch, Y_batch, id_batch = [], np.zeros((self.output_batch_size, self.num_classes)), []
                    sample_no = 0
                    if self.do_timings:
                        t.tic()

            if self.do_shuffle:
                # shuffle each time we do a full iteration through the dataset
                random.shuffle(self.segments)


class VideoSegmentDataGenerator:
    """
        This class manages a Python generator for loading data for video segments, e.g., CNN feature data.
        Call the method generator() to get the actual Python generator object.

        Example:
            mygenmgr = VideoSegmentDataGenerator('parent', '*.*', 10)
            gen = mygenmgr.generator()

            (a,b,c) = next(gen)

            for (a,b,c) in gen:
                do_processing_on(a,b,c)
    """

    def __init__(self, parent_folder, file_mask, output_batch_size, do_shuffle=True, return_sample_id=False, do_timings=False):
        self.parent_folder = parent_folder
        self.file_mask = file_mask
        self.output_batch_size = output_batch_size
        self.do_shuffle = do_shuffle
        self.return_sample_id = return_sample_id
        self.do_timings = do_timings

        self.segments = []
        self.classes = []    # this list is used to keep track of the classes and to assign them a class number (list index)

        # we might have multiple folders specified for the parent_folder, e.g. if using K-fold validation and multiple folds are used for training.
        # multiple folders should be separated with a semi-colon (;)
        self.parent_folder = self.parent_folder.split(';')

        # get all the video segments
        print('Searching for samples...')
        seg_files = []
        for parent_folder_i in self.parent_folder:
            pth = os.path.join(parent_folder_i, file_mask if len(file_mask) > 0 else '*.*')
            seg_files.extend(glob.glob(pth))
        seg_files.sort(key=common.natural_sort_key)
        print('Found %d video segments in total...' % len(seg_files))

        # for each video segment
        for seg_j in seg_files:
            # read its data...
            with open(seg_j, 'r') as f:
                data_lines = f.readlines()
                seg_j_data = []
                class_label = None
                for dt in data_lines:
                    k = dt.strip().split(' ')
                    assert class_label is None or class_label == k[1], 'Video segment has inconsistent class labelling'
                    class_label = k[1]
                    seg_j_data.append(k[0])
                
                if class_label != '?':      # ignore this special case
                    if not class_label in self.classes:
                        self.classes.append(class_label)

                    # save the video segment data (video frame paths), its class label, and video segment file location
                    self.segments.append((seg_j_data, class_label, seg_j))

        self.num_classes = len(self.classes)
        self.classes.sort()         # we must sort the class labels to ensure that the numbering of the classes done by the training and the validation generator are the same
        print('Found %d samples belonging to %d classes %s in folder %s' % (len(self.segments), self.num_classes, self.classes, self.parent_folder))
        assert self.num_classes > 0 and len(self.segments) > 0, 'No data found!!'

        if do_shuffle:
            # do an initial shuffle
            random.shuffle(self.segments)

        print('Starting VideoSegmentReader thread(s)...')
        self.stop_event = threading.Event()
        self.queue = queue.Queue(100)
        self.reader_thread1 = VideoSegmentReader(self.queue, self.stop_event, self.segments[:len(self.segments)//3], self.output_batch_size,
                                    self.classes, self.num_classes, self.do_shuffle, self.return_sample_id, self.do_timings)
        self.reader_thread1.start()
        self.reader_thread2 = VideoSegmentReader(self.queue, self.stop_event, self.segments[len(self.segments)//3:2*len(self.segments)//3], self.output_batch_size,
                                    self.classes, self.num_classes, self.do_shuffle, self.return_sample_id, self.do_timings)
        self.reader_thread2.start()
        self.reader_thread3 = VideoSegmentReader(self.queue, self.stop_event, self.segments[2*len(self.segments)//3:], self.output_batch_size,
                                    self.classes, self.num_classes, self.do_shuffle, self.return_sample_id, self.do_timings)
        self.reader_thread3.start()

    def __del__(self):
        self.stop()

    def stop(self):
        print('Signaling VideoSegmentReader thread(s) to stop...')
        self.stop_event.set()
        print('Waiting for VideoSegmentReader thread(s) to stop...')
        self.reader_thread1.join(10)
        self.reader_thread2.join(10)
        self.reader_thread3.join(10)
        print('VideoSegmentDataGenerator exiting')

    def __len__(self):
        return len(self.segments)

    def number_of_batches(self):
        return math.ceil(len(self.segments) / self.output_batch_size)

    # the generator method
    def generator(self):
        while True:
            batch = self.queue.get(block=True, timeout=None)
            yield batch

    def get_class_weights(self):
        # get all the class labels
        y = []
        for seg in self.segments:
            y.append(seg[1])
        y = np.array(y)

        # compute class totals
        _, ct = np.unique(y, return_counts=True)

        # compute the class weights
        cw = class_weight.compute_class_weight('balanced', self.classes, y)
        print('class weights:', cw, 'totals:', ct, 'class labels:', self.classes)

        return (cw, ct)

    def get_class_names(self):
        return self.classes


if __name__ == "__main__":
    gen_mgr = VideoSegmentDataGenerator(parent_folder='E:\\sld\\frames_cnnfc1_seg20_folds\\1', file_mask='*.txt', output_batch_size=32)
    gen = gen_mgr.generator()

    cw = gen_mgr.get_class_weights()
    print(cw)

    (a, b) = next(gen)
