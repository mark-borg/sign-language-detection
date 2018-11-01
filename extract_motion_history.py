import os
import math
import time
import glob
import msvcrt   # Microsoft specific
import argparse
import functools
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool
import numpy as np
import cv2


def _extract_mhi(file_chunk, output_path, resize_shape, output_fps, num_stacked_frames, max_frames_per_video, do_delete_processed_videos):
    file_num = file_chunk[0]
    video_file = file_chunk[1]

    video = cv2.VideoCapture(video_file)

    output_video_dir = os.path.join(output_path, os.path.splitext(os.path.basename(video_file))[0])

    # compute the frame read step based on the video's fps and the output fps
    orig_framerate = video.get(cv2.CAP_PROP_FPS)
    total_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
    read_step = math.ceil(orig_framerate / output_fps)

    print('(%d) Extracting & processing video frames from %s into %s...   (%dx%d, %f fps, %d frames)' % (file_num, video_file,
        output_video_dir, int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        orig_framerate, total_frames))

    # read ahead so that we can generate the MHI in a temporal window centred around the sampled frame
    img_buffer = []
    read_ahead = math.floor(num_stacked_frames / 2)
    for k in range(read_ahead):
        _, img = video.read()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, resize_shape, interpolation = cv2.INTER_AREA)
        img_buffer.append(img)

    # create the output folder
    os.makedirs(output_video_dir)
    
    frame_count = 0
    save_count = 0
    while video.isOpened():
        #frameId = video.get(1)

        if save_count > max_frames_per_video:
            break

        # add the image to our buffer
        success, img = video.read()
        if success is False:
            break
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
        img = cv2.resize(img, resize_shape, interpolation = cv2.INTER_AREA)
        img_buffer.append(img)

        # keep only the last K images that have been read
        img_buffer = img_buffer[-num_stacked_frames:]

        if frame_count % read_step == 0:         # sample at every Nth frame position
            mhi = np.zeros(resize_shape, np.float32)

            for k in range(1, len(img_buffer)):
                fd = cv2.absdiff(img_buffer[k-1], img_buffer[k])
                _, fd = cv2.threshold(fd, 32, 1, cv2.THRESH_BINARY)
                cv2.motempl.updateMotionHistory(fd, mhi, k, num_stacked_frames)

            mhi = np.uint8(np.clip(mhi / num_stacked_frames, 0, 1) * 255)
            #cv2.imshow('MHI', mhi)
            #cv2.waitKey(1)

            # save MHI to disk
            cv2.imwrite(os.path.join(output_video_dir, str(int(frame_count)) + ".jpg"), mhi)

            save_count += 1
        frame_count += 1

    print('(%d)       ...saved %d frames' % (file_num, save_count))
    video.release()

    if do_delete_processed_videos:
        os.remove(video_file)


def extract_motion_history(input_path, output_path, resize_shape, output_fps, num_stacked_frames=2, max_frames_per_video=999999, 
        do_delete_processed_videos=False):
    """
        Given an input video, this function extracts a motion history image (MHI) of K consecutive frames, with K determined
        by parameter num_stacked_frames. The MHI operation is performed at a rate of N times per second (N determined by
        parameter output_fps). The video frames are rescaled to the specified frame size (resize_shape). At most, L MHI images
        are generated, with L given by parameter max_frames_per_video. Once processed, the video can be deleted, if prameter
        do_delete_processed_videos is set to True.
    """
    assert num_stacked_frames > 1, "Must be 2 or more."

    # create the output folder if it does not exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # input path must have a file mask
    if os.path.isdir(input_path):
        input_path = os.path.join(input_path, '*.*')

    # go through each input video
    listing = glob.glob(input_path)
    print('Processing %d video(s)...' % len(listing))

    print('creating video list...')
    videofiles_to_process = []
    file_count = 1
    for file in listing:
        if os.path.isfile(file):
            output_video_dir = os.path.join(output_path, os.path.splitext(os.path.basename(file))[0])

            # if we haven't already generated the MHI for this video...
            if not os.path.exists(output_video_dir):
                videofiles_to_process.append((file_count, file))

            file_count += 1

    # parallelise execution
    print('processing video list...')
    pool = Pool()

    process_fn = functools.partial(_extract_mhi, output_path=output_path, resize_shape=resize_shape, output_fps=output_fps,
        num_stacked_frames=num_stacked_frames, max_frames_per_video=max_frames_per_video, do_delete_processed_videos=do_delete_processed_videos)
    pool.map(process_fn, videofiles_to_process)


if __name__ == "__main__":
    argparser  =argparse.ArgumentParser()
    argparser.add_argument("--input", help="Path to the input folder containing the downloaded YouTube videos. Can contain a file mask.", default="")
    argparser.add_argument("--output", help="Path to the output folder where the output will be saved to", default="")
    argparser.add_argument("--fps", help="The rate at which frames will be extracted", default=5)
    argparser.add_argument("--K", help="Flow data for K consecutive frames are stacked together", default=2)
    argparser.add_argument("--max-frames", help="Maximum number of frames extracted for each individual video", default=2000)
    argparser.add_argument("--imwidth", help="Extracted frames wil be resized to this width (in pixels)", default=224)
    argparser.add_argument("--imheight", help="Extracted frames wil be resized to this height (in pixels)", default=224)
    argparser.add_argument("--del-videos", help="Delete each video once frames have been extracted from it", default=False)
    args = argparser.parse_args()

    if not args.input or not args.output:
        argparser.print_help()
        exit()

    extract_motion_history(input_path=args.input, output_path=args.output, output_fps=int(args.fps),
            num_stacked_frames=int(args.K), max_frames_per_video=int(args.max_frames),
            resize_shape=(int(args.imwidth), int(args.imheight)), do_delete_processed_videos=args.del_videos)
