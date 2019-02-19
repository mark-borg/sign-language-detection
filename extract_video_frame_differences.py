"""
    This script extracts video frames from downloaded YouTube videos, and computes
    frame differencing on the extracted video frames.
    Video frames are extracted at a rate of N frames every second and rescaled to the
    given size (224x224 by default). The temporal distance between 2 difference video 
    frames can be specified by the user.
"""
import os
import glob
import math
import cv2
import msvcrt   # Microsoft specific
import argparse


def extract_video_frame_differences(input_path, output_path, resize_shape, output_fps, K_frame_differencing=2, max_frames_per_video=999999, 
        grayscale=False, do_delete_processed_videos=False):
    """
        Given an input video, this function extracts 'motion summary images' at a rate of N frames every second (determined by output_fps).
        In this particular case, a motion summary image consists of frame differences of K consecutive frames.
        The video frames are rescaled to the specified frame size.
    """
    assert K_frame_differencing > 1, "Must be 2-frame-differencing or more."

    # create the output folder if it does not exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # input path must have a file mask
    if os.path.isdir(input_path):
        input_path = os.path.join(input_path, '*.*')

    # go through each input video
    listing = glob.glob(input_path)
    print('Processing %d video(s)...' % len(listing))

    file_count = 1
    for file in listing:
        if os.path.isfile(file):
            video = cv2.VideoCapture(file)

            # compute the frame read step based on the video's fps and the output fps
            orig_framerate = video.get(cv2.CAP_PROP_FPS)
            total_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
            read_step = math.ceil(orig_framerate / output_fps)

            output_video_dir = os.path.join(output_path, os.path.splitext(os.path.basename(file))[0])

            # extract video frames if not already done
            if not os.path.exists(output_video_dir):
                os.makedirs(output_video_dir)

                print('Extracting & processing video frames from %s into %s...   (%dx%d, %f fps, %d frames)' % (file,
                    output_video_dir, int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                    orig_framerate, total_frames))

                # read ahead so that we can perform the differencing operation centred around the sampled frame
                img_buffer = []
                read_ahead = math.floor(K_frame_differencing / 2)
                for k in range(read_ahead):
                    _, img = video.read()
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if grayscale else img
                    img_buffer.append(img)

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
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if grayscale else img
                    img_buffer.append(img)

                    # keep only the last K images that have been read
                    img_buffer = img_buffer[-K_frame_differencing:]

                    if frame_count % read_step == 0:         # sample at every Nth frame position
                        mfd = None
                        for k in range(1, len(img_buffer)):
                            fd = cv2.absdiff(img_buffer[k-1], img_buffer[k])    # compute 2-frame-diff
                            mfd = fd if mfd is None else cv2.add(mfd, fd)       # stack differenced images together to get multi-frame-differences
                        mfd = cv2.resize(mfd, resize_shape, interpolation = cv2.INTER_AREA)
                        filename = os.path.join(output_video_dir, str(int(frame_count)) + ".jpg")
                        #print(filename)
                        cv2.imwrite(filename, mfd)
                        save_count += 1
                    frame_count += 1

                print('      ...saved %d frames' % save_count)
                video.release()
                print('done')

                if do_delete_processed_videos:
                    os.remove(file)

                if msvcrt.kbhit():  # if a key is pressed 
                    key = msvcrt.getch()
                    if key == b'q' or key == b'Q':
                        print('User termination')
                        return

                file_count += 1


if __name__ == "__main__":
    argparser  =argparse.ArgumentParser()
    argparser.add_argument("--input", help="Path to the input folder containing the downloaded YouTube videos. Can contain a file mask.", default="")
    argparser.add_argument("--output", help="Path to the output folder where the output will be saved to", default="")
    argparser.add_argument("--fps", help="The rate at which frames will be extracted", default=5)
    argparser.add_argument("--diff", help="The amount of consecutive frames to use when calculting the frame differencing. Can be 2-frame differencing or multi-frame differencing.", default=2)
    argparser.add_argument("--max-frames", help="Maximum number of frames extracted for each individual video", default=2000)
    argparser.add_argument("--imwidth", help="Extracted frames wil be resized to this width (in pixels)", default=224)
    argparser.add_argument("--imheight", help="Extracted frames wil be resized to this height (in pixels)", default=224)
    argparser.add_argument("--gray", help="produce grayscale output", default=False)
    argparser.add_argument("--del-videos", help="Delete each video once frames have been extracted from it", default=False)
    args = argparser.parse_args()

    if not args.input or not args.output:
        argparser.print_help()
        exit()

    extract_video_frame_differences(input_path=args.input, output_path=args.output, output_fps=int(args.fps), 
            K_frame_differencing=int(args.diff), max_frames_per_video=int(args.max_frames), grayscale=args.gray,
            resize_shape=(int(args.imwidth), int(args.imheight)), do_delete_processed_videos=args.del_videos)
