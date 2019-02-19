"""
    This script extracts video frames from downloaded YouTube videos.
    Video frames are extracted at a rate of N frames every second and rescaled to the
    given size (224x224 by default).
"""
import os
import glob
import math
import cv2
import msvcrt   # Microsoft specific
import argparse


def extract_video_frames(input_path, output_path, resize_shape, output_fps, max_frames_per_video=999999, do_delete_processed_videos=False):
    """
        extracts video frames from a video at a rate of N frames every second (determined by output_fps).
        The video frames are rescaled to the specified frame size.
    """
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

                print('Extracting video frames from %s into %s...   (%dx%d, %f fps, %d frames)' % (file,
                    output_video_dir, int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                    orig_framerate, total_frames))

                frame_count = 0
                save_count = 0
                while video.isOpened():
                    #frameId = video.get(1)

                    if save_count > max_frames_per_video:
                        break

                    success, image = video.read()
                    if success is False:
                        break

                    if frame_count % read_step == 0:         # save every Nth frame
                        if image is not None:
                            image = cv2.resize(image, resize_shape, interpolation = cv2.INTER_AREA)
                        filename = os.path.join(output_video_dir, str(int(frame_count)) + ".jpg")
                        #print(filename)
                        cv2.imwrite(filename,image)
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
    argparser.add_argument("--output", help="Path to the output folder where the image frames will be extracted to", default="")
    argparser.add_argument("--fps", help="The rate at which frames will be extracted", default=5)
    argparser.add_argument("--max-frames", help="Maximum number of frames extracted for each individual video", default=2000)
    argparser.add_argument("--imwidth", help="Extracted frames wil be resized to this width (in pixels)", default=224)
    argparser.add_argument("--imheight", help="Extracted frames wil be resized to this height (in pixels)", default=224)
    argparser.add_argument("--del-videos", help="Delete each video once frames have been extracted from it", default=False)
    args = argparser.parse_args()

    if not args.input or not args.output:
        argparser.print_help()
        exit()

    extract_video_frames(input_path=args.input, output_path=args.output, output_fps=int(args.fps), 
            max_frames_per_video=int(args.max_frames), resize_shape=(int(args.imwidth), 
            int(args.imheight)), do_delete_processed_videos=args.del_videos)
