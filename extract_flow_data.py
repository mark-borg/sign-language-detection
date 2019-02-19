"""
    This script extracts video frames from downloaded YouTube videos, and computes
    optical fl on the extracted video frames.
    Video frames are extracted at a rate of N frames every second and rescaled to the
    given size (224x224 by default). 
"""
import os
import numpy as np
import glob
import math
import msvcrt   # Microsoft specific
import argparse
import cv2


def extract_flow_data(input_path, output_path, resize_shape, output_fps, num_stacked_frames=2, max_frames_per_video=999999, 
        do_delete_processed_videos=False, save_images=False):
    """
        Given an input video, this function extracts flow data at a rate of N frames every second (determined by output_fps).
        Optical flow is extracted, with flow data of K consecutive frames stacked together, where K is specified by parameter stacked_frames.
        The video frames are rescaled to the specified frame size.
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

    file_count = 1
    for file in listing:
        if os.path.isfile(file):
            output_video_dir = os.path.join(output_path, os.path.splitext(os.path.basename(file))[0])

            # if we haven't already generated flow data for this video...
            if not os.path.exists(output_video_dir):
                video = cv2.VideoCapture(file)

                # compute the frame read step based on the video's fps and the output fps
                orig_framerate = video.get(cv2.CAP_PROP_FPS)
                total_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
                read_step = math.ceil(orig_framerate / output_fps)

                print('(%d) Extracting & processing video frames from %s into %s...   (%dx%d, %f fps, %d frames)' % (file_count, file,
                    output_video_dir, int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                    orig_framerate, total_frames))

                # read ahead so that we can perform the stacking of the flow data in a temporal window centred around the sampled frame
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
                        #stacked = None
                        stacked = []
                        base_name = os.path.join(output_video_dir, str(int(frame_count)))

                        for k in range(0, len(img_buffer)):
                            flow = cv2.calcOpticalFlowFarneback(img_buffer[k-1], img_buffer[k], None, 0.5, 3, 15, 3, 5, 1.2, 0)   # compute dense optical flow                 
                            # save each individual raw flow output
                            #cv2.optflow.writeOpticalFlow(base_name + ".flo", flow)
                                                
                            # discretise the flow data
                            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                            hsv = np.zeros(resize_shape + (3,), np.uint8)
                            hsv[..., 0] = ang * 180 / np.pi / 2
                            hsv[..., 1] = 255    # saturation
                            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
                            rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

                            if k == 0:
                                cv2.imshow('flow', rgb)
                                cv2.waitKey(1)
                                if save_images:
                                    # save visualisation of each individual flow output
                                    cv2.imwrite(base_name + ".jpg", rgb)

                            # stack flow data together
                            #stacked = flow if stacked is None else np.concatenate((stacked, flow), axis=2)     # This takes too much space: 3MB/frame!
                            
                            # compress the channels using JPEG and stack the image together as an array
                            _, enc_rgb = cv2.imencode('.jpg', rgb, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                            stacked.append(enc_rgb)

                        # save stacked flow data to disk
                        #np.savez_compressed(base_name + ".npz", stacked)
                        np.save(base_name + ".npy", stacked)

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
    argparser.add_argument("--K", help="Flow data for K consecutive frames are stacked together", default=2)
    argparser.add_argument("--max-frames", help="Maximum number of frames extracted for each individual video", default=2000)
    argparser.add_argument("--imwidth", help="Extracted frames wil be resized to this width (in pixels)", default=224)
    argparser.add_argument("--imheight", help="Extracted frames wil be resized to this height (in pixels)", default=224)
    argparser.add_argument("--del-videos", help="Delete each video once frames have been extracted from it", default=False)
    argparser.add_argument("--save-images", help="Save the visualisations of optical flow as JPEG images", default=False)
    args = argparser.parse_args()

    if not args.input or not args.output:
        argparser.print_help()
        exit()

    extract_flow_data(input_path=args.input, output_path=args.output, output_fps=int(args.fps),
            num_stacked_frames=int(args.K), max_frames_per_video=int(args.max_frames),
            resize_shape=(int(args.imwidth), int(args.imheight)), do_delete_processed_videos=args.del_videos, save_images=args.save_images)
