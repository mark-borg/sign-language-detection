"""
    A simple tool to do basic groundtruthing of the video segments.
    This is quite primitive, and needs further rework, including the ability to
    correct errors during groundtruthing and move backwards.
"""
import os, glob
import cv2
import argparse
import common


def _help():
    print("Press 's' for signing")
    print("      'p' for speaking")
    print("      'n' for not signing or speaking")
    print("      '.' for unknown") 
    print()
    print("      '1' for marking the rest of the video frames with the last used label")
    print("      <ESC> to quit groundtruthing")
    print()


def groundtruthing(input_path, input_file_mask, groundtruth_file):

    # first check which videos have already been groundtruthed
    already_groundtruthed = {}
    gt = open(groundtruth_file, "r")
    existing_data = gt.readlines()
    gt.close()
    for d in existing_data:
        video_id = d.split(' ')[0]
        already_groundtruthed[video_id] = True
    print('Found %d videos with groundtruth data' % len(already_groundtruthed))


    cv2.namedWindow('t', cv2.WINDOW_NORMAL)
    cv2.namedWindow('t+1', cv2.WINDOW_NORMAL)
    cv2.namedWindow('t+2', cv2.WINDOW_NORMAL)
    cv2.namedWindow('t+3', cv2.WINDOW_NORMAL)
    cv2.namedWindow('t+20', cv2.WINDOW_NORMAL)

    font = cv2.FONT_HERSHEY_SIMPLEX

    # open the groundtruth file
    gt = open(groundtruth_file, "a", buffering=1)      # use line bufferring

    gt_lookup = {'s':'S', 'p':'P', 'n':'n', '.':'?', '1':'?'}

    # get all the video folders
    video_folders = os.listdir(input_path)
    video_folders.sort(key=common.natural_sort_key)

    for (i, video_i) in enumerate(video_folders):
        print('processing video %d of %d:  %s\n' % (i, len(video_folders), video_i))   

        if video_i in already_groundtruthed:
            print('already groundtruthed!')
            continue

        # for each video, get the list of its extracted frames
        video_i_images = glob.glob(os.path.join(input_path, video_i, input_file_mask))
        video_i_images.sort(key=common.natural_sort_key)       # ensure images are in the correct order to preserve temporal sequence 
        assert(len(video_i_images) > 0), "video %s has no frames!!!" % video_i

        # preload all the images
        imgs = []
        for image_j in video_i_images:
            img = cv2.imread(image_j)
            cv2.imshow('t', img)
            ky = cv2.waitKey(1)
            imgs.append(img)

        # wait for a key press
        print('%d images loaded\nPress any key to start groundtruthing...' % len(imgs))
        ky = cv2.waitKey(-1)
        _help()

        # now groundtruth the video frames...
        gt_label, prev_gt_label = '?', '?'
        ask_user = True
        for (n, img) in enumerate(imgs):
            image_id = os.path.basename(video_i_images[n])

            cv2.putText(img, '%d' % n, (8,12), font, 0.5, (255,255,255), 1)

            cv2.imshow('t', img)
            try:
                cv2.imshow('t+1', imgs[n+1])
                cv2.imshow('t+2', imgs[n+2])
                cv2.imshow('t+3', imgs[n+3])
                cv2.imshow('t+20', imgs[n+20])
            except:
                pass
            cv2.waitKey(1)

            if ask_user:
                ok = False
                while ok == False:
                    ky = cv2.waitKey(-1)
                    if ky == 27:
                        return
                    try:
                        ky = chr(ky)
                        prev_gt_label = gt_label
                        gt_label = gt_lookup[ky]
                        ok = True
                    except:
                        print('unknown key (%d)\n' % int(ky))
                        _help()

            gt.write('%s %s %s\n' % (video_i, os.path.splitext(image_id)[0], gt_label))

            if ky == '1':
                gt_label = prev_gt_label
                print("Continuing till the end with label '%s'..." % gt_label)
                ask_user = False
                ky = ''

        print('video ready.\nPress any key to process the next video...')
        cv2.waitKey(-1)

    gt.close()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    argparser  =argparse.ArgumentParser()
    argparser.add_argument("--input", help="Path to the input parent folder containing the video frames of the downloaded YouTube videos", default="")
    argparser.add_argument("--mask", help="The file mask to use for the video frames of the downloaded YouTube videos", default="*.jpg")      
    argparser.add_argument("--output", help="Path to the groundtruth output file", default="")
    args = argparser.parse_args()

    if not args.input or not args.output:
        argparser.print_help()
        exit()

    groundtruthing(input_path=args.input, input_file_mask=args.mask, groundtruth_file=args.output)
