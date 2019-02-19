"""
    This script can be used to download a list of YouTube videos.
    Must be given a file with the list of YouTube video URLs.
    Adds a slight delay in between downloads so as to follow the terms and conditions
    on call requests as specified by YouTube.
"""
import urllib.parse as urlparse
import pytube
import argparse
import time
import re
import os
import sys
import msvcrt   # Microsoft specific
import common


def download(video_urls_file, output_path, delay):
    # create the output folder if it does not exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # open the file with the video urls to get the list of YouTube videos
    with open(video_urls_file) as f:
        video_urls = f.readlines()
        num_videos = len(video_urls)
        print('Video URLs file has %d entries' % num_videos)

    print("\n\nWhile downloading the videos, press 'Q' to quit...\n\n")

    # download each video
    for ndx, video_url in enumerate(video_urls):
        downloaded = False
        print('Downloading %d of %d: %s' % (ndx+1, num_videos, video_url))
        try:
            # get video ID
            parsed = urlparse.urlparse(video_url)
            video_id = urlparse.parse_qs(parsed.query)['v']
            video_id = video_id[0].strip()

            # construct the output
            video_file = video_id     # extension will be added automatically based on the chosen video stream

            # has file been already downloaded?
            needs_download = True
            for f in os.listdir(output_path):
                if re.search(video_file + '\..*', f):     # need regular expression because we don't know the extension as of yet
                    needs_download = False

            # download the YouTube video
            if needs_download:
                # search for the video and set up the callback to run the progress indicator
                yt = pytube.YouTube(video_url, on_progress_callback=common.progress_fn, on_complete_callback=common.complete_fn)
                print(yt.streams.all())

                # get the first video type - usually the best quality
                video_type = yt.streams.filter(progressive=True) \
                    .order_by('resolution').desc().first()

                # download and save to disk
                common.file_size = video_type.filesize
                video_type.download(output_path=output_path, filename=video_file)
                downloaded = True
            else:
                print('already downloaded')

        except Exception as e:
            print("Exception: ", e)     # a regex exception probably means video is missing

        if msvcrt.kbhit():  # if a key is pressed
            key = msvcrt.getch()
            if key == b'q' or key == b'Q':
                print('User termination')
                return

        # YouTube's service agreement mentions a reasonable number of calls. We delay
        # a bit between each download so as not to fall foul of this condition.
        if downloaded:
            time.sleep(delay)       


if __name__ == "__main__":   
    argparser  =argparse.ArgumentParser()
    argparser.add_argument("--urls", help="Path to a text file with YouTube video urls (one per line)", default="")
    argparser.add_argument("--output", help="Path to the output folder where the video files will be saved", default="")
    argparser.add_argument("--delay", help="A time delay between each video download (in seconds)", default=5)
    args = argparser.parse_args()

    if not args.urls or not args.output:
        argparser.print_help()
        exit()

    download(video_urls_file=args.urls, output_path=args.output, delay=args.delay)
