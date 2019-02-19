"""
    This script can be used to download a single YouTube video, given its URL.
"""
import urllib.parse as urlparse
import pytube
import argparse
import re
import os
import common


def download(video_url, output_path):
    # create the output folder if it does not exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    print('Downloading %s...' % video_url)
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
            if re.search(video_file + '\\..*', f):     # need regular expression because we don't know the extension as of yet
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
            global file_size
            file_size = video_type.filesize
            video_type.download(output_path=output_path, filename=video_file)
        else:
            print('already downloaded')

    except Exception as e:
        print("Exception: ", e)     # a regex exception probably means video is missing


if __name__ == "__main__":   
    argparser  =argparse.ArgumentParser()
    argparser.add_argument("--url", help="The YouTube video url to download", default="")
    argparser.add_argument("--output", help="Path to the output folder where the video file will be saved", default="")
    args = argparser.parse_args()

    if not args.url or not args.output:
        argparser.print_help()
        exit()

    download(args.url, args.output)
