#!/usr/bin/env python

"""
This Python script is given a search query term and it uses YouTube's API
to search for videos that are tagged with the given search term.
Videos belonging to YouTube Channels tagged with the given search term are
also included. But playlists are skipped, unless the playlists form part
of a channel. The video URLs and titles are saved to a text file.

Since YouTube's API is used, a developer key must be set up, as part of the
Google Cloud services account. Either set the value of the global variable
YOUTUBE_API_DEVELOPER_KEY with your key value, or else save your key to a
JSON file with name 'YOUTUBE_API_DEVELOPER_KEY.json' in the current folder.

When using the YouTube API, please ensure that you are familiar with YouTube's
terms of service agreement, especially the part about making frequent repeated
automated calls.
"""

from apiclient.discovery import build
from apiclient.errors import HttpError
from oauth2client.tools import argparser
import json


# YOUTUBE_API_DEVELOPER_KEY should be set to your API key value, taken from
# the APIs & auth > Registered apps tab of
#   https://cloud.google.com/console
# Please ensure that you have enabled the YouTube Data API for your project.
# If not set, then the methods will search for a JSON file in the current
# directory with the same name and load it from file
YOUTUBE_API_DEVELOPER_KEY = ""

_YOUTUBE_API_SERVICE_NAME = "youtube"
_YOUTUBE_API_VERSION = "v3"


# remove any non-ascii characters (unicode chars, special chars, etc.)
_remove_non_ascii = lambda string: "".join(letter for letter in string if ord(letter) < 128)


def search_youtube(options):
    global YOUTUBE_API_DEVELOPER_KEY, _YOUTUBE_API_SERVICE_NAME, _YOUTUBE_API_VERSION

    # ensure the YOUTUBE_API_DEVELOPER_KEY is not empty   
    try:
        if len(YOUTUBE_API_DEVELOPER_KEY) == 0:
            with open('YOUTUBE_API_DEVELOPER_KEY.json') as json_file:  
                json_data = json.load(json_file)
                YOUTUBE_API_DEVELOPER_KEY = json_data['YOUTUBE_API_DEVELOPER_KEY']
    except Exception as e:
        print('YOUTUBE_API_DEVELOPER_KEY has not been set!')
        print('Either assign it to the global variable YOUTUBE_API_DEVELOPER_KEY or else save it in a JSON file with name YOUTUBE_API_DEVELOPER_KEY.json in the current folder.')
        print(e)
        return

    # start the API session
    youtube = build(_YOUTUBE_API_SERVICE_NAME, _YOUTUBE_API_VERSION, developerKey=YOUTUBE_API_DEVELOPER_KEY)

    videos = []
    channels = []
    playlists = []

    f = open(options.output_file, mode = "wt" if options.overwrite == True else "at")
    page_token = ''
    page_num = 0

    while not page_token is None:
        try:
            page_num += 1
            print("Page #%d:" % page_num)

            # Call the search.list method to retrieve results matching the specified query term
            # Paging is used for the breaking the results into chunks, hence the pageToken parameter.
            search_response = youtube.search().list(
                q=options.q,
                part="id,snippet",
                maxResults=options.max_results,
                pageToken=page_token if page_num > 1 else None,
                ).execute()

            # Add each result to the list, and then display the lists of matching videos  (we ignore playlists and channels)
            for search_result in search_response.get("items", []):
                if search_result["id"]["kind"] == "youtube#video":
                    video_id = search_result["id"]["videoId"]
                    video_title = search_result["snippet"]["title"]
                    video_title = _remove_non_ascii(video_title.replace('\"', '\'').replace(',', '.'))
                    st = "https://www.youtube.com/watch?v=%s \"%s\"\n" % (video_id, video_title)
                    f.write(st)
                    videos.append(video_id)
                    print(st)
                elif search_result['id']['kind'] == 'youtube#channel':
                    channel_id = search_result['id']['channelId']
                    title = search_result["snippet"]["title"]
                    print('*** channel: %s (%s)\n' % (channel_id, title))
                    channels.append(channel_id)
                elif search_result['id']['kind'] == 'youtube#playlist':
                    playlist_id = search_result['id']['playlistId']
                    title = search_result["snippet"]["title"]
                    print('*** playlist: %s (%s)\n' % (playlist_id, title))
                    playlists.append(playlist_id)                  

            # next page available?
            page_token = search_response['nextPageToken'] if 'nextPageToken' in search_response else None
        except Exception as e:
            print("Exception: ", e)
            

    # now go through the channels (if any)
    for chn in channels:
        print('CHANNEL: ', chn)

        try:
            # get the channel's details (including its playlist)
            search_response = youtube.channels().list(
                        id=chn,
                        part="id,contentDetails",
                        maxResults=options.max_results,
                        ).execute()

            # for each playlist...
            for search_result in search_response.get("items", []):
                ply = search_result['contentDetails']['relatedPlaylists']['uploads']

                page_token = ''
                page_num = 0

                # go through the pages of video items for the current playlist... 
                while not page_token is None:
                    try:
                        page_num += 1
                        print("Page #%d:" % page_num)

                        # Get the video items making up the uploaded playlist of the current channel
                        search_response = youtube.playlistItems().list(
                            playlistId=ply,
                            part="id,snippet",
                            maxResults=options.max_results,
                            pageToken=page_token if page_num > 1 else None,
                            ).execute()

                        # Add each result to the list, and then display the lists of matching videos  (we ignore playlists and channels)
                        for search_result in search_response.get("items", []):
                            if search_result["snippet"]["resourceId"]["kind"] == "youtube#video":
                                video_title = search_result["snippet"]["title"]
                                video_title = _remove_non_ascii(video_title.replace('\"', '\'').replace(',', '.'))
                                video_id = search_result["snippet"]["resourceId"]["videoId"]
                                st = "https://www.youtube.com/watch?v=%s \"%s\"\n" % (video_id, video_title)
                                f.write(st)
                                videos.append(video_id)
                                print(st)
                        # next page available?
                        page_token = search_response['nextPageToken'] if 'nextPageToken' in search_response else None
                    except Exception as e:
                        print("Exception: ", e)

        except Exception as e:
            print("Exception: ", e)

    f.close()


if __name__ == "__main__":
    argparser.add_argument("--q", help="Search term", default="")
    argparser.add_argument("--max-results", help="Max results", default=50)
    argparser.add_argument("--output-file", help="Output file", default="")
    argparser.add_argument("--overwrite", help="Overwrite output file", default=True)
    args = argparser.parse_args()

    # has the user specified any arguments?
    if not args.output_file or not args.q:
        print('Usage:\n')
        print('    search_youtube.py --q="sign language" --output-file=search_results.txt --overwrite=T\n\n')
    else:
        try:
            search_youtube(args)
        except HttpError as e:
            print("An HTTP error %d occurred:\n%s" % (e.resp.status, e.content))
