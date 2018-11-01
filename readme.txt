# sign-language-detection

This repository contains code for the paper "Sign Language Detection 'In the Wild' with Recurrent Neural Networks"


Requirements:
  TensorFlow,
  Keras,
  Python,
  Python wrapper for OpenCV,
  and some additional python libraries
  

The code is quite modular and consists of a number of separate programs that can be run from the command prompt.
Each program writes its results to disk, and these in turn can serve as input to the next program in the pipeline.
Since in our paper we try out many different neural network architectures and different types of motion data, the
order of how programs are called and the "pipeline" created from a sequence of calls to such program can vary a lot.

DATASET:

The code requires a dataset like the one we provide here:
https://github.com/mark-borg/Signing-in-the-Wild-dataset.

This dataset was created using the following pipeline:
- searching for YouTube videos via the script search_youtube.py,
- then, downloading the videos via the script download_youtube_video_list.py

Program search_youtube.py provides the functionality to search for YouTube videos using keywords, such as "sign language", "ASL", "signing", "deaf", etc.
Usage is as follows:
		python.exe  search_youtube.py  --q="sign language"  --output-file=candidate_video_urls.txt
		
Downloading of the videos is done via a call to:
		python.exe  download_youtube_video_list.py  --urls="vetted_video_urls.txt" --output=\sld\videos
		
The download script (like many of the other scripts) can be run multiple times, and it will only download videos that have not been downloaded before. 
So one could run download_youtube_video_list.py and terminate it half way (by pressing 'q' to quit the process). Then one can restart the script and it
will continue from the last position it was stopped on.

Rather than using download_youtube_video_list.py, one could also make use of the 3rd party program youtube-dl.exe available from: http://rg3.github.io/youtube-dl.


EXTRACTING FRAMES AND MOTION DATA:

Once the videos are downloaded, then raw video frames and motion data can be extracted via a call to the following scripts:
- extract_video_frames.py : extracts video frames at a specified output frame rate (e.g. 5Hz) and resizing the video frames to a given size
- extract_video_frame_differences.py : computes multi-frame differecing at a specific output frame rate
- extract_flow_data.py : extracts stacked optical flow data of K consecutive frames at a specific output frame rate
- extract_motion_history.py : extracts the motion history image (MHI) of K consecutive frames at a specific output frame rate

Usage is as follows:

	python extract_video_frames.py --input=\sld\videos --output=\sld\frames  --max-frames=2000  --fps=5
	
	python extract_video_frame_differences.py --input=\sld\videos  --output=\sld\diff  --max-frames=2000  --fps=5  --diff=10  --gray=True
	
	python extract_flow_data.py --input=\sld\videos --output=\sld\flow  --max-frames=2000  --fps=5  --K=10
	
	extract_motion_history.py --input=\sld\videos --output=\sld\mhi  --max-frames=2000  --fps=5  --K=10
	
For all of the above scripts, one can also specify the output frame size. By default this is set to 224x224 pixels.
Parameter K specifies the stack size.

As one can see from the examples above, we have adopted the procedure of saving the modified input into separate folders. Example, while the original
videos are ins subfolder 'sld\videos', the extracted video frames will be saved into sub-folder 'sld\frames'. And we use separate folders for the
motion data. The folder names can be chosen to be different. For the remaining steps, we follow the same principle: whenever we change the input, we
store the results in a separate folder.


EXTRACTING CNN FEATURES:

For extracting CNN features using the VGG-16 network, two scripts are available:
- generate_CNN_features.py
- generate_CNN_features_from_flow_data.py

The first one can be used for raw video frames, MHI images, and multi-frame difference images. 
The second is used for optical flow data. The reason is because stacked optical flow data consists of an "image" with K*3 channels and we need to replicate
the first convolutional layer of VGG-16 to handle such an input with more than 3 channels (more details in the paper).

Usage is as follows:

	python generate_CNN_features.py --input=\sld\frames --output=\sld\frames_cnnfc1 --groundtruth=\sld\groundtruth.txt  --fc1_layer=True
	
	python generate_CNN_features.py --input=\sld\diff  --output=\sld\diff_cnnfc1  --groundtruth=\sld\groundtruth.txt  --fc1_layer=True
	
	python generate_CNN_features.py --input=\sld\mhi  --output=\sld\mhi_cnnfc1  --groundtruth=\sld\groundtruth.txt  --fc1_layer=True
	
	python generate_CNN_features_from_flow_data.py --input=\sld\flow  --output=\sld\flow_cnnfc1  --groundtruth=\sld\groundtruth.txt  --fc1_layer=True  --K=10
	
Note that the above scripts have a parameter called fc1_layer. This is set to True if you want to extract the features from the first dense layer (fc1) of VGG-16.
If parameter fc1_layer is set to False, then CNN features from the last convolutional layer of VGG-16 will be extracted instead (see the paper for details).

The groundtruth file is specified (if available) so that no CNN features are extracted for unlabelled video frames. If not specified, then CNN features are extracted
for all the video frames.

As mentioned earlier, you can stop the scripts (by pressing 'q') and then restart the script later; it will continue from the point where it was stopped.

No fine-tuning was performed for the VGG-16 CNN.


CUTTING THE VIDEO SEGMENTS:

Once the raw image data and the motion data has been extracted from the videos, the next step is to "cut" the videos into chunks (video segments/video clips) of a 
given size. The script cut_into_video_segments.py can be used for this operation.

Usage is as follows:

	python cut_into_video_segments.py  --input=\sld\frames_cnnfc1  --mask=*.npy  --len=20  --output=\sld\frames_cnnfc1_seg20  --gt=\sld\groundtruth.txt
	
	python cut_into_video_segments.py  --input=\sld\diff_cnnfc1  --mask=*.npy  --len=20  --output=\sld\diff_cnnfc1_seg20  --gt=\sld\groundtruth.txt
	
	python cut_into_video_segments.py  --input=\sld\mhi_cnnfc1 --mask=*.npy  --len=20  --output=\sld\mhi_cnnfc1_seg20  --gt=\sld\groundtruth.txt
	
	python cut_into_video_segments.py  --input=\sld\flow_cnnfc1 --mask=*.npy  --len=20  --output=\sld\flow_cnnfc1_seg20  --gt=\sld\groundtruth.txt
	
The groundtruth file is required here, so that only video segments that have a consistent label are extracted, i.e., a video segment were all its video frames
are labelled with the same class label, e.g., 'signing'. This requirement is there to assist the training process.

Parameter len determines the length of the video segments. For example, assume that video frames were extracted from the original videos at 5Hz (parameter fps 
of extract_video_frames.py). Then setting len to 20, means 20/5Hz = 4-second video segments (video clips)


PARTITIONING INTO FOLDS:

Once we have the video segments, we need to partition them into folds for the purposes of training. Script prepare_validation_folds.py does this.

Usage is as follows:

	python prepare_validation_folds.py  --input=\sld\frames_cnnfc1_seg20  --output=\sld\frames_cnnfc1_seg20_folds  --folds=5
	
	python prepare_validation_folds.py  --input=\sld\diff_cnnfc1_seg20  --output=\sld\diff_cnnfc1_seg20_folds  --folds=5
	
	python prepare_validation_folds.py  --input=\sld\flow_cnnfc1_seg20  --output=\sld\flow_cnnfc1_seg20_folds  --folds=5
	
For example, in the above examples we use 5 folds. These could then be used for 5-fold cross validation, or simply using folds 1 to 4 for
training and the 5th fold as a hold-out validation set.


TRAINING THE NETWORK:

Training the RNN network is done via the train_RNN_model.py script.

Usage is as follows:

	python train_RNN_model.py  --train=/sld/frames_cnnfc1_seg20_folds/1;/sld/frames_cnnfc1_seg20_folds/2;/sld/frames_cnnfc1_seg20_folds/3;/sld/frames_cnnfc1_seg20_folds/4  --validate=/sld/frames_cnnfc1_seg20_folds/5  --model=/sld/rnn.h5  --timesteps=20  --fc1_layer=True  --batch=32  --lr=0.001
	
Note that as a training fodler, you can pass a single folder or a set of folders. For the latter case, it must be a single string and the folders must be separated with 
semi-colons. In the above example, we train on folds 1 to 4, hence the four folders passed to parameter 'train'.  Parameter 'validate' specifies the folder with the 
videos of the validation fold.


TESTING THE NETWORK:

To test the trained model against a fold, use:

	python test_RNN_model.py  --test=/sld/frames_cnnfc1_seg20_folds/5  --model=/sld/rnn.h5  --results=/sld/results.txt   --timesteps=20  --fc1_layer=True  --batch=1024

This uses the video segments created earlier. Each video segment has a consistent label and the matching is checked at video segment level.

If you want to test a full video via a sliding window approach (regardless of whether the video segment has a consistent segment or not), then use the script:
test_one_file.py. This does checking of the predicted class label at frame level. Usage is as follows:

	python test_one_file.py  --videos=/sld/frames_cnnfc1  --video_id=HUMEcnkvhJU  --gt=\sld\groundtruth.txt  --timesteps=20  --fc1_layer=True  --model=/sld/rnn.h5  --output=/sld
	
	

------
	
For any queries, please send an email to:    mborg2005@gmail.com

