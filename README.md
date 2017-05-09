Run scripts in the following order:

	(1) correct_data.py : Corrects the data, creating a new directory with the corrected data.
	(2) compute_trajectory.py : Fuses accelerometer and gyroscope streams to compute the trajectory.
	(3) segmentation.py : Finds candidate segments from each subject’s data.
	(4) lopo.py : Trains and evaluates classifier in a leave-one-participant-out fashion.
	(5) train_classifier.py : Trains and saves a classifier. Run when ready for prototyping, once the performance metrics are satisfactory.
	(6) real_time_prediction.py : Makes real-time predictions on incoming sensor data. This script uses the trajectory computation in compute_trajectory, the segmentation approach in segmentation.py and the classifier output by train_classifier. If you make any modification to the algorithms in those files, the real-time prediction will likely not perform as expected by the results reported in step (4).

	* Run show_segmentation_by_distance_to_rest_point.py -p=### at any time to visualize the segmentation output. ### refers to the patient ID.
	* Run perfect_segmentation.py instead of segmentation.py in order to obtain the ground-truth segments. This can be useful in determining how effective the classification is and whether the model parameters or feature extraction need tuning. Expect a higher performance with perfect segmentation. If it’s not higher, then most likely the segmentation approach is missing many positive instances (or it’s just really good, but unlikely…).
	* Run real_time_trajectory_visualization.py at any time to visualize the trajectory for incoming data.

There’s no promise you can run real_time_prediction.py and real_time_trajectory_visualization.py simultaneously. There is no need to, so don’t try it!

Other files include:

	* features.py : Used to compute features over candidate segments for classification.
	* util.py : Offers various utility functions. The only one that is applicable is the SlidingWindow function, which is currently not being used anywhere in the code.
	* quaternions.py : Provides methods for quaternion arithmetic. Note that the Madgwick fusion algorithm uses its own quaternion functions, but the functionality is essentially the same. The only method being used from quaternions.py is the qv_mult() function, which multiples a quaternion by a vector, specifically to obtain a position by multiplying a rotation quaternion by the starting position.
	* peakdetect.py : Used to detect peaks and troughs in a signal.
	* load_data.py : Used to load the raw (uncorrected) data from the original directory. So as not to change the original data directory, most scripts use the processing directory, i.e. /corrected_data/. Only the correct_data.py script calls the load_data.py script.
	* constants.py : Contains various constants.
	* client.py : Used to connect to the back-end to receive data in real-time. Used for real_time_prediction.py and real_time_trajectory_visualization.py.