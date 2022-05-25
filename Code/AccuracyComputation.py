from MeanAverageAccuracy import MeanAverageAccuracy

# "accuracy_washington" or "accuracy_notre_dame"
ACCURACY_PATH = "accuracy_notre_dame/rgb_out_sift_maa1.json"
ALSO_COMPUTE = True

# Create the mAA computation object
# "washington" or "notre_dame"
mAA = MeanAverageAccuracy(threshold=10,
						  frames_distance=10,
						  last_image=715,
						  merge_features=5000,
						  detect_method="ORB",
						  match_method="FLANN",
						  network_path="neuralnetwork/model/rgb_triplet_patch_encoder_model_euclidean_neg_outside_SIFT_circle.pt",
						  ransac_iterations=5000,
						  patch_side=8,
						  dataset="notre_dame",
						  method="rgb")

# If the accuracy must be computed, we compute it
if ALSO_COMPUTE:
	dnn_acc, cv_acc = mAA.compute_metric()
	mAA.save_to_file(ACCURACY_PATH, dnn_acc, cv_acc)

# Prints the accuracy
mAA.print_accuracy_computation(ACCURACY_PATH)
