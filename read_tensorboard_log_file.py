
from tensorboard.backend.event_processing import event_accumulator

# Path to the directory containing the TensorBoard log files
log_dir = "/usr/mvl2/esdft/3d-segmentation-monai-main_main/ckpt_lesions_all_training/best_bonbid_unet/lightning_logs/version_5/"

# Initialize the event accumulator
event_acc = event_accumulator.EventAccumulator(log_dir)

# Load the event files
event_acc.Reload()

# Print available tags (summary keys)
print("Available tags:", event_acc.Tags())

val_loss = []


# Retrieve and print the scalar values for a specific tag (summary key)
tag_to_retrieve = "val_loss"
loss_values = event_acc.Scalars(tag_to_retrieve)
for event in loss_values:
    step = event.step
    value = event.value
    #print(f"Step: {step}, Value: {value}")

    val_loss.append(value)

# Similarly, you can access other types of data like histograms, images, etc.
# For example, to retrieve images: event_acc.Images(tag_to_retrieve)
# To retrieve histograms: event_acc.Histograms(tag_to_retrieve)



print(val_loss)
