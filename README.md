# fashion-segmentation
 unet with attention bottleneck and optional local attention upblocks

# dataset

https://github.com/yumingj/DeepFashion-MultiModal

# tips for running on colab
* checkpoints should be stored on google drive in case colab disconnects you
* limit number of checkpoints stored on google drive to 1 or 2
* empty the google drive recycling bin often if you are going through checkpoints, they still take up space
    * you can purge google drive files by opening them in write mode and then saving them with 0 bytes of content before deleting them
* do not store directories of training data on google drive, you will hit a read cap during training and then google drive operations will randomly fail
* you can train for about 3 hours before being prompted with captcha, after about 8 hours it is very likely that you are disconnected 
* tensorboard crashes if you have multiple runs in the same directory, the colab file provided purges the log directory before starting a tensorboard session
* RNG is not consistent between colab and local environments if you have different versions of numpy etc installed
    * Since our code used RNG to split the training and testing set, we manually copied the test set indicies for local testing 
    * You are more than welcome to split the data into training, validation, and test sets into different directories beforehand and not worry about RNG
    * Fixing RNG between runs does not work exactly if you enable multithreaded dataloaders 


# enabling local attention
* copy the most recent run.py command (using the previous --output_file_name and --output_dir in your new --pretrained_model_path) and add --upgrade_local_attention as a flag for 1 epoch
* (using the previous --output_file_name and --output_dir in your new --pretrained_model_path) copy the most recent run.py command and replace --upgrade_local_attention with --model_use_local_attention
    * It is strongly recommended to modify unet.py to restrict which layers have local attention enabled, the performance penalty increases with the spatial resolution (becoming massive at 512 x 512) 


