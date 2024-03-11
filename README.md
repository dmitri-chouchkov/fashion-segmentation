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

# tips for compiling floodfill.c
* You can see the compiling and linking commands for gcc in the tasks.json file. You need to provide the include paths for your python installation and your numpy installation
* You need -DMS_WIN64 flag when building on windows 
* If you are on windows then you either have to recompile python using gcc or you have to generate a new libpython38.def (using gendef.exe) and libpython38.a file (using the previous def file and dlltool.exe) for python38.dll (or whatever version you are using) and put it in the libs directory (not lib)
* If you are using linux like a sane person then it should be enough to just have the include directories in the build command and you should not have to worry about linking (unless you are not using gcc) 
* You can build and link into an .so file with a single command, I have them separate as a holdover when I was desperately trying to make linking on windows work 
* Python version of the algorithm takes a minute to run. You don't want to do it that way. 

# give me your weight file
* Weight files are pretty big, if you contact me I can provide a temporary google drive link. The current model is not commercially viable since it cannot handle backgrounds.

# does this actually use deepspeed
* I think you need to run "accelerate config" before deepspeed functionality is enabled (I did not test it locally because it does not work on windows). Alternatively you can try using torch.compile instead if you are looking for a speed up, deepspeed and torch.compile are currently not compatible with each other  

# will executing this code steal all of my bitcoin
* that is a fairly safe assumption


