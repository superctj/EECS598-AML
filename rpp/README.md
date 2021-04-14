# Robust Physical Perturbation

This folder contains part of the source code (and some results) used in
[Robust Physical-World Attacks on Deep Learning Visual Classification](https://arxiv.org/abs/1707.08945).
These source code are adapted from the author's [repository](https://github.com/evtimovi/robust_physical_perturbations),
and carry an [MIT license](https://github.com/evtimovi/robust_physical_perturbations/blob/master/LICENSE).
Original source code copyright belongs to the authors. Note that the original
authors of RPP included portions of an older version of the
[cleverhans](https://github.com/tensorflow/cleverhans) library for
compatibility, which carry its own
[MIT License](https://github.com/tensorflow/cleverhans/blob/master/LICENSE).
This README is also partially adapted from README written by original authors

Only the code to attack the
LISA-CNN that classifies US road signs from the LISA dataset is modified and
included in this repo. The  model achieves 91% accuracy on that dataset.
This is the most rudimentary implementation of the algorithm.

The large data and model contained in the author's repository has been removed.
Please refer to the author's
[repository](https://github.com/evtimovi/robust_physical_perturbations)
to obtain these data.

## Changes
For EECS 598-AML replication, we converted this directory to python 3.6.
To get started, download [tf 1.4.1](https://pypi.org/project/tensorflow/1.4.1/#files)
```
conda create -n rpp python=3.6
conda activate rpp
pip install # the wheel files you downloaded
pip install keras==1.2.0 scipy opencv-python pillow
```

***

# LISA-CNN
The rest of the file is the README written by original authors:

This directory contains the code for attacking the LISA-CNN model and the model itself. It is self-contained and there should be nothing for you to download in order to run the attack. It also contains the outputs of the runs that generated two of the attacks in the paper (in `optimization_output`).

## Important: Tensorflow and Keras Versions
This code runs with tensorflow version 1.4.1 and keras version 1.2.0. We recommend following these steps to ensure that you're not running into version mismatch problems:

1. Make sure you have [pipenv](https://docs.pipenv.org/) installed.
2. From the top-level folder of the repo, execute the following commands:
```
cd lisa-cnn-attack
rm Pipfile
pipenv install tensorflow==1.4.1
```
(change this to `pipenv install tensorflow-gpu==1.4.1`, if you have a GPU on your system)

```
pipenv install keras==1.2.0
pipenv install scipy
pipenv install opencv-python
pipenv install pillow
pipenv shell
```

3. The last command opens up a pipenv shell for you and in it, `run_attack_many.sh` should run fine.

## Driver Scripts

To run, use the script `run_attack_many.sh` inside a [pipenv](https://docs.pipenv.org/) shell. It is set up in the repo so that it replicates the subliminal poster attack. To see a description of what all the parameters mean, run `python gennoise_many_images.py -h` or look at the definitions of the various command line flags that specify the optimization parameters.

The file `Pipfile` specifies the exact version of the packages we used. Newer versions of tensorflow and keras don't always work  with this code. We also include an older version of cleverhans (see below).

Moreover, the script `run_noise_to_big_img.sh` takes the noise (adversarial perturbation), as stored in some checkpoint file, resizes it, and applies it to a high-res image.

## Outputs of Optimization Runs
The "traces" of our optimization runs are stored in these folders:
* `optimization_output/noinversemask_second_trial_run` has the noise and optimization parameters for the subliminal poster attack. The `run_attack_many.sh` is set up to replicate that training run and save it under a folder called `octagon`.

* `optimization_output/l1basedmask_uniformrectangles` contains the outputs from optimizing for a camouflage sticker attack.

In both of these folders, the `model` subfolder contains the final tensorflow checkpoint and `noisy_images` holds images with the perturbation applied to them saved at regular intervals during the attack optimization.

The `optimization_output_*.txt` files hold the printouts of the optimization parameters. Use these values in `run_noise_to_big_img.sh`  if you want to replicate any one optimization run.

## Classify Using the Model
The model is to be found under `models/all_r_ivan`. To classify images using it run `python manyclassify.py --attack_srcdir <folder>` where `<folder>` is the path to a folder **of only 32 by 32 png images**. This code is *not* set up to auto-resize images or throw away non-png files in the directory, so it might error out if you don't follow that guideline.

## Attack Code and Cleverhans
The attack graph itself and the code to run it are in the files `gennoise_many_images.py` and `utils.py`. We also include an older version of the core of the [cleverhans](https://github.com/tensorflow/cleverhans) library. It carries its own MIT license.

