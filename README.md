# PyMosaic

**!README IN CONSTRUCTION!**

Recreate a target images by making a mosaic out of multiple input images.

## Examples

<p align="center">
    <img src="https://i.imgur.com/n82gCys.jpg" alt="bond grayscale mosaic examples"/>
</p>


<p align="center">
    <img src="https://i.imgur.com/g6w1319.jpg" alt="bond3 grayscale mosaic examples"/>
</p>

*Note: Images have been scale down to take less space*

## Quickstart

To get started quickly you can use the provided example input and target images. The example is using the `--grayscale` option as it yields better results. Also, the `--multithreading` option is really important to get sufficient speed. The algorithm is quite long, and without multithreading you'll wait for a long time, especially if the image is big. 

Simply run

    make lena_grayscale_multithreading

or use the equivalent script call

    python mosaic.py --target_im=images/target/lena.jpg			        \
		--inputs=images/input/ --resize_factor=1 --pixel_density=0.25	\
		--grayscale --output_size_factor=5 --contrast 					\
		--multithreading --num_workers=12 --save

To get a detailed list of the arguments, head to [How to use](#how-to-use) section.

- `--resize_factor=1` ensures the initial image size is kept the same when going through it
- `--pixel_density=0.25` tells how big the pixel scanning must be
- `--grayscale` to use grayscale mode (yields better results)
- `--output_size_factor=5` the output image will be 5 times the initial image size
- `--contrast` applies CLAHE to the image to enhance the constrat before using it

Resulting image

<p align="center">
    <img src="https://i.imgur.com/9M8qXnK.jpg" alt="mosaic examples"/>
</p>

*Note: Image has been scale down to take less space*

## How to use

## Requirements

Simply run

    pip install -r requirements.txt

## TODO

- [X] Add argument parsing
- [X] B&W support
- [X] Be able to select output resolution
- [X] Add option to save mosaic
- [X] Add/test contrast increasing before mosaic
- [X] Use pixel density argument
- [X] Add color filtering option (for smoother results)
- [] Improve color filtering
- [] Add option to choose to display the output or not
- [] Add web interface
