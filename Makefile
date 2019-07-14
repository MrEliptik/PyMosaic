python-bin = ~/.pyenv/versions/3.6.7/envs/pymosaic/bin/python

bond_RGB:
	$(python-bin) mosaic.py --target_im=images/target/bond.jpg 		\
		--inputs=images/input/ --resize_factor=1

bond_RGB_multithreading:
	$(python-bin) mosaic.py --target_im=images/target/bond.jpg 		\
		--inputs=images/input/ --resize_factor=1 --multithreading	\
		--num_workers=4

bond3_RGB_multithreading:
	$(python-bin) mosaic.py --target_im=images/target/bond3.jpg 	\
		--inputs=images/input/ --resize_factor=1 --multithreading 	\
		-num_workers=4

bond3_grayscale_multithreading:
	$(python-bin) mosaic.py --target_im=images/target/bond3.jpg 				\
		--inputs=images/input/ --resize_factor=1 --pixel_density=0.25 			\
		--grayscale --multithreading --num_workers=12 --output_size_factor=5	\
		--save

bond3_grayscale_autocontrast_multithreading:
	$(python-bin) mosaic.py --target_im=images/target/bond3.jpg			\
		--inputs=images/input/ --resize_factor=1 --pixel_density=0.25	\
		--grayscale --output_size_factor=5 --multithreading 			\
		--num_workers=12 --save --contrast

girl_grayscale_multithreading:
	$(python-bin) mosaic.py --target_im=images/target/girl.jpg			\
		--inputs=images/input/ --resize_factor=1 --pixel_density=0.25	\
		--grayscale --output_size_factor=5 --contrast 					\
		--multithreading --num_workers=12 --save

lena_grayscale_multithreading:
	$(python-bin) mosaic.py --target_im=images/target/lena.jpg			\
		--inputs=images/input/ --resize_factor=1 --pixel_density=0.25	\
		--grayscale --output_size_factor=5 --contrast 					\
		--multithreading --num_workers=12 --save

bond2_grayscale_autocontrast_multithreading:
	$(python-bin) mosaic.py --target_im=images/target/bond2.jpg			\
		--inputs=images/input/ --resize_factor=1 --pixel_density=0.25	\
		--grayscale --output_size_factor=5 --multithreading 			\
		--num_workers=12 --save --contrast

bond_grayscale_autocontrast_multithreading:
	$(python-bin) mosaic.py --target_im=images/target/bond.jpg			\
		--inputs=images/input/ --resize_factor=1 --pixel_density=0.25	\
		--grayscale --output_size_factor=5 --multithreading 			\
		--num_workers=12 --save --contrast

bond_grayscale_multithreading:
	$(python-bin) mosaic.py --target_im=images/target/bond.jpg			\
		--inputs=images/input/ --resize_factor=1 --pixel_density=0.25	\
		--grayscale --output_size_factor=5 --multithreading 			\
		--num_workers=12 --save

stranger_things_grayscale_multithreading:
	$(python-bin) mosaic.py --target_im=images/target/stranger_things.jpg			\
		--inputs=images/input/ --resize_factor=1 --pixel_density=0.3				\
		--grayscale --output_size_factor=10 --multithreading 						\
		--num_workers=12 --save