python-bin = ~/.pyenv/versions/3.6.7/envs/pymosaic/bin/python

bond_RGB:
	$(python-bin) mosaic.py --target_im=images/target/bond.jpg \
		--inputs=images/input/ --resize_factor=1

bond_RGB_multithreading:
	$(python-bin) mosaic.py --target_im=images/target/bond.jpg \
		--inputs=images/input/ --resize_factor=1 --multithreading --num_workers=4

bond_grayscale:
	$(python-bin) mosaic.py --target_im=images/target/bond.jpg \
		--inputs=images/input/ --resize_factor=1 --grayscale

bond3_RGB_multithreading:
	$(python-bin) mosaic.py --target_im=images/target/bond3.jpg \
		--inputs=images/input/ --resize_factor=1 --multithreading --num_workers=4

bond3_grayscale_multithreading:
	$(python-bin) mosaic.py --target_im=images/target/bond3.jpg \
		--inputs=images/input/ --resize_factor=3 --pixel_density=0.2 --grayscale --multithreading --num_workers=10 --save

bond3_grayscale_autocontrast_multithreading:
	$(python-bin) mosaic.py --target_im=images/target/bond3.jpg \
		--inputs=images/input/ --resize_factor=1 --grayscale --multithreading --num_workers=4 --save --contrast

girl_grayscale_multithreading:
	$(python-bin) mosaic.py --target_im=images/target/girl.jpg \
		--inputs=images/input/ --resize_factor=1 --grayscale --multithreading --num_workers=4 --save