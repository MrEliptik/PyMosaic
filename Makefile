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