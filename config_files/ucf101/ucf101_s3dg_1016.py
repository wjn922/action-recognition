data = dict(
	dataset='ucf101',
	batch_size=6,   	# batch size on each gpu
	size=224,           # frame size
	train_file='./data/ucf101/train_folder01.txt',
	train_img_tmp='{:05d}.jpg',
	train_clip_len=16,
	val_file='./data/ucf101/test_folder01.txt',
	val_img_tmp='{:05d}.jpg',
	val_clip_len=16,
	test_file='./data/ucf101/test_folder01.txt',
	test_img_tmp='{:05d}.jpg',
	test_clip_len=16,)
model = dict(num_class=101
	)

seed = 0
num_epochs = 150 	 # Number of epochs for training
interval = 1 	     # Store a model every snapshot epochs
display = 100        # display info every 'display' steps
lr = 1e-2 # Learning rate
lr_scheduler = dict(
	type='multistep',
	gamma=0.1,
	step=[100,125])


# Please keep these four cfgs as None, and specify them in the args
work_dir = 'runs/ucf101_s3dg_1016'
load_from = 'ckpt/S3D_kinetics400.pt'
resume_from = None
gpus = 8


