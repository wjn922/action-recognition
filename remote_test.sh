srun -p sensevideo  --mpi=pmi2 \
--gres=gpu:8 -n1 --ntasks-per-node=1 --job-name=test \
python test.py config_files/ucf101/ucf101_s3dg_1016.py \
--resume_from runs/ucf101_s3dg_1016/epoch-56.pth