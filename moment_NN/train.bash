#!/bin/bash
#SBATCH -N 1 # number of minimum nodes
#SBATCH -c 2 # number of cores
#comment#--gres=gpu:1080Ti:1 # Request 1 gpu
#comment SBATCH --gres=gpu:1 # Request 1 gpu
#SBATCH --gres=gpu:new:1 --constraint="titanxp|1080ti"
#SBATCH -p gip,all # submit the job to two queues(all,gip) , once it gets elected to run in one of them , it automatically removed in the other
#SBATCH --mail-user=mor1joseph@cs.technion.ac.il
# (change to your own email if you wish to get one, or just delete this and the following lines)
#SBATCH --mail-type=ALL # Valid values are NONE, BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH --job-name="FebVer"
#SBATCH -o slurm.%N.%j.out # stdout goes here
#SBATCH -e slurm.%N.%j.out # stderr goes here
#SBATCH --time=30:00:00
#SBATCH -w gaon2

source /etc/profile.d/modules.sh
# comment module load cuda/9.0-7.0.5
module load cuda/9.0-7.0.5
#module load cuda/9.2-7.2.1
nvidia-smi
#source ~/tensorflow/bin/activate
#python3 train2.py --log_dir "log_test35" --learning_rate 0.001 --model "pointnet_cls_basic_79" --data_dir "data6"
python3 train.py --log_dir "log_6"
