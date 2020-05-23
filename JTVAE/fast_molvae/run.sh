# Note: all paths referenced here are relative to the Docker container.
#
# Add the Nvidia drivers to the path
export PATH="/usr/local/nvidia/bin:$PATH"
export HOME="/storage/home/sidnayak"
export LD_LIBRARY_PATH="/usr/local/nvidia/lib:$LD_LIBRARY_PATH"
# Tools config for CUDA, Anaconda installed in the common /tools directory

source /tools/config.sh
# Activate your environment
#export PATH="/storage/home/minerl/anaconda3/envs/minerliitm/bin":$PATH
#source activate /storage/home/minerl/covid
source activate /storage/home/minerl/.conda/envs/covid
# Change to the directory in which your code is present
cd /storage/home/minerl/JTVAE/fast_molvae/

# Run the code. The -u option is used here to use unbuffered writes
# so that output is piped to the file as and when it is produced.
# Here, the code is the MNIST Tensorflow example.
# python -u main.py --tensorboard=1 &> out

#python -u -W ignore main.py --tensorboard=1 --grad_explore=0 --env_name=SeaquestNoFrameskip-v0 --env_max_rew=50209 --num_exps=2 --algo=DDDQN $
#python -u -W ignore main.py --tensorboard=1 --grad_explore=1 --env_name=SeaquestNoFrameskip-v0 --env_max_rew=50209 --num_exps=2 --algo=DDDQN $
#python -u -W ignore main.py &> out
mkdir outs
python -u -W ignore ../fast_jtnn/mol_tree.py < ../data/sars/train.txt &> outs/outs1
mv vocab.txt ../data/sars/vocab.txt

python -u -W ignore ../fast_jtnn/mol_tree.py < ../data/psued/train.txt & outs/out2
mv vocab.txt ../data/psued/vocab.txt

python -u -W ignore preprocess.py --train ../data/moses/train.txt --split 100 --jobs 16 outs/out3
mkdir moses-processed     
mv tensor* moses-processed

mkdir vae_model/
# python -u -W ignore vae_train.py --train moses-processed --vocab ../data/moses/vocab.txt --save_dir vae_model & outs/out_main

