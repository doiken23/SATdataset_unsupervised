#PBS -q cq
#PBS -b 1
#PBS -l cpunum_job=3
#PBS -l gpunum_job=1
#PBS --group=g-nedo-geospatial

python train_dcgan.py \
    data/sat-6-full.mat \
    --batchsize 16 \
    --epochs 1 \
    --log out/0001
