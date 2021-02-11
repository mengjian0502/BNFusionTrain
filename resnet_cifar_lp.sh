PYTHON="/home/mengjian/anaconda3/envs/neurosim_test/bin/python3"

############ directory to save result #############

if [ ! -d "$DIRECTORY" ]; then
    mkdir ./save
    mkdir ./dataset
fi

model=resnet20_QF
dataset=cifar10
epochs=200
batch_size=128
optimizer=SGD

wd=0.0005
lr=0.1

save_path="./save/${model}/${model}_lr${lr}_wd${wd}_QReLU_021021/"
log_file="${model}_lr${lr}_wd${wd}.log"

pretrained_model="./save/resnet20_Quant/resnet20_Quant_lr0.1_wd0.0005_QReLU/model_best.pth.tar"


$PYTHON -W ignore train.py --dataset ${dataset} \
    --data_path ./dataset/ \
    --model ${model} \
    --save_path ${save_path} \
    --epochs ${epochs} \
    --log_file ${log_file} \
    --lr  ${lr} \
    --schedule 60 120 \
    --gammas 0.1 0.1 \
    --batch_size ${batch_size} \
    --ngpu 1 \
    --wd ${wd} \
    --clp \
    --a_lambda ${wd};

    
