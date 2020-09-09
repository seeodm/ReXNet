# Pytorch ReXNet with center loss train code

## Introduce

This project is Clova AI ReXNet with center loss train code.

## How to train?
You have to prepare AffectNet Dataset. Data loader code is focused on AffectNet.

Here is link to download dataset. [Download](http://mohammadmahoor.com/affectnet/)

        $  python -m rexnet train --train_data data/affectnet \
                                  --eval_data data/affectnet \
                                  --batch_train 256 \
                                  --batch_eval 256 \
                                  --epochs 400 \
                                  --save_epochs 10 \
                                  --eval_epochs 5 \


The detail of command-line is as follows:

        usage: rexnet train [-h] --train_data TRAIN_DATA --valid_data VALID_DATA
                    [--train_batch_size TRAIN_BATCH_SIZE]
                    [--valid_batch_size VALID_BATCH_SIZE]
                    [--train_shuffle TRAIN_SHUFFLE]
                    [--valid_shuffle VALID_SHUFFLE]
                    [--num_workers NUM_WORKERS] [--epochs EPOCHS]
                    [--save_epochs SAVE_EPOCHS] [--eval_epochs EVAL_EPOCHS]
                    [--base_lr BASE_LR] [--lr_min LR_MIN]
                    [--lr_decay LR_DECAY] [--warmup_lr_init WARMUP_LR_INIT]
                    [--warmup_t WARMUP_T] [--cooldown_epochs COOLDOWN_EPOCHS]
                    [--momentum MOMENTUM] [--nesterov NESTEROV]
                    [--model_save_path MODEL_SAVE_PATH]
                    [--checkpoint_path CHECKPOINT_PATH] [--gpus GPUS]
                    [--center_loss CENTER_LOSS]
                    [--center_loss_lambda CENTER_LOSS_LAMBDA]
                    [--center_loss_alpha CENTER_LOSS_ALPHA]

        optional arguments:
        -h, --help            show this help message and exit

        Dataset:
        --train_data TRAIN_DATA
                                affectnet train data file path
        --valid_data VALID_DATA
                                affectnet valid data file path

        Dataset Config:
        --train_batch_size TRAIN_BATCH_SIZE
                                train batch size
        --valid_batch_size VALID_BATCH_SIZE
                                valid batch size
        --train_shuffle TRAIN_SHUFFLE
                                train data shuffle
        --valid_shuffle VALID_SHUFFLE
                                valid data shuffle
        --num_workers NUM_WORKERS
                                number of workers for data load

        Train Config:
        --epochs EPOCHS       num of total epochs
        --save_epochs SAVE_EPOCHS
                                interval epohcs of saving
        --eval_epochs EVAL_EPOCHS
                                interval epochs of eval
        --base_lr BASE_LR     base lr value
        --lr_min LR_MIN       minimum value of lr
        --lr_decay LR_DECAY   lr decay value
        --warmup_lr_init WARMUP_LR_INIT
                                base warmup lr
        --warmup_t WARMUP_T   warmup epochs
        --cooldown_epochs COOLDOWN_EPOCHS
                                cooldown epochs

        Optimizer Config:
        --momentum MOMENTUM   momentum of SGD
        --nesterov NESTEROV

        Saving Config:
        --model_save_path MODEL_SAVE_PATH
                                model save path
        --checkpoint_path CHECKPOINT_PATH
                                checkpoint save path

        Others:
        --gpus GPUS           number of gpu devices
        --center_loss CENTER_LOSS
                                use center loss
        --center_loss_lambda CENTER_LOSS_LAMBDA
                                center loss lambda
        --center_loss_alpha CENTER_LOSS_ALPHA
                                center loss alpha

## Contact Info

- 주식회사 두들린 (Doodlin Corp.)
    
    * 서동민 (Dongmin Seo)
    * sdm025@doodlin.co.kr