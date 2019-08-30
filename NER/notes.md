## Data
The raw data is in `resources/` and the processed data is in `data/`. You can also re-process the data using `python process_data.py <each command>`.

## Model
To train the model from scratch (in our case, pretrained BERT without finetuning), run:
1. `python -W ignore domain-tuning.py --data_dir="data/" --bert_model="bert-base-cased" --output_dir="lm_output/" --max_seq_length=128 --do_train --train_batch_size=64 --learning_rate=5e-5 --num_train_epochs=3 --warmup_proportion=0.1 --seed=2019 --fp16`
2. `python -W ignore task-tuning.py --data_dir="data/" --bert_model="bert-base-cased" --output_dir="trained_model/" --trained_model_dir="lm_output/" --max_seq_length=128 --do_train --train_batch_size=64 --learning_rate=5e-5 --num_train_epochs=3 --warmup_proportion=0.1 --seed=2019 --fp16`

We also provide our trained model at https://drive.google.com/file/d/1CygiljpJoQVfYMry8xYDtGki_B_qNXAX/view?usp=sharing. If you want to use it, simply download and unzip the linked file, and then rename the folder to `trained_model/`. This folder should contain two files: `bert_config.json` and `pytorch_model.bin`.

## Evaluation
To see the performance of the model, run `python -W ignore test.py --data_dir="data/" --bert_model="bert-base-cased" --output_dir="error_analysis_output/" --trained_model_dir="trained_model/" --max_seq_length=128 --do_test --eval_batch_size=1 --seed=2019`
