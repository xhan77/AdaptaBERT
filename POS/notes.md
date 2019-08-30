## Data
We cannot share PPCEME or PTB data directly, but if you have access to those datasets, we have the preprocessing code ready for you in `data/process_raw.py`.
Under `data/`, run `python process_raw.py <each command except "taglist">`.

## Model
To train the model from scratch (in our case, pretrained BERT without finetuning), run:
1. `python -W ignore domain-tuning.py --data_dir="data/processed/" --bert_model="bert-base-cased" --output_dir="lm_output/" --max_seq_length=256 --do_train --train_batch_size=30 --learning_rate=5e-5 --num_train_epochs=3 --warmup_proportion=0.1 --seed=2019 --fp16`
2. `python -W ignore task-tuning.py --data_dir="data/processed/" --bert_model="bert-base-cased" --output_dir="trained_model/" --trained_model_dir="lm_output/" --max_seq_length=256 --do_train --train_batch_size=30 --learning_rate=5e-5 --num_train_epochs=3 --warmup_proportion=0.1 --seed=2019 --fp16`

We also provide our trained model at https://drive.google.com/file/d/1YWRBdo-sSwKaHgIJjevRFrhVMcrz6Vrm/view?usp=sharing. If you want to use it, simply download and unzip the linked file, and then rename the folder to `trained_model/`. This folder should contain two files: `bert_config.json` and `pytorch_model.bin`.

## Evaluation
To see the performance of the model, run `python -W ignore test.py --data_dir="data/processed/" --bert_model="bert-base-cased" --output_dir="error_analysis_output/" --trained_model_dir="trained_model/" --max_seq_length=256 --do_test --eval_batch_size=1 --seed=2019`.
To evaluate with a coarse tagset instead of the standard PTB tagset, simply add a `--coarse_tagset` flag to the command above.
