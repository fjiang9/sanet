## Speaker Attractor Network: Generalizing Speech Separation to Unknown Number of Sources

### Dataset
- [LibriMix](https://github.com/JorisCos/LibriMix)
- [SparseLibriMix](https://github.com/fjiang9/SparseLibriMix)
***
### Training
#### Stage 1 - Encoder-decoder pre-training
python train.py --stage 1 --n_filters 64 --kernel_size 16 --stride 8 --mask irm --enc_act relu --bias no --cuda $cuda_id --train_metadata $train_set --val_metadata $val_set --train_n_src 2 --val_n_src 2
#### Stage 2 - Embedding network training
- Conv-DANet  
python train.py --stage 2 --model_dir conv-danet --v_act yes --v_norm no --sim dotProduct --sisdr 1.0 --cuda $cuda_id --train_metadata $train_set --val_metadata $val_set --train_n_src 2 --val_n_src 2
- SANet  
python train.py --stage 2 --model_dir sanet --v_act no --v_norm yes --sim cos --alpha 10.0 --sisdr 1.0 --spk_circle 1.0 --compact 5.0 --cuda $cuda_id --train_metadata $train_set --val_metadata $val_set --train_n_src 2 --val_n_src 2
- SANet (w/o ![](https://latex.codecogs.com/svg.latex?\mathcal{L}_{spk}))  
python train.py --stage 2 --model_dir sanet-0lspk --v_act no --v_norm yes --sim cos --alpha 10.0 --sisdr 1.0 --compact 1.0 --cuda $cuda_id --train_metadata $train_set --val_metadata $val_set --train_n_src 2 --val_n_src 2
- SANet (w/o ![](https://latex.codecogs.com/svg.latex?\mathcal{L}_{com}))  
python train.py --stage 2 --model_dir sanet-0lcom --v_act no --v_norm yes --sim cos --alpha 10.0 --sisdr 1.0 --spk_circle 1.0 --cuda $cuda_id --train_metadata $train_set --val_metadata $val_set --train_n_src 2 --val_n_src 2
- SANet (trained on 2- & 3-speaker mixtures)
python train.py --stage 2 --model_dir sanet-23 --v_act no --v_norm yes --sim cos --alpha 10.0 --sisdr 1.0 --spk_circle 0.4 --compact 5.0 --cuda $cuda_id --train_metadata $train_set1 $train_set2 --val_metadata $val_set1 $val_set2 --train_n_src 2 3 --val_n_src 2 3 --batch_size 24
***
### Testing
python test.py --stage 2 --model_dir $model --cuda $cuda_id
