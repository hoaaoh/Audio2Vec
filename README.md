# Audio2Vec

You have to get the "prons.n.gz" from kaldi/get\_prons.sh and concate the files
together and named as all\_prons, also gets the relative features as
cmvned\_feats.ark

Then you can use the scripts inside the directory like  

        ./script/train\_origin.sh ./feat 100 ./model ./log 1 80000

Since the criteria of the script is 

        ./script/train\_origin.sh feat-dir memory-dim model-dir log-dir gpu-num max-training steps


# Training Inference 
----------------

You may change the model structure by modifying the function `inference`, `loss`
in `src/audio2vec\_train.py`, if you want to modify the cell or the
elemental connection of sequence-to-sequence structure, please visit `src/seq2seq.py`


Starts training by using `script/train\_origin.sh`
Evaluation by using `src/audio2vec_eval.py`

