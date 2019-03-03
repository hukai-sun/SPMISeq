# SPMISeq toolkit

A SPMI Lab toolkit for sequence tagging tasks based on Pytorch

This toolkit includes the source code corresponding to ["Neural CRF transducers for sequence labeling"](http://arxiv.org/abs/1811.01382)

## Quick Start

### Installing dependencies

1. Install python 3.6. Anaconda is suggested.

2. Install pytorch 0.4.1. For better performance, install with GPU support.

3. Install tensorflow 1.3.0.

### Training

1. **Prepare the sequence tagging task rawtext.**
    
    * You should convert the tagging scheme to BIOES tagging scheme. 
    
    * Unpack the dataset into `~/raw/rawtext`, after unpacking, your file tree should look like this:
        ```
        SPMISeq
          |- raw
            |- rawtext
                |- train.txt
                |- valid.txt
                |- test.txt
        ```

2. **Download a word embedding.**
   
     * After downloading the word embedding, your file tree should look like this:
       ```
       SPMISeq
         |- raw
           |- embedd
             |- XXX_word_embedding.txt
       ```
3. **Train a model.**
    ```
    python train.py
    ```
    Tunable hyperparameters are found in [hparams.py](hparams.py). 
    
    You can also adjust some basic settings at the command line, try `python train.py -h`  for more help.
   

## Notes
  
  * For the training of the Neural CRF Transducer, you need to pretrain a RNN Transducer and then load the RNN Transducer's parameters to initialize the weigths of the Neural CRF Transducer (which is found to yield better and faster learning).

## Code reference
  * By XuezheMax/NeuroNLP2: https://github.com/XuezheMax/NeuroNLP2
