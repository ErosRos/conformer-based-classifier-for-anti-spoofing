A conformer-based classifier for variable-length utterance processing in anti-spoofing
===============

Implementation of our work "A conformer-based classifier for variable-length utterance processing in anti-spoofing" published in Interspeech 2023. For detailed insights into our methodology, you can access the complete paper [here](https://www.isca-speech.org/archive/interspeech_2023/rosello23_interspeech.html).

## Requirements
First create and activate the environment, and install torch:
```
conda create -n Conformer_W2V python=3.7
conda activate Conformer_W2V
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
```
Then install the requirements:
```
pip install -r requirements.txt
```
The implementation relies on the "fairseq" package, particularly for the Wav2Vec model. To install the specific version used in our work, navigate to  [this link](https://github.com/facebookresearch/fairseq/tree/a54021305d6b3c4c5959ac9395135f63202db8f1). Download the package and execute the following command inside the downloaded folder:
```
pip install --editable ./
```
The Wav2vec model can be dowloaded from [here](https://github.com/facebookresearch/fairseq/tree/main/examples/wav2vec/xlsr).
## Training and evaluation of fixed size
We used the LA particion of ASVspoof 2019 for training and validation, it can can be downloaded from [here](https://datashare.ed.ac.uk/handle/10283/3336).

We used ASVspoof 2021 database for evaluation. LA can be found [here](https://zenodo.org/records/4837263#.YnDIinYzZhE) and DF [here](https://zenodo.org/records/4835108#.YnDIb3YzZhE).

To train and evaluate your own model with fixed size utterances using the default parameters, execute the following command:
```
python main.py
```
Upon running this command, a new folder will be generated within the 'models' directory, containing the top 5 epochs from this training loop. Additionally, two score files will be created: one for LA and another for DF, both located in the 'Scores' folder.

### Evaluation of variable size
If you wish to evaluate one of your models using variable size utterances in the evaluation process, execute the following command:
```
python eval_lv.py
```
This command evaluates the previously generated model using the default parameters. If you prefer to use custom parameters, ensure they are specified in this step. Similar to fixed size utterance evaluation, this process also results in the creation of two score files: one for LA and another for DF, with the suffix _eval_lv.txt.

## Training and evaluation of variable size
To train and evaluate your own model using variable size utterances, simply replace main.py with main_lv.py.

When you run main_lv.py, it will generate a new folder within the 'models' directory, containing the top 5 models from different epochs. Additionally, two score files will be created: one for LA and another for DF, both located in the 'Scores' folder.

## Calculate Equal Error Rate (EER)

For the computation of the minimum t-DCF and EER (Equal Error Rate), we strongly recommend using the official ASVspoof 2021 [evaluation package](https://github.com/asvspoof-challenge/2021/tree/main/eval-package). 

While the scores for our most significant results are available in the 'Scores' folder, the precise metrics reported in our paper can be obtained by utilizing the official evaluation package.

## Citation

If you find this repository valuable for your work, please consider citing our paper:

```
@inproceedings{rosello23_interspeech,
  author={Eros Rosello and Alejandro Gomez-Alanis and Angel M. Gomez and Antonio Peinado},
  title={{A Conformer-Based Classifier for Variable-Length Utterance Processing in Anti-Spoofing}},
  year=2023,
  booktitle={Proc. INTERSPEECH 2023},
  pages={5281--5285},
  doi={10.21437/Interspeech.2023-1820}
}
```

Your citation helps us track the impact of our research and acknowledges the contributors' efforts. If you have any questions, feedback, or need further assistance, please feel free to contact me in erosrosello@ugr.es.