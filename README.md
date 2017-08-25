# Context-Aware Representations for Knowledge Base Relation Extraction

## Relation extraction on an open-domain knowledge base


Accompanying code for our EMNLP 2017 paper providing the code to replicate the experiments and the pre-trained models.
 
This repository contains experimental software and is published for the sole purpose of giving additional background details on the respective publication. 
 
Please use the following citation:

```
@inproceedings{TUD-CS-2017-0119,
	title = {Context-Aware Representations for Knowledge Base Relation Extraction},
	author = {Sorokin, Daniil and Gurevych, Iryna},
	booktitle = {Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
	pages = {(to appear)},
	year = {2017},
	location = {Copenhagen, Denmark},
}
```

### Paper abstract:
> We demonstrate that for sentence-level relation extraction it is beneficial to consider other relations in the sentential context while predicting the target relation. Our architecture uses an LSTM-based encoder to jointly learn representations for all relations in a single sentence. 
We combine the context representations with an attention mechanism to make the final prediction. 
>We use the Wikidata knowledge base to construct a dataset of multiple relations per sentence and to evaluate our approach. Compared to a baseline system, our method results in an average error reduction of 24\% on a held-out set of relations.



The dataset described in the paper can be found here:
 * https://www.ukp.tu-darmstadt.de/data/
 
 
### Contacts:
If you have any questions regarding the code, please, don't hesitate to contact the authors or report an issue.
  * Daniil Sorokin, lastname@ukp.informatik.tu-darmstadt.de
  * https://www.ukp.tu-darmstadt.de
  * https://www.tu-darmstadt.de
  
### Project structure:


### Setup:

* We recommend that you setup a new pip environment first: http://docs.python-guide.org/en/latest/dev/virtualenvs/ 

* Check out the repository and run:
```
pip3 install -r requirements.txt
```
* Set the Keras (deep learning library) backend to Theano (even deeper deep learning library) with the following command:
```
export KERAS_BACKEND=theano
```
You can also permanently change Keras backend (read more: https://keras.io/backend/).

#### Requirements:
* Python 3.4
* See requirements.txt for library requirements. 

### License:
* Apache License Version 2.0