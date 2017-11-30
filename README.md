# Context-Aware Representations for Knowledge Base Relation Extraction

## Relation extraction on an open-domain knowledge base


Accompanying repository for our **EMNLP 2017 paper** ([full paper](https://www.ukp.tu-darmstadt.de/fileadmin/user_upload/Group_UKP/publikationen/2017/2017_EMNLP_DS_relation_extraction_camera_ready.pdf)). It contains the code to replicate the experiments and the pre-trained models for sentence-level relation extraction.

> This repository contains experimental software and is published for the sole purpose of giving additional background details on the respective publication.
 
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
> We use the Wikidata knowledge base to construct a dataset of multiple relations per sentence and to evaluate our approach. Compared to a baseline system, our method results in an average error reduction of 24\% on a held-out set of relations.

Please, refer to the paper for more details.

The dataset described in the paper can be found here:
 * https://www.ukp.tu-darmstadt.de/data/lexical-resources/wikipedia-wikidata-relations/
 
 
### Contacts:
If you have any questions regarding the code, please, don't hesitate to contact the authors or report an issue.
  * Daniil Sorokin, lastname@ukp.informatik.tu-darmstadt.de
  * https://www.ukp.tu-darmstadt.de
  * https://www.tu-darmstadt.de
  
### Demo:

You can try out the model on single sentences in our demo: 

http://semanticparsing.ukp.informatik.tu-darmstadt.de:5000/relation-extraction/

### Project structure:
```
relation_extraction/
├── apply-model.py
├── eval.py
├── model-train-and-test.py
├── notebooks
├── optimization_space.py
├── parsing
│   ├── parser.py
│   └── keras_models.py
├── relextserver
│   └── server.py
├── semanticgraph
│   ├── graph_utils.py
│   ├── io.py
│   └── vis_utils.py
├── stanford_tag_dataset.py
└── utils
    ├── embedding_utils.py
    ├── evaluation_utils.py
    └── graph.py
resources/
└── property_blacklist.txt
```

<table>
    <tr>
        <th>File</th><th>Description</th>
    </tr>
    <tr>
        <td>relation_extraction/</td><td>Main Python module</td>
    </tr>
    <tr>
        <td>relation_extraction/parsing</td><td>Models for joint relation extraction</td>
    </tr>
    <tr>
        <td>relation_extraction/relextserver</td><td>The code for the web demo.</td>
    </tr>
    <tr>
        <td>relation_extraction/semanticgraph</td><td>IO and processing for relation graphs</td>
    </tr>
    <tr>
        <td>relation_extraction/utils</td><td>IO and evaluation utils</td>
    </tr>
    <tr>
        <td>resources/</td><td>Necessary resources</td>
    </tr>
    <tr>
        <td>data/curves/</td><td>The precision-recall curves for each model on the held out data</td>
    </tr>
</table>

### Setup:

1. We recommend that you setup a new pip environment first: http://docs.python-guide.org/en/latest/dev/virtualenvs/

2. Check out the repository and run:
```
pip3 install -r requirements.txt
```

3. Set the Keras (deep learning library) backend to Theano (even deeper deep learning library) with the following command:
```
export KERAS_BACKEND=theano
```
   You can also permanently change Keras backend (read more: https://keras.io/backend/).

4. Download the [data](https://www.ukp.tu-darmstadt.de/data/lexical-resources/wikipedia-wikidata-relations/), if you want to replicate the experiments from the paper.
Extract the archive inside `emnlp2017-relation-extraction/data/wikipedia-wikidata/`.

5. Download the [GloVe embeddings, glove.6B.zip](https://nlp.stanford.edu/projects/glove/)
and put them into the folder `emnlp2017-relation-extraction/resources/glove/`.

6. Set the Theano flags:
```
 export THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32
```

#### Reproducing the experiments from the paper

1. Complete the setup above 

2. Run `python model-train-and-test.py` in `emnlp2017-relation-extraction/relation_extraction/` to see the list of parameters

3. If you put the data into teh default folders you can train the `ContextWeighted` model with the following command:
```
python model-train-and-test.py model_ContextWeighted --mode train-test
```

#### Notes

- The web demo code is provided for information only. It is not meant to be run elsewhere.

### Pre-trained models:
* You can download the models that were used in the experiments [here](https://www.ukp.tu-darmstadt.de/fileadmin/user_upload/Group_UKP/data/wikipediaWikidata/EMNLP2017_DS_IG_relation_extraction_trained_models.zip)
* We will soon make available updated models and will publish usage instructions

#### Requirements:
* Python 3.4
* See requirements.txt for library requirements. 

### License:
* Apache License Version 2.0
