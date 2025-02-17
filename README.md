
# subnetwork-probing

This repository is forked from the original repository to extend functionality

This is code for the paper:  
[Low-Complexity Probing via Finding Subnetworks](https://github.com/stevenxcao/subnetwork-probing)  
Steven Cao, Victor Sanh, Alexander M. Rush  
NAACL-HLT 2021  

### installation

run `$ conda env create -n subnet_prob --file environment.yml` to install dependencies

### Dependencies
This code was tested with `python 3.6`, `pytorch 1.1`, and `pytorch-transformers 1.2`.

### Data
This paper uses the [Universal Dependencies](https://universaldependencies.org/) dataset for dependencies and part-of-speech tagging, and the [CoNLL 2003 NER Shared Task](https://www.clips.uantwerpen.be/conll2003/ner/) for named entity recognition. 

### Running the code
To run the main experiment in the paper, run `python main.py PATH_TO_SAVE_DIRECTORY`.
