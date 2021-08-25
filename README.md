# This GitHub creatred by Andre Godin is provided for the course 'McGill Neural Networks and Deep Learning - YCNG 229'

# __Objective__

General Data Protection Regulation (GDPR) has set a new standard for Ericsson’s customer rights regarding their data. Ericsson is looking at migrating more and more workload into some off-premises cloud solution (e.g., public cloud like AWS, GCP, Azure) in different geographically regions. Hundreds of customer managed service contracts to revisit to __validate if there are any constraint with regards to what we are allowed to do with customer data__. We need some automation and intelligence to facilitate this tedious task.  

The goal is to use a text classifier to determine whether or not an extracted sentence indicates that the customer contract is putting some constraints on what we are allowed to do with their data.

One of the important factors that determine the generalization of a model is the amount and variety of related data it sees during training.

__Data augmentation techniques can be used to enrich the dataset__ to train the classifier on. 2 techniques were explored and implemented

* Lexical substitution: EDA (Easy Data Augmentation)

* Generative Model/Pre-trained Transformer Models: GPT2

___

# __Data augmentation techniques for NLP__

# 1- EDA (Easy Data Augmentation)
---
Data Augmentation techniques in NLP using EDA show substantial improvements on datasets with less than 500 observations, as illustrated by the original paper: https://arxiv.org/abs/1901.11196

One technique called EDA is using lexical substition with synonym replacement as a technique in which we replace a word by one of its synonyms. It uses WordNet, a large linguistic database, to identify relevant synonyms.

## Getting Started
---
1. Load the __EDA.iypnb__ file in Google Colab as a Notebook 
2. Mount your Google Drive with automated authentication by opening the file browser, and clicking the 'Mount Drive' button.
3. Create a __data/__ folder into MyDrive 
4. Upload the dataset, __EDA_dataset.txt__ file, into to __data__ folder.


## Generating Text
---
1. Assuming you are done with steps under __Getting Started__
2. Run each cell of the __EDA.iypnb__ Notebook
3. Results are printed in the Notebook as well as being stored in the /MyDrive/data/output_<textfile\>.


_Sample Results_
```
Input Sentence: 
the customer wants its data to stay in its premises
Outputs:
the customer need its data to stay in its premises
the customer need its data to remain in its premises
the client need its data to remain in its premises
the client need its information to remain in its premises
```

# 2- Generative Model/Pre-trained Transformer Models: GPT2
---
GPT2 for high-quality text augmentation. GTP2 is based on Transformer Architecture trained on 40GB of WebText. It’s a stack of multiple decoder units on top of each other enabled with some advanced learning concepts like Masked Self Attention (giving more importance to some input states in which it has more contextual relation). __The objective that GPT2 tries to optimize is essentially to predict the next word in the sequence having seen past words.__

The concept is mainly two steps: Train/fine-tune and generate. Once the model is trained (and stored), the model is ready to be used to generate samples. It uses a Top-k, Top-p sampling strategy to sample word at each timestep (t) while decoding. This strategy helps to generate variety in the text under controlled circumstances.

## Getting Started
---
1. Load the __GPT2.iypnb__ file in Google Colab as a Notebook
* Mount your Google Drive with automated authentication by opening the file browser, and clicking the 'Mount Drive' button.
* Create a __data/__ folder into MyDrive 
* Upload the dataset, __GPT2_dataset.csv__ file, into to __data__ folder.

  
## Training and Generating Text
---
1. Assuming you are done with steps under __Getting Started__
* Run each cell of the __GPT2.iypnb__ Notebook
* Results are printed in the Notebook as well as being stored in /MyDrive/data/output_<textfile\>.

__Note:__
* For training and fine-tuning the GPT2 model, you can change those constant default values in the train_gpt2() method of the MyDataset class:
  * BATCH_SIZE = 32
  * EPOCHS = 5
  * LEARNING_RATE = 3e-5
  * WARMUP_STEPS = 30
  * MAX_SEQ_LEN = 200


* For generating text, you can change those constant default values that you pass to the generate() function: 
  * SENTENCES = 20
  * SENTENCE_LENGTH = 20
  * START_SENTENCE = 'the customer data cannot'

_Sample Results_
```
the customer data cannot be retained for any longer than permitted under applicable law.
the customer data cannot be shared to anyone else, as it would be stored on a third-party server or otherwise compromised by anyone)," they write.
the customer data cannot be deleted and there is no liability for any loss caused by the use of this data").
the customer data cannot be shared with the NSA or others", he said.
```

## Important Points to Note
---

__Note:__ First time you run, it will take much more amount of time because of the following reasons - 
1. Downloads pre-trained gpt2-medium model  _(Depends on your Network Speed)_
2. Fine-tunes the gpt2 with your dataset _(Depends on size of the data, number of Epochs, Hyperparameters, etc.)_


## Files included
---
* README.md: intructions file
* EDA.iypnb: EDA Notebook
* EDA_dataset.txt: EDA dataset sample
* EDA_outout.txtx: EDA results
* GTP2.iypnb: GPT2 Notebook
* GPT2_dataset.csv: GPT2 dataset sample
* GPT2_output.txt: GPT2 results
* Presentation.pptx: Power point presentation file of the project


# __References__

* [1] [A Visual Survey of Data Augmentation in NLP](https://amitness.com/2020/05/data-augmentation-for-nlp/)
* [2] [EDA: Easy Data Augmentation Techniques for Boosting Performance on
Text Classification Tasks](*https://arxiv.org/pdf/1901.11196.pdf)
* [3] [Data Augmentation in NLP using GPT2](https://prakhartechviz.blogspot.com/2020/04/data-autmentation-in-nlp-gpt2.html)
* [4] [The Illustrated GPT-2 - Visualizing Transformer Language Models](http://jalammar.github.io/illustrated-gpt2/)
* [5] [OpenAI GPT2](https://huggingface.co/transformers/model_doc/gpt2.html#gpt2tokenizer)

---