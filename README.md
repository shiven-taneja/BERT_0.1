# BERT_0.1

Investigating the Impact of Pre-training on Child Language Data for BERT Models and its Performance on Downstream Grammatical Tasks

The following project is the creation of a limited BERT model based on the guidance of a Coaxsoft tutorial (https://coaxsoft.com/blog/building-bert-with-pytorch-from-scratch). Beyond building the model, we use data from a collection of child-direct corpora (CHILDES dataset, children stories dataset from Hugging Face, and children book dataset from Hugging Face) as well as standard corpora (Wikitext-103-v1 from Hugging Face) in order to pre-train our model and analyze MLM and NSP loss functions.

Beyond this we also used such datasets to evaluate a given Bert-uncased model (provided by Hugging Face) on downstream MLM tasks, such that we can explore the grammatical aspects that are captured and not captured by the model when pre-trained on the different quality and quantity of data. 

Abstract:
The transformer architecture, specifically BERT (Bidirectional Encoder Representations from Transformers), has recently become a popular choice for Natural Language Processing (NLP) tasks. This is due to its ability to capture long-term dependencies as well as contextual information. This paper aims to investigate the influence of pre-training datasets on BERT models, focusing on child language data, which is characterized by variability and variations from common linguistic structures. By replicating a BERT model, we were able to examine the loss difference between different pre-training datasets. We then evaluated the performance of a Hugging Face Bert model pretrained on these different datasets (both child oriented and adult oriented) on a Masked Language Modeling task with different grammatical sentence structures. We found that all the models were able to capture the correct grammatical parts of speech but failed to consider specific contextual information which is likely due to insufficient training time. The large differences in the sizes of the datasets may have impeded our ability to isolate the effects of training on child-directed inputs and thus further investigation is necessary.


Created by: 
Shiven Taneja (https://github.com/shiven-taneja)
Priyanka Shakira Raj (https://github.com/prishakira)
