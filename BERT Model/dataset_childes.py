import random
import typing
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from tqdm import tqdm
from torch.utils.data import Dataset
from torchtext.vocab import vocab
from torchtext.data.utils import get_tokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class WikitextBertDataset(Dataset):
    """
    A class to represent the CHILDES transcript dataset
    

    Attributes
    ----------
    path: str
        the relative path to the file
    ds_from: int
        starting row of data
    ds_to: int
        ending row of data
    should_include_test: boolean
        whether or not to include textual representation of the created sentences in the data frame
    """
    CLS = '[CLS]'
    PAD = '[PAD]'
    SEP = '[SEP]'
    MASK = '[MASK]'
    UNK = '[UNK]'

    MASK_PERCENTAGE = 0.15  # percentage of words to mask

    MASKED_INDICES_COLUMN = 'masked_indices'
    TARGET_COLUMN = 'indices'
    NSP_TARGET_COLUMN = 'is_next'
    TOKEN_MASK_COLUMN = 'token_mask'

    OPTIMAL_LENGTH_PERCENTILE = 70

    def __init__(self, path, ds_from=None, ds_to=None, should_include_text=False):
        self.ds: pd.Series = pd.read_csv(path)['Transcript']

        if ds_from is not None or ds_to is not None:
            self.ds = self.ds[ds_from:ds_to]

        self.tokenizer = get_tokenizer('basic_english')
        self.counter = Counter()
        self.vocab = None

        self.optimal_sentence_length = None
        self.should_include_text = should_include_text

        if should_include_text:
            self.columns = ['masked_sentence', self.MASKED_INDICES_COLUMN, 'sentence', self.TARGET_COLUMN,
                            self.TOKEN_MASK_COLUMN,
                            self.NSP_TARGET_COLUMN]
        else:
            self.columns = [self.MASKED_INDICES_COLUMN, self.TARGET_COLUMN, self.TOKEN_MASK_COLUMN,
                            self.NSP_TARGET_COLUMN]
        self.df = self.prepare_dataset()

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        item = self.df.iloc[idx]

        inp = torch.Tensor(item[self.MASKED_INDICES_COLUMN]).long()
        token_mask = torch.Tensor(item[self.TOKEN_MASK_COLUMN]).bool()

        mask_target = torch.Tensor(item[self.TARGET_COLUMN]).long()
        mask_target = mask_target.masked_fill_(token_mask, 0)

        attention_mask = (inp == self.vocab[self.PAD]).unsqueeze(0)

        if item[self.NSP_TARGET_COLUMN] == 0:
            t = [1, 0]
        else:
            t = [0, 1]

        nsp_target = torch.Tensor(t)

        return (
            inp.to(device),
            attention_mask.to(device),
            token_mask.to(device),
            mask_target.to(device),
            nsp_target.to(device)
        )
    
    def prepare_dataset(self) -> pd.DataFrame:
        """
        Split dataset as necessary to create vocabulary and preprocess dataset by categorizing sentences on being a true Next Sentence Predictor (NSP) or a false NSP
        """
        sentences = []
        nsp = []
        sentence_lens = []

        # Split dataset on sentences
        for review in self.ds:
            review_sentences = review.split('. ')
            sentences += review_sentences
            self._update_length(review_sentences, sentence_lens)
        self.optimal_sentence_length = self._find_optimal_sentence_length(sentence_lens)

        print("Create vocabulary")  
        for sentence in tqdm(sentences):
            s = self.tokenizer(sentence)
            self.counter.update(s)
  
        self._fill_vocab()

        print("Preprocessing dataset")
        for paragraph in tqdm(self.ds):
            paragraph_sentences = paragraph.split('. ')
            if len(paragraph_sentences) > 1:
                for i in range(len(paragraph_sentences) - 1):
                    # True NSP item
                    first, second = self.tokenizer(paragraph_sentences[i]), self.tokenizer(paragraph_sentences[i + 1])
                    nsp.append(self._create_item(first, second, 1))

                    # False NSP item
                    first, second = self._select_false_nsp_sentences(sentences)
                    first, second = self.tokenizer(first), self.tokenizer(second)
                    nsp.append(self._create_item(first, second, 0))
        df = pd.DataFrame(nsp, columns=self.columns)
        return df
    
    def _update_length(self, sentences: typing.List[str], lengths: typing.List[int]):
        for v in sentences:
            l = len(v.split())
            lengths.append(l)
        return lengths
    
    def _find_optimal_sentence_length(self, lengths: typing.List[int]): 
        """
        Helper function for prepare_dataset method to calculate the optimal sentence length by finding the 70th percentile of sentence lengths from dataset

        Parameters
        ----------
        lengths: List[int]
            A list of integers of lengths of sentences in dataset
        """
        arr = np.array(lengths)  
        return int(np.percentile(arr, self.OPTIMAL_LENGTH_PERCENTILE))

    def _fill_vocab(self):  
        """
        Helper methdod for prepare_dataset method to build the vocabulary from dataset
        """
        # specials= argument is only in 0.12.0 version  
        # specials=[self.CLS, self.PAD, self.MASK, self.SEP, self.UNK]
        self.vocab = vocab(self.counter, min_freq=2)  

        # 0.11.0 uses this approach to insert specials  
        self.vocab.insert_token(self.CLS, 0)  
        self.vocab.insert_token(self.PAD, 1)  
        self.vocab.insert_token(self.MASK, 2)  
        self.vocab.insert_token(self.SEP, 3)  
        self.vocab.insert_token(self.UNK, 4)  
        self.vocab.set_default_index(4)

    def _create_item(self, first: typing.List[str], second: typing.List[str], target: int = 1):
        """
        Helper method for prepare_dataset method to create the NSP item. Returns the nsp_sentence indices, the original indices, the inverse token mask, and the target.

        Parameters
        ----------
        first: List[str]
            A list of strings represententing a sentence in dataset
        second: List[str]
            A list of strings representing the sentence that follows the first sentence in dataset
        """
        # Create masked sentence item
        updated_first, first_mask = self._preprocess_sentence(first.copy())
        updated_second, second_mask = self._preprocess_sentence(second.copy())

        nsp_sentence = updated_first + [self.SEP] + updated_second
        nsp_indices = self.vocab.lookup_indices(nsp_sentence)
        inverse_token_mask = first_mask + [True] + second_mask

        # Create sentence item without masking random words
        first, _ = self._preprocess_sentence(first.copy(), should_mask=False)
        second, _ = self._preprocess_sentence(second.copy(), should_mask=False)
        original_nsp_sentence = first + [self.SEP] + second
        original_nsp_indices = self.vocab.lookup_indices(original_nsp_sentence)

        if self.should_include_text:
            return (
                nsp_sentence,
                nsp_indices,
                original_nsp_sentence,
                original_nsp_indices,
                inverse_token_mask,
                target
            )
        else:
            return (
                nsp_indices,
                original_nsp_indices,
                inverse_token_mask,
                target
            )

    def _select_false_nsp_sentences(self, sentences: typing.List[str]):
        """
        Helper method for prepare_dataset method to select sentences to create false NSP item. Returns a tuple of two sentences, one sentence and another that is random (NOT the next sentence following first).


        Parameters
        ----------
        sentences: List[str]
            A list of strings representing all sentences
        """
        sentences_len = len(sentences)
        sentence_index = random.randint(0, sentences_len - 1)
        next_sentence_index = random.randint(0, sentences_len - 1)

        #it's NOT real next sentence
        while next_sentence_index == sentence_index + 1:
            next_sentence_index = random.randint(0, sentences_len - 1)

        return sentences[sentence_index], sentences[next_sentence_index]

    def _preprocess_sentence(self, sentence: typing.List[str], should_mask: bool = True):
        """
        Helper method for _create_item method to mask and pad sentence. Returns the list of strings of sentences and the masking token

        Parameters
        ----------
        sentence: List[str]
            A list of strings represententing sentences
        should_mask: boolean
            A boolean representing whether sentence should be masked
        """
        inverse_token_mask = None
        if should_mask:
            sentence, inverse_token_mask = self._mask_sentence(sentence)
        sentence, inverse_token_mask = self._pad_sentence([self.CLS] + sentence, inverse_token_mask)

        return sentence, inverse_token_mask

    def _mask_sentence(self, sentence: typing.List[str]):
        """
        Helper method of _preprocess_sentence to replace MASK_PERCENTAGE (15%) of words with special [MASK] symbol
        or with random word from vocabulary. Returns the sentence and the masking token.

        Parameters
        ----------
        sentence: List[str]
            A list of strings represententing the sentence in question
        """
        len_s = len(sentence)
        inverse_token_mask = [True for _ in range(max(len_s, self.optimal_sentence_length))]

        mask_amount = round(len_s * self.MASK_PERCENTAGE)
        for _ in range(mask_amount):
            i = random.randint(0, len_s - 1)

            if random.random() < 0.8:
                sentence[i] = self.MASK
            else:
                # All is below 5 is special token
                # see self._insert_specials method
                j = random.randint(5, len(self.vocab) - 1)
                sentence[i] = self.vocab.lookup_token(j)
            inverse_token_mask[i] = False
        return sentence, inverse_token_mask

    def _pad_sentence(self, sentence: typing.List[str], inverse_token_mask: typing.List[bool] = None):
        """
        Helper method for _preprocess_sentence method to insert [CLS] to beginning of sentence and add [PAD] token to the end of the sentence

        Parameters
        ----------
        sentence: List[str]
            A list of strings represententing sentences
        inverse_token_mask: List[bool]
            A list of booleans that representing which indices are masked or not
        """
        len_s = len(sentence)

        if len_s >= self.optimal_sentence_length:
            s = sentence[:self.optimal_sentence_length]
        else:
            s = sentence + [self.PAD] * (self.optimal_sentence_length - len_s)

        # inverse token mask should be padded as well
        if inverse_token_mask:
            len_m = len(inverse_token_mask)
            if len_m >= self.optimal_sentence_length:
                inverse_token_mask = inverse_token_mask[:self.optimal_sentence_length]
            else:
                inverse_token_mask = inverse_token_mask + [True] * (self.optimal_sentence_length - len_m)
        return s, inverse_token_mask


if __name__ == '__main__':
    BASE_DIR = Path(__file__).resolve().parent.parent

    ds = ChildesBertDataset(BASE_DIR.joinpath('train_output_child.csv'), ds_from=0, ds_to=50000,
                         should_include_text=True)
    print(ds.df)

    



    
        
