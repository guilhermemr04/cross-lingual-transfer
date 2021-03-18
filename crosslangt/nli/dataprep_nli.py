import logging
import os
import torch

from pickle import HIGHEST_PROTOCOL
from typing import List
from xml.etree import ElementTree as etree

from .datasets import NLIExample, NLIDataset

from crosslangt.dataset_utils import download_and_extract
from tqdm import tqdm
from transformers import PreTrainedTokenizer
from transformers.data.processors import DataProcessor, InputExample
from transformers.data.processors.glue import MnliProcessor

#### editeeeei
import spacy
from spacy.lang.pt import Portuguese
from tqdm.notebook import tqdm
from transformers import MarianMTModel, MarianTokenizer

# Model
model_name = 'Helsinki-NLP/opus-mt-ROMANCE-en'
marian_tokenizer = MarianTokenizer.from_pretrained(model_name)
marian_model = MarianMTModel.from_pretrained(model_name)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#### editeeeei

logger = logging.getLogger(__name__)

#### editeeei
def chunkstring_spacy(text):
    chunck_sentences = []
    nlp = Portuguese()
    nlp.add_pipe(nlp.create_pipe('sentencizer'))
    doc = nlp(text)
    for sent in doc.sents:
        chunck_sentences.append('>>en<<' + ' ' + sent.text)
        
    return chunck_sentences

def translate(aux_sent):
    max_length = 512
    num_beams = 1

    sentence = chunkstring_spacy(aux_sent)

    #Move o modelo para a GPU
    marian_model.to(device)
    marian_model.eval()

    tokenized_text = marian_tokenizer.prepare_translation_batch(sentence, max_length=max_length)
                        
    translated = marian_model.generate(input_ids=tokenized_text['input_ids'].to(device), 
                                        max_length=max_length, 
                                        num_beams=num_beams, 
                                        early_stopping=True, 
                                        do_sample=False)
                        
    tgt_text = [marian_tokenizer.decode(t, skip_special_tokens=True) for t in translated]
    return ' '.join(tgt_text)
####### editeeeei

class MnliNoContradictionProcessor(MnliProcessor):
    def get_train_examples(self, data_dir):
        """See base class."""
        examples = super().get_train_examples(data_dir)
        return filter(self._no_contradiction, examples)

    def get_dev_examples(self, data_dir):
        """See base class."""
        examples = super().get_dev_examples(data_dir)
        return filter(self._no_contradiction, examples)

    def get_test_examples(self, data_dir):
        """See base class."""
        examples = super().get_test_examples(data_dir)
        return filter(self._no_contradiction, examples)

    def get_labels(self):
        return ['entailment', 'neutral']

    def _no_contradiction(self, example: InputExample):
        return example.label != 'contradiction'


class ASSIN2Processor(DataProcessor):
    """ Converts ASSIN2 dataset files into InputExample objects. """
    def get_train_examples(self, data_dir):
        return self._create_examples(
            os.path.join(data_dir, 'assin2-train-only.xml'), 'train')

    def get_dev_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, 'assin2-test.xml'),
                                     'dev')

    def get_labels(self):
        """Gets the list of labels for this data set."""
        return ['entailment', 'none']

    def map_label(self, label_original_value):
        return label_original_value

    def _create_examples(self, file, set_type) -> List[InputExample]:
        with open(file, 'r') as assin_xml:
            xml_file_contents = assin_xml.read()
            xml_file_contents = xml_file_contents.encode('utf-8')

        root = etree.fromstring(xml_file_contents)
        examples = []

        for i, pair in enumerate(root):
            entailment = self.map_label(pair.attrib['entailment'].lower())
            pairID = pair.attrib['id']
            sentence1 = sentence2 = ''

            if entailment not in self.get_labels():
                continue  # Ignoring labels that are not in MNLI dataset

            for child in pair:
                if child.tag == 't':
                    sentence1 = child.text
                else:
                    sentence2 = child.text

            examples.append(
                InputExample(pairID, sentence1, sentence2, entailment))

        return examples


class ASSIN2MnliAlignedProcessor(ASSIN2Processor):
    def get_labels(self):
        """Gets the list of labels for this data set."""
        return ["contradiction", "entailment", "neutral"]

    def map_label(self, label_original_value):
        if label_original_value == 'none':
            return 'neutral'

        return label_original_value


# In the NLI Datasets, the labels must be aligned, so transfer learning
# across languages is possible.
NLI_DATASETS = {
    'mnli': {
        'zip': 'GOOGLE_DRIVE', #'https://dl.fbaipublicfiles.com/glue/data/MNLI.zip', # 'GOOGLE_DRIVE'
        'train': 'MNLI/train.tsv',
        'eval': 'MNLI/dev_matched.tsv',
        'processor': MnliProcessor,
    },
    'assin2': {
        'zip': 'https://github.com/lersouza/cross-lingual-transfer/raw/'
               'master/datasets/ASSIN2.zip',
        'train': 'ASSIN2/assin2-train-only.xml',
        'eval': 'ASSIN2/assin2-dev.xml',
        'test': 'ASSIN2/assin2-test.xml',
        'processor': ASSIN2Processor,
    },
    'assin2-mnlialigned': {
        'zip': 'https://github.com/lersouza/cross-lingual-transfer/raw/'
               'master/datasets/ASSIN2.zip',
        'train': 'ASSIN2/assin2-train-only.xml',
        'eval': 'ASSIN2/assin2-dev.xml',
        'test': 'ASSIN2/assin2-test.xml',
        'processor': ASSIN2MnliAlignedProcessor,
    },
    'mnli-nocontradiction': {
        'zip': 'https://dl.fbaipublicfiles.com/glue/data/MNLI.zip',
        'train': 'MNLI/train.tsv',
        'eval': 'MNLI/dev_matched.tsv',
        'processor': MnliNoContradictionProcessor,
    },
}


def get_features_file(data_dir: str, dataset: str, split: str,
                      max_seq_length: int, features_key: str):
    """
    Returns the name of the feature file.
    """
    filename = f'nli-{features_key}-{dataset}-{split}-{max_seq_length}.dataset'
    filepath = os.path.join(data_dir, filename)

    return filepath


def prepare_nli_dataset(dataset: str,
                        split: str,
                        data_dir: str,
                        tokenizer: PreTrainedTokenizer,
                        max_seq_length: int,
                        features_key: str = '',
                        force: bool = False):
    """
    Prepares the NLI Dataset file for training/eval.

    - dataset: The dataset to prepare (mnli, assin2)
    - split: The split to prepare (train or eval)
    - data_dir: The data directory where dataset files are.
    - tokenizer: The PreTrainedTokenizer to use when tokenizing texts.
    - max_seq_length: The max Sequence length to produce.
    - features_key: An optional key to identify dataset file.
    - force: Whether or not to force dataset creation.
    """
    final_dataset_path = get_features_file(data_dir, dataset, split,
                                           max_seq_length, features_key)

    if os.path.exists(final_dataset_path) and not force:
        logger.info(f'Dataset {final_dataset_path} already there. Skipping.')

        return final_dataset_path

    logger.info(f'Dataset will be generated at: {final_dataset_path}.')

    if os.path.exists(final_dataset_path):
        os.remove(final_dataset_path)

    data_config = NLI_DATASETS[dataset]
    data_file_path = os.path.join(data_dir, data_config[split])

    if not os.path.exists(data_file_path):
        logger.info(f'Downloading zip file for {data_file_path}.')
        download_and_extract(data_config['zip'], data_dir)

    processor = data_config['processor']()
    features = extract_features(data_file_path, split, max_seq_length,
                                tokenizer, processor)

    with open(final_dataset_path, 'wb') as processed_dataset_file:
        torch.save(features,
                   processed_dataset_file,
                   pickle_protocol=HIGHEST_PROTOCOL)

    return final_dataset_path


def extract_features(data_file_path: str, split: str, max_seq_length: int,
                     tokenizer: PreTrainedTokenizer,
                     processor: DataProcessor) -> List[NLIExample]:
    """
    Extract features for the given dataset file, using `processor`.
    Returns a list of NLIExample.
    """
    nli_base_dir, _ = os.path.split(data_file_path)

    logger.info(f'About to extract examples in {nli_base_dir} '
                f'with {type(processor)}')

    examples = (processor.get_train_examples(nli_base_dir) if split == 'train'
                else processor.get_dev_examples(nli_base_dir))

    features = []
    available_labels = processor.get_labels()

    for example in tqdm(examples, desc='tokenizing examples'):
        ###### editeeeei
    #    if split != 'train':
    #        example.text_a = translate(example.text_a)
    #        example.text_b = translate(example.text_b)
        ##### editeeeeeei
        encoded = tokenizer.encode_plus(example.text_a,
                                        example.text_b,
                                        max_length=max_seq_length,
                                        pad_to_max_length=True,
                                        truncation=True)

        encoded['label'] = available_labels.index(example.label)
        encoded['pairID'] = example.guid

        features.append(NLIExample(**encoded))

    return features


def load_nli_dataset(data_dir: str, dataset: str, split: str,
                     max_seq_length: int, features_key: str = ''):

    dataset_filepath = get_features_file(data_dir, dataset, split,
                                         max_seq_length, features_key)

    if not os.path.exists(dataset_filepath):
        logger.error(f'Dataset {dataset_filepath} not found. '
                     'Consider prepare first!')

        raise IOError(f'Dataset {dataset_filepath} was not found.')

    with open(dataset_filepath, 'rb') as dataset:
        contents = torch.load(dataset)

    return NLIDataset(contents)
