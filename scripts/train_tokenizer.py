import argparse
import logging
import os

from pathlib import Path
from tokenizers import BertWordPieceTokenizer

from typing import List


logger = logging.getLogger(__name__)


def train_bert_tokenizer(dataset_base_path: str, target_path: str,
                         tokenizer_name: str,
                         files_pattern: str = '**/*',
                         vocab_size: int = 30000, lower_case: bool = False):
    """
    Trains a BERT WordPiece Tokenizer based on data
    located in dataset_base_path.

    By default it reads all files in dataset_base_path. One can
    specify `files_pattern` for filtering.

    The files generated by the tokenizer will be saved under
    <target_path>/<tokenizer_name> namespace.
    """
    files = [str(f) for f
             in Path(dataset_base_path).glob(files_pattern)
             if os.path.isfile(f)]

    logger.info(f'Found {len(files)} files to use for training.')
    logger.debug(f'Files are: {files}')

    tokenizer_args = {
        'lowercase': lower_case,
        'strip_accents': False,
    }

    wordpiece_tokenizer = BertWordPieceTokenizer(**tokenizer_args)
    wordpiece_tokenizer.train(files=files, vocab_size=vocab_size)

    save_out = wordpiece_tokenizer.save(target_path, tokenizer_name)

    logger.info(f'Train finish. Result is in {save_out}')


def main():
    """ Executes tokenizer training from sys.args. """
    parser = argparse.ArgumentParser()

    parser.add_argument('dataset_base_path', type=str,
                        help='The root directory where dataset files'
                             'are located.')

    parser.add_argument('target_path', type=str,
                        help='the directory where tokenizer '
                             'will be saved.')

    parser.add_argument('tokenizer_name', type=str,
                        help='the name of the tokenizer.')

    parser.add_argument('--files_pattern', type=str, default='**/*',
                        help='A pattern to filter files in dataset_base_path')

    parser.add_argument('--vocab_size', type=int, default=30000,
                        help='the vocab size to generate (default=30k).')

    parser.add_argument('--lower_case', action='store_true',
                        help='indicates whether to perform lowercase on'
                             'vocabulary words.')

    parser.add_argument('--verbose', action='store_true',
                        help='enable verbose logging')

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if not args.verbose else logging.DEBUG)

    train_bert_tokenizer(
        args.dataset_base_path,
        args.target_path,
        args.tokenizer_name,
        args.files_pattern,
        args.vocab_size,
        args.lower_case
    )


if __name__ == '__main__':
    main()
