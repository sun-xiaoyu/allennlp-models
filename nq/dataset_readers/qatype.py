import itertools
from typing import Dict, Optional
import json
import logging

from overrides import overrides
from allennlp.data.tokenizers import Token
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, LabelField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Tokenizer, SpacyTokenizer, PretrainedTransformerTokenizer

logger = logging.getLogger(__name__)


@DatasetReader.register("qatype")
class QatypeReader(DatasetReader):
    """
    Reads a file from the Stanford Natural Language Inference (SNLI) dataset.  This data is
    formatted as jsonl, one json-formatted instance per line.  The keys in the data are
    "gold_label", "sentence1", and "sentence2".  We convert these keys into fields named "label",
    "premise" and "hypothesis", along with a metadata field containing the tokenized strings of the
    premise and hypothesis.

    Registered as a `DatasetReader` with name "snli".

    # Parameters

    tokenizer : `Tokenizer`, optional (default=`SpacyTokenizer()`)
        We use this `Tokenizer` for both the premise and the hypothesis.  See :class:`Tokenizer`.
    token_indexers : `Dict[str, TokenIndexer]`, optional (default=`{"tokens": SingleIdTokenIndexer()}`)
        We similarly use this for both the premise and the hypothesis.  See :class:`TokenIndexer`.
    combine_input_fields : `bool`, optional
            (default=`isinstance(tokenizer, PretrainedTransformerTokenizer)`)
        If False, represent the premise and the hypothesis as separate fields in the instance.
        If True, tokenize them together using `tokenizer.tokenize_sentence_pair()`
        and provide a single `tokens` field in the instance.
    """

    def __init__(
        self,
        tokenizer: Optional[Tokenizer] = None,
        token_indexers: Dict[str, TokenIndexer] = None,
        combine_input_fields: Optional[bool] = None,
        with_answer=True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.with_answer = with_answer
        self._tokenizer = tokenizer or SpacyTokenizer()
        if isinstance(self._tokenizer, PretrainedTransformerTokenizer):
            assert not self._tokenizer._add_special_tokens
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self.non_content_type_id = max(
            self._tokenizer.tokenizer.encode_plus("left", "right", return_token_type_ids=True)[
                "token_type_ids"
            ]
        )
        if combine_input_fields is not None:
            self._combine_input_fields = combine_input_fields
        else:
            self._combine_input_fields = isinstance(self._tokenizer, PretrainedTransformerTokenizer)

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        import torch.distributed as dist
        from allennlp.common.util import is_distributed
        #
        # if is_distributed():
        #     start_index = dist.get_rank()
        #     step_size = dist.get_world_size()
        #     logger.info(
        #         "Reading SNLI instances %% %d from jsonl dataset at: %s", step_size, file_path
        #     )
        # else:
        #     start_index = 0
        #     step_size = 1
        #     logger.info("Reading SNLI instances from jsonl dataset at: %s", file_path)
        logger.info("Reading file at %s", file_path)
        with open(file_path, 'r') as dataset_file:
            dataset_json = json.load(dataset_file)
            dataset = dataset_json["data"]
        logger.info("Reading the dataset")
        logger.info(f'Predict with answer:{self.with_answer}')
        logger.info(f'Type 3(yes) and 4(no) will be combined: {not self.with_answer}')
        num_yielded = 0

        logger.info(f'Will add "answer" to instances: {self.with_answer}')
        for example in dataset:
            label = example["answer_type"]
            question = example["question"]
            answer = example["answer_text"]
            # no no-answer!
            # if label == 0:
            #     continue
            # new_version = True
            # if new_version and label == 1:
            #     continue
            # if not self.with_answer:
            #     if label == 4:
            #         label = 3
            #     yield self.text_to_instance(label, question)
            #     num_yielded += 1
            # else:
            yield self.text_to_instance(label, question, answer)
            num_yielded += 1
        logger.info(f'Instance yielded: {num_yielded}')

    @overrides
    def text_to_instance(
        self,  # type: ignore
        label: int or None = None,
        question: str = '',
        answer: str = None,
    ) -> Instance:
        # logger.info(str(label) + ' ' + question + answer[:50])

        cls_token = Token(
            self._tokenizer.tokenizer.cls_token,
            text_id=self._tokenizer.tokenizer.cls_token_id,
            type_id=self.non_content_type_id,
        )
        sep_token = Token(
            self._tokenizer.tokenizer.sep_token,
            text_id=self._tokenizer.tokenizer.sep_token_id,
            type_id=self.non_content_type_id,
        )

        fields: Dict[str, Field] = {}
        question = self._tokenizer.tokenize(question)

        if answer:
            answer = self._tokenizer.tokenize(answer)
            tokens = [cls_token] + question + [sep_token, sep_token] + answer + [sep_token]
            if len(tokens) > 512:
                tokens = tokens[:511] + [sep_token]
            fields["tokens"] = TextField(tokens, self._token_indexers)
        else:
            tokens = [cls_token] + question + [sep_token]
            if len(tokens) > 512:
                tokens = tokens[:511] + [sep_token]
            fields["tokens"] = TextField(tokens, self._token_indexers)

        if label:
            assert label is not None
            fields["label"] = LabelField(label, skip_indexing=True)

        return Instance(fields)
