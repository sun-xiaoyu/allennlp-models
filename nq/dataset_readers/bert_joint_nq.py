import json
import logging, random
import gzip
import json
import enum
import collections
import re
import glob
from typing import Any, Dict, List, Tuple, Optional, Iterable

from allennlp.common.util import sanitize_wordpiece
from allennlp.data.fields import MetadataField, TextField, SpanField, LabelField
from overrides import overrides

from allennlp.common.file_utils import cached_path, open_compressed
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import PretrainedTransformerIndexer
from allennlp.data.tokenizers import Token, PretrainedTransformerTokenizer

from nq.utils import char_span_to_token_span

# from allennlp_models.rc.common.reader_utils import char_span_to_token_span
# from nq import BertJointNQReaderSimple

logger = logging.getLogger(__name__)


class Flag:
    def __init__(self):
        self.mode = 'all_examples'
        self.skip_nested_contexts = True
        self.max_contexts = 48
        self.max_position = 50
        # self.max_instances = 100000


FLAGS = Flag()


class AnswerType(enum.IntEnum):
    """Type of NQ answer."""
    UNKNOWN = 0
    SHORT = 1
    LONG = 2
    YES = 3
    NO = 4


def make_nq_answer(input_text) -> AnswerType:
    """Makes an Answer object following NQ conventions.

    Args:
    contexts: string containing the context
    answer: dictionary with `span_start` and `input_text` fields

    Returns:
    an Answer object. If the Answer type is YES or NO or LONG, the text
    of the answer is the long answer. If the answer type is UNKNOWN, the text of
    the answer is empty.
    """

    if input_text.lower() == "short":
        answer_type = AnswerType.SHORT
    elif input_text.lower() == "yes":
        answer_type = AnswerType.YES
    elif input_text.lower() == "no":
        answer_type = AnswerType.NO
    elif input_text.lower() == "long":
        answer_type = AnswerType.LONG
    else:
        answer_type = AnswerType.UNKNOWN

    return answer_type


def has_long_answer(a):
    return (a["long_answer"]["start_token"] >= 0 and
            a["long_answer"]["end_token"] >= 0)


def should_skip_context(e, idx):
    if (FLAGS.skip_nested_contexts and
            not e["long_answer_candidates"][idx]["top_level"]):
        return True
    elif not get_candidate_text(e, idx).text.strip():
        # Skip empty contexts.
        return True
    else:
        return False


def get_first_annotation(e):
    """Returns the first short or long answer in the example.

    Args:
    e: (dict) annotated example.

    Returns:
    annotation: (dict) selected annotation
    annotated_idx: (int) index of the first annotated candidate.
    annotated_sa: (tuple) char offset of the start and end token
        of the short answer. The end token is exclusive.
    """
    positive_annotations = sorted(
        [a for a in e["annotations"] if has_long_answer(a)],
        key=lambda a: a["long_answer"]["candidate_index"])

    for a in positive_annotations:
        if a["short_answers"]:
            idx = a["long_answer"]["candidate_index"]
            start_token = a["short_answers"][0]["start_token"]
            end_token = a["short_answers"][-1]["end_token"]
            return a, idx, (token_to_char_offset(e, idx, start_token),
                            token_to_char_offset(e, idx, end_token) - 1)

    for a in positive_annotations:
        idx = a["long_answer"]["candidate_index"]
        return a, idx, (-1, -1)

    return None, -1, (-1, -1)


def get_text_span(example, span):
    """Returns the text in the example's document in the given token span."""
    token_positions = []
    tokens = []
    for i in range(span["start_token"], span["end_token"]):
        t = example["document_tokens"][i]
        if not t["html_token"]:
            token_positions.append(i)
            token = t["token"].replace(" ", "")
            tokens.append(token)
    return TextSpan(token_positions, " ".join(tokens))


def token_to_char_offset(e, candidate_idx, token_idx):
    """Converts a token index to the char offset !within! the candidate."""
    c = e["long_answer_candidates"][candidate_idx]
    char_offset = 0
    for i in range(c["start_token"], token_idx):
        t = e["document_tokens"][i]
        if not t["html_token"]:
            token = t["token"].replace(" ", "")
            char_offset += len(token) + 1
    return char_offset


def get_candidate_type(e, idx):
    """Returns the candidate's type: Table, Paragraph, List or Other."""
    c = e["long_answer_candidates"][idx]
    first_token = e["document_tokens"][c["start_token"]]["token"]
    if first_token == "<Table>":
        return "Table"
    elif first_token == "<P>":
        return "Paragraph"
    elif first_token in ("<Ul>", "<Dl>", "<Ol>"):
        return "List"
    elif first_token in ("<Tr>", "<Li>", "<Dd>", "<Dt>"):
        return "Other"
    else:
        logger.warning("Unknown candidate type found: %s", first_token)
        return "Other"


def add_candidate_types_and_positions(e):
    """Adds type and position info to each candidate in the document."""
    counts = collections.defaultdict(int)
    for idx, c in candidates_iter(e):
        context_type = get_candidate_type(e, idx)
        if counts[context_type] < FLAGS.max_position:
            counts[context_type] += 1
        c["type_and_position"] = "[%s=%d]" % (context_type, counts[context_type])
        #   TODO
        #   not change here!
        # c["type_and_position"] = "[%s]" % context_type


def get_candidate_type_and_position(e, idx):
    """Returns type and position info for the candidate at the given index."""
    if idx == -1:
        return "[NoLongAnswer]"
    else:
        return e["long_answer_candidates"][idx]["type_and_position"]


def get_candidate_text(e, idx):
    """Returns a text representation of the candidate at the given index."""
    # No candidate at this index.
    if idx < 0 or idx >= len(e["long_answer_candidates"]):
        return TextSpan([], "")

    # This returns an actual candidate.
    return get_text_span(e, e["long_answer_candidates"][idx])


def candidates_iter(e):
    """Yield's the candidates that should not be skipped in an example."""
    for idx, c in enumerate(e["long_answer_candidates"]):
        if should_skip_context(e, idx):
            continue
        yield idx, c


def create_example_from_jsonl(line):
    """Creates an NQ example from a given line of JSON."""
    e = json.loads(line, object_pairs_hook=collections.OrderedDict)
    add_candidate_types_and_positions(e)
    annotation, annotated_idx, annotated_sa = get_first_annotation(e)

    # annotated_idx: index of the first annotated context, -1 if null.
    # annotated_sa: short answer start and end char offsets, (-1, -1) if null.
    # important! answer span are character level! not token level.
    answer = {
        "candidate_id": annotated_idx,
        "span_text": "",
        "span_start": -1,
        "span_end": -1,
        "answer_type": "unknown",
    }

    # Yes/no answers are added in the input text.
    if annotation is not None:
        assert annotation["yes_no_answer"] in ("YES", "NO", "NONE")
        if annotation["yes_no_answer"] in ("YES", "NO"):
            answer["answer_type"] = annotation["yes_no_answer"].lower()

    # Add a short answer if one was found.
    if annotated_sa != (-1, -1):
        answer["answer_type"] = "short"
        span_text = get_candidate_text(e, annotated_idx).text
        answer["span_text"] = span_text[annotated_sa[0]:annotated_sa[1]]
        answer["span_start"] = annotated_sa[0]
        answer["span_end"] = annotated_sa[1]
        expected_answer_text = get_text_span(
            e, {
                "start_token": annotation["short_answers"][0]["start_token"],
                "end_token": annotation["short_answers"][-1]["end_token"],
            }).text
        assert expected_answer_text == answer["span_text"], (expected_answer_text,
                                                             answer["span_text"])

    # Add a long answer if one was found.
    elif annotation and annotation["long_answer"]["candidate_index"] >= 0:
        answer["span_text"] = get_candidate_text(e, annotated_idx).text
        answer["span_start"] = 0
        answer["span_end"] = len(answer["span_text"])
        answer["answer_type"] = "long"

    # todo  为什么要加这个 -1 context？为什么
    # 因为为了给没有长答案的加一个 token，好让对应的start,end都指向它
    context_idxs = [-1]
    context_list = [{"id": -1, "type": get_candidate_type_and_position(e, -1)}]
    context_list[-1]["text_map"], context_list[-1]["text"] = (
        get_candidate_text(e, -1))
    for idx, _ in candidates_iter(e):
        context = {"id": idx, "type": get_candidate_type_and_position(e, idx)}
        context["text_map"], context["text"] = get_candidate_text(e, idx)
        context_idxs.append(idx)
        context_list.append(context)
        if len(context_list) >= FLAGS.max_contexts:
            break

    # Assemble example.
    example = {
        # "name": e['document_title'] if 'document_title' in e else 'Wikipedia',
        # the official simplify_nq script does not preserve doc title, that is why we don't use it.
        "id": str(e["example_id"]),
        "doc_url": e['document_url'],
        "question_text": e["question_text"],
        "answers": [answer],
        "has_correct_context": annotated_idx in context_idxs  # what is this?
    }
    # todo 不has？？

    single_map = []
    single_context = []
    offset = 0
    for context in context_list:
        single_map.extend([-1, -1])
        single_context.append("[ContextId=%d] %s" %
                              (context["id"], context["type"]))
        # TODO change here
        # single_context.append("[Context] %s" % context["type"])
        offset += len(single_context[-1]) + 1
        if context["id"] == annotated_idx:
            answer["span_start"] += offset
            answer["span_end"] += offset

        # Many contexts are empty once the HTML tags have been stripped, so we
        # want to skip those.
        if context["text"]:
            single_map.extend(context["text_map"])
            single_context.append(context["text"])
            offset += len(single_context[-1]) + 1

    example["contexts"] = " ".join(single_context)
    example["contexts_map"] = single_map
    if annotated_idx in context_idxs:
        expected = example["contexts"][answer["span_start"]:answer["span_end"]]

        # This is a sanity check to ensure that the calculated start and end
        # indices match the reported span text. If this assert fails, it is likely
        # a bug in the data preparation code above.
        assert expected == answer["span_text"], (expected, answer["span_text"])

    return example


# A special token in NQ is made of non-space chars enclosed in square brackets.
_SPECIAL_TOKENS_RE = re.compile(r"^\[[^ ]*\]$", re.UNICODE)
# TODO change here, can be moved into config or add number
# NQ_SPECIAL_TOKEN = {'bos_token': ['[Table]', '[Paragraph]', '[List]', '[Other]']}
TextSpan = collections.namedtuple("TextSpan", "token_positions text")


@DatasetReader.register("bert_joint_nq")
class BertJointNQReader(DatasetReader):
    """
    Reads a JSON-formatted SQuAD file and returns a ``Dataset`` where the ``Instances`` have four
    fields:
     * ``question_with_context``, a ``TextField`` that contains the concatenation of question and context,
     * ``answer_span``, a ``SpanField`` into the ``question`` ``TextField`` denoting the answer.
     * ``context_span`` a ``SpanField`` into the ``question`` ``TextField`` denoting the context, i.e., the part of
       the text that potential answers can come from.
     * A ``MetadataField`` that stores the instance's ID, the original question, the original passage text, both of
       these in tokenized form, and the gold answer strings, accessible as ``metadata['id']``,
       ``metadata['question']``, ``metadata['context']``, ``metadata['question_tokens']``,
       ``metadata['context_tokens']``, and ``metadata['answers']. This is so that we can more easily use the
       official SQuAD evaluation script to get metrics.

    We also support limiting the maximum length for the question. When the context+question is too long, we run a
    sliding window over the context and emit multiple instances for a single question. At training time, we only
    emit instances that contain a gold answer. At test time, we emit all instances. As a result, the per-instance
    metrics you get during training and evaluation don't correspond 100% to the SQuAD task. To get a final number,
    you have to run the script in scripts/transformer_qa_eval.py.

    Parameters
    ----------
    transformer_model_name : ``str``, optional (default=``bert-base-cased``)
        This reader chooses tokenizer and token indexer according to this setting.
    length_limit : ``int``, optional (default=384)
        We will make sure that the length of context+question never exceeds this many word pieces.
    stride : ``int``, optional (default=128)
        When context+question are too long for the length limit, we emit multiple instances for one question,
        where the context is shifted. This parameter specifies the overlap between the shifted context window. It
        is called "stride" instead of "overlap" because that's what it's called in the original huggingface
        implementation.
    neg_pos_ratio : ``int``, optional (default=1)
        This is the sup (upper bound) of the neg/pos ratio of instances to be yielded from one entry. Default if one
        neg_instance vs 1 pos_instance, if it is set at 2, for 1 entry, the neg_instance per entry is no greater than 2.
    enable_downsample_strategy : ``bool``, optional (default=False)
        If we will employ a strategy to select negative instances instead of random sampling. The current strategy is to
        focus on rank the neg_instances by the elmo similarity between their context and the question, and use stratified
        sampling with the downsample_strategy_partition
    downsample_strategy_partition : ``Tuple``, optional (default=(0.05, 0.2, 0.75))
        The portion of the downsample strategy (if enable_downsample_strategy== True)
    skip_invalid_examples: ``bool``, optional (default=False)
        If this is true, we will skip examples that don't have a gold answer. You should set this to True during
        training, and False any other time.
    max_query_length : ``int``, optional (default=64)
        The maximum number of wordpieces dedicated to the question. If the question is longer than this, it will be
        truncated.
    """

    def __init__(
            self,
            transformer_model_name: str = "bert-base-cased",
            length_limit: int = 512,
            stride: int = 128,
            max_query_length: int = 64,
            skip_invalid_examples: bool = False,
            neg_pos_ratio: int = 1,
            enable_downsample_strategy=False,
            downsample_strategy_partition=(0.05, 0.2, 0.75),
            vocab_len_location="/home/sunxy-s18/data/nq/vocab.len",
            **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self._tokenizer = PretrainedTransformerTokenizer(
            transformer_model_name, add_special_tokens=False, calculate_character_offsets=True
        )
        self._token_indexers = {"tokens": PretrainedTransformerIndexer(transformer_model_name)}
        self.length_limit = length_limit
        self.stride = stride
        self.skip_invalid_examples = skip_invalid_examples
        self.neg_pos_ratio = max(1, int(neg_pos_ratio))  # has to be an integer >= 1
        self.enable_downsample_strategy = enable_downsample_strategy
        self.downsample_strategy_partition = downsample_strategy_partition
        self.max_query_length = max_query_length
        self.non_content_type_id = max(
            self._tokenizer.tokenizer.encode_plus("left", "right", return_token_type_ids=True)[
                "token_type_ids"
            ]
        )
        self.tokenizer = self._tokenizer.tokenizer
        self.vocab_len_location = vocab_len_location

        # workaround for a bug in the transformers library
        if "distilbert" in transformer_model_name:
            self.non_content_type_id = 0

    @overrides
    def _read(self, input_paths: str):
        # if `file_path` is a URL, redirect to the cache
        path_to_dir = cached_path(input_paths)

        logger.info("Reading file at %s", path_to_dir)

        """Read a NQ json file into a list of NqEntry."""

        input_files = glob.glob(path_to_dir + '*')

        def _open(file_path):
            if file_path.endswith(".gz"):
                return gzip.GzipFile(fileobj=open(file_path, "rb"))
            else:
                return open(file_path, 'r', encoding='utf-8')

        def _report(nc, nk, p, yqc, qwmtoi, iyn):
            logger.info('$' * 30 + ' Total instances ' + '$' * 30)
            logger.info(
                'Total yielded: %d; Neg(cut/keep): (%d, %d); Pos: %d',
                p + nk, nc, nk, p)
            if yqc > 0:
                logger.info(
                    "%d (%.2f%%) questions have more than one instance",
                    qwmtoi,
                    100 * qwmtoi / yqc
                )
                logger.info('Average instance generated per entry: %.2f',
                            (p + nk + nc) / len(iyn))
                logger.info('Average instance yielded per entry: %.2f', sum(iyn) /
                            len(iyn))
                logger.info('Current vocab length: %d', len(self.tokenizer))
                with open(self.vocab_len_location, 'w') as f:
                    f.write(str(len(self.tokenizer)))

        yielded_question_count = 0
        questions_with_more_than_one_instance = 0
        instances_yielded_num = []

        neg_cut = 0
        neg_kept = 0
        pos = 0
        report_step = max(1, self.max_instances // max(5, len(input_files))) if self.max_instances is not None else 100
        last_report_num = 0
        instances_yielded = 0
        # instance 的数目每过一个 report_step，打印一次 instance 的正负比例

        for path in input_files:
            logger.info("Reading: %s", path)
            # instance_this_file = 0
            with _open(path) as input_file:
                for line in input_file:
                    entry = create_example_from_jsonl(line)

                    doc_info = {
                        'id': entry['id'],
                        'doc_url': entry['doc_url'],
                        # 'name': entry['name']
                    }
                    instances = self.make_instances(
                        doc_info=doc_info,
                        question=entry["question_text"],
                        answers=entry['answers'],
                        context=entry['contexts'],
                        first_answer_offset=entry['answers'][0]['span_start'],
                    )

                    # 此处区分正负例
                    # 随后对负例进行（有条件）的采样
                    pos_instances = []
                    neg_instances = []
                    for instance in instances:
                        if instance.fields['answer_type'].label == AnswerType.UNKNOWN:
                            neg_instances.append(instance)
                        else:
                            pos_instances.append(instance)

                    lucky_ones = []
                    pos_keep_this_entry = 0
                    if pos_instances:
                        # assert len(pos_instances) <= 1  # this is not true because of the stride
                        lucky_ones.extend(random.sample(pos_instances, 1))
                        pos_keep_this_entry = 1
                        # todo should we randomly choose one pos_ex? or employ a strategy
                        # 正例不止一个的时候随便选一个？
                    neg_kept_this_entry = 0
                    if neg_instances:
                        # try:
                        #     assert len(neg_instances) > 1  # 不一定对
                        # except:
                        #     print(neg_instances)
                        #     input()
                        #     assert False
                        neg_kept_this_entry = min(len(neg_instances), self.neg_pos_ratio)
                        lucky_ones.extend(random.sample(neg_instances, neg_kept_this_entry))
                    assert len(lucky_ones) == pos_keep_this_entry + neg_kept_this_entry
                    neg_cut += len(neg_instances)
                    for i, instance in enumerate(lucky_ones):
                        instances_yielded += 1
                        if i < pos_keep_this_entry:
                            pos += 1
                        else:
                            neg_kept += 1
                            neg_cut -= 1
                        if self.max_instances is not None and instances_yielded == self.max_instances:
                            _report(neg_cut, neg_kept, pos, yielded_question_count,
                                    questions_with_more_than_one_instance, instances_yielded_num)
                        yield instance

                    # report by step here!
                    if instances_yielded - last_report_num >= report_step:
                        logger.info('yielded: %d neg(cut/keep): (%d, %d) pos: %d',
                                    pos + neg_kept, neg_cut, neg_kept, pos)
                        last_report_num = instances_yielded

                    instances_yielded_num.append(len(lucky_ones))
                    if len(lucky_ones) > 1:
                        questions_with_more_than_one_instance += 1
                    yielded_question_count += 1

                    assert instances_yielded == pos + neg_kept
                    assert sum(instances_yielded_num) == pos + neg_kept
                    assert len(instances_yielded_num) == yielded_question_count
                    # todo
                    # if we want to sample examples from multiple files"
                    # instance_this_file += instances_yielded
                    # if instance_this_file > self.max_instance / len(input_files):
                    #     break
                    # break

        _report(neg_cut, neg_kept, pos, yielded_question_count,
                questions_with_more_than_one_instance, instances_yielded_num)

        # raw_entry = json.loads(line)
        # simplified_entry = simplify_nq_example(raw_entry)
        # entry = answer_and_context(simplified_entry)
        # instances = self.make_instances(
        #     qid=entry['example_id'],
        #     doc_url=entry['document_url'],
        #     question=entry["question_text"],
        #     context=entry['document_text'],
        #     answer=entry['answer']['text'],
        #     answer_type=entry['answer']['type'],
        #     first_answer_offset=entry['answer']['span_start'],
        # )

        #  这里把每个doc对应的qa对儿都拿出来。NQ中q和doc是一一对应的，所以下面count其实没啥用
        # Convert entry to NQExample

    def make_instances(
            self,
            doc_info: Dict,
            question: str,
            answers: List[Dict],
            context: str,
            first_answer_offset: Optional[int],
    ) -> Iterable[Instance]:
        # tokenize context by spaces first, and then with the wordpiece tokenizer
        # For RoBERTa, this produces a bug where every token is marked as beginning-of-sentence. To fix it, we
        # detect whether a space comes before a word, and if so, add "a " in front of the word.
        def tokenize_slice(start: int, end: int) -> Iterable[Token]:
            text_to_tokenize = context[start:end]
            # wordpieces = self._tokenizer.tokenize(text_to_tokenize)
            # print(wordpieces, wordpieces[0].idx)
            # return wordpieces
            if start - 1 >= 0 and context[start - 1].isspace():
                prefix = "a "  # must end in a space, and be short so we can be sure it becomes only one token
                wordpieces = self._tokenizer.tokenize(prefix + text_to_tokenize)
                # print(wordpieces, wordpieces[1].idx)
                for wordpiece in wordpieces:
                    if wordpiece.idx is not None:
                        wordpiece.idx -= len(prefix)
                return wordpieces[1:]
            else:
                return self._tokenizer.tokenize(text_to_tokenize)

        instance_generated = []
        # print(context)
        tokenized_context = []
        token_start = 0
        for i, c in enumerate(context):
            if c.isspace():
                token_text = context[token_start:i]
                if _SPECIAL_TOKENS_RE.match(token_text):
                    # TODO change here
                    # if token_text in self._tokenizer.vocab:  这里不这么搞，是因为没有手动改vocab。还是把所有这样的都加上吧
                    #     tokens balblabla
                    # else:
                    #     tokens.append(tokenizer.wordpiece_tokenizer.unk_token)

                    # 之前的代码，统一把特殊字符指向了0
                    # new_special_token = Token(token_text, idx=token_start,
                    #                                    # text_id=self._tokenizer.tokenizer.additional_special_tokens_ids,
                    #                                    text_id=0,
                    #                                    type_id=self.non_content_type_id)
                    #
                    # tokenized_context.append(new_special_token)
                    if token_text not in self.tokenizer.added_tokens_encoder:
                        self.tokenizer.add_tokens(token_text)
                        id = self.tokenizer.convert_tokens_to_ids(token_text)
                        new_special_token = Token(token_text, idx=i,
                                                  text_id=id,
                                                  # text_id=1,
                                                  type_id=self.non_content_type_id)
                        tokenized_context.append(new_special_token)
                    else:
                        id = self.tokenizer.convert_tokens_to_ids(token_text)
                        special_token = Token(token_text, idx=i,
                                              text_id=id,
                                              # text_id=1,
                                              type_id=self.non_content_type_id)
                        tokenized_context.append(special_token)

                else:
                    for wordpiece in tokenize_slice(token_start, i):
                        if wordpiece.idx is not None:
                            wordpiece.idx += token_start
                        tokenized_context.append(wordpiece)
                # else:
                # tokenized_context.append(Token(token_text, idx=token_start))
                token_start = i + 1
        # we add the last word
        for wordpiece in tokenize_slice(token_start, len(context)):
            if wordpiece.idx is not None:
                wordpiece.idx += token_start
            tokenized_context.append(wordpiece)

        # we try to transform the answer char span to token span
        try:
            if first_answer_offset is None:
                (token_answer_span_start, token_answer_span_end) = (-1, -1)
            else:
                (token_answer_span_start, token_answer_span_end), _ = char_span_to_token_span(
                    [
                        (t.idx, t.idx + len(sanitize_wordpiece(t.text))) if t.idx is not None else None
                        for t in tokenized_context
                    ],
                    (first_answer_offset, first_answer_offset + len(answers[0]["span_text"])),
                )
                # The returned span here are inclusive on both sides.

                # if (random.random() < 0.01):
                #     raise Exception("Debug")
        except:
            # Only add answers where the text can be exactly recovered from the
            # document. If this CAN'T happen it's likely due to weird Unicode
            # stuff so we will just skip the example.
            #
            # Note that this means for training mode, every example is NOT
            # guaranteed to be preserved.
            # actual_text = " ".join(doc_tokens[start_position:(end_position + 1)])
            # cleaned_answer_text = " ".join(
            #     tokenization.whitespace_tokenize(answer.text))
            # if actual_text.find(cleaned_answer_text) == -1:
            #     tf.logging.warning("Could not find answer: '%s' vs. '%s'", actual_text,
            #                        cleaned_answer_text)
            #     continue
            print('===========' * 5)
            print(doc_info['id'])
            print(question)
            print(answers)
            print('------------')
            print(context)
            print('===========' * 5)
            print(tokenized_context)
            print([
                (t.idx, t.idx + len(sanitize_wordpiece(t.text))) if t.idx is not None else None
                for t in tokenized_context
            ])
            return []

        # Tokenize the question
        tokenized_question = self._tokenizer.tokenize(question)
        tokenized_question = tokenized_question[: self.max_query_length]
        for token in tokenized_question:
            token.type_id = self.non_content_type_id
            token.idx = None

        # Stride over the context, making instances
        # Sequences are [CLS] question [SEP] [SEP] context [SEP], hence the - 4 for four special tokens.
        # This is technically not correct for anything but RoBERTa, but it does not affect the scores.
        space_for_context = self.length_limit - len(tokenized_question) - 4
        stride_start = 0
        while True:
            tokenized_context_window = tokenized_context[stride_start:]
            tokenized_context_window = tokenized_context_window[:space_for_context]

            window_token_answer_span = (
                token_answer_span_start - stride_start,
                token_answer_span_end - stride_start,
            )
            if any(i < 0 or i >= len(tokenized_context_window) for i in window_token_answer_span):
                # The answer is not contained in the window.
                window_token_answer_span = None

            if not self.skip_invalid_examples or window_token_answer_span is not None:
                additional_metadata = doc_info
                # exclusive on left ends!
                additional_metadata['context_span_orig'] = (stride_start, stride_start + space_for_context)
                instance = self.text_to_instance(
                    question,
                    tokenized_question,
                    context,
                    tokenized_context_window,
                    answers,
                    window_token_answer_span,
                    additional_metadata,
                )
                # yield instance
                instance_generated.append(instance)

            stride_start += space_for_context
            # print(f'stride_start: {stride_start}, len(tokenized_context): {len(tokenized_context)}')
            if stride_start >= len(tokenized_context):
                break
            stride_start -= self.stride
        return instance_generated

    @overrides
    def text_to_instance(
            self,  # type: ignore
            question: str,
            tokenized_question: List[Token],
            context: str,
            tokenized_context: List[Token],
            answers: List[Dict],
            token_answer_span: Optional[Tuple[int, int]],
            additional_metadata: Dict[str, Any] = None,
    ) -> Instance:
        fields = {}

        # make the question field
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

        question_with_context_field = TextField(
            (
                    [cls_token]
                    + tokenized_question
                    + [sep_token, sep_token]
                    + tokenized_context
                    + [sep_token]
            ),
            self._token_indexers,
        )
        fields["question_with_context"] = question_with_context_field
        start_of_context = 1 + len(tokenized_question) + 2

        # make the answer span
        if token_answer_span is not None:
            assert all(i >= 0 for i in token_answer_span)
            assert token_answer_span[0] <= token_answer_span[1]
            # TODO 可以加一个答案内容匹配的assert
            fields["answer_span"] = SpanField(
                token_answer_span[0] + start_of_context,
                token_answer_span[1] + start_of_context,
                question_with_context_field,
            )
            ans_type = answers[0]['answer_type']
            ans_text = answers[0]['span_text']
        else:
            # We have to put in something even when we don't have an answer, so that this instance can be batched
            # together with other instances that have answers.
            # TODO change to cls token here？
            fields["answer_span"] = SpanField(-1, -1, question_with_context_field)
            ans_type = 'unknown'
            ans_text = ""

        fields['answer_type'] = LabelField(make_nq_answer(ans_type), skip_indexing=True)

        # make the context span, i.e., the span of text from which possible answers should be drawn
        fields["context_span"] = SpanField(
            start_of_context, start_of_context + len(tokenized_context) - 1, question_with_context_field
        )

        # make the metadata
        metadata = {
            "question": question,
            "question_tokens": tokenized_question,
            "context": context,
            "context_tokens": tokenized_context,
            "answers_orig": answers,
            "answers": ans_text
        }
        # TODO add doc url & ID
        if additional_metadata is not None:
            metadata.update(additional_metadata)
        fields["metadata"] = MetadataField(metadata)
        # todo remove below
        # todo why is that?
        if ans_type == 'long' or ans_type == 'short':
            if ans_text == '':
                print(fields["metadata"])

        return Instance(fields)
