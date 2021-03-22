import json
import logging, random
import gzip
import json
import enum
import random
import collections
import re
import glob
import tqdm
import numpy as np
import scipy as sp
from typing import Any, Dict, List, Tuple, Optional, Iterable

from allennlp.common.util import sanitize_wordpiece
from allennlp.data.fields import MetadataField, TextField, SpanField, LabelField
from overrides import overrides

from allennlp.common.file_utils import cached_path, open_compressed
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import PretrainedTransformerIndexer
from allennlp.data.tokenizers import Token, PretrainedTransformerTokenizer

from nq.nq_text_utils import simplify_nq_example
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# from allennlp_models.rc.common.reader_utils import char_span_to_token_span

# from allennlp.commands.elmo import ElmoEmbedder
# ee = ElmoEmbedder()

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


def make_nq_answer(input_text) -> int:
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
        answer_type = 1
    elif input_text.lower() == "yes":
        answer_type = 3
    elif input_text.lower() == "no":
        answer_type = 4
    elif input_text.lower() == "long":
        answer_type = 2
    else:
        answer_type = 0

    return answer_type


def has_long_answer(a):
    return (a["long_answer"]["start_token"] >= 0 and
            a["long_answer"]["end_token"] >= 0)


def should_skip_context(e, idx):
    if (FLAGS.skip_nested_contexts and
            not e["long_answer_candidates"][idx]["top_level"]):
        return True
    elif not get_candidate_text(e, idx).text:
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
    # 选 candidate index 靠前的那个人的标注
    if 'document_tokens' not in e:
        e['document_tokens'] = e['document_text'].split(' ')
    positive_annotations = sorted(
        [a for a in e["annotations"] if has_long_answer(a)],
        key=lambda a: a["long_answer"]["candidate_index"])

    for a in positive_annotations:
        if a["short_answers"]:
            idx = a["long_answer"]["candidate_index"]
            sas = sorted([sa for sa in a['short_answers']], key=lambda sa: sa["start_token"])
            if len(sas) > 1:
                # sas = a['short_answers']
                non_overlapped_sa = [sas[0]]
                for i in range(1, len(sas)):
                    if sas[i]['start_token'] >= sas[i-1]['end_token']:
                        non_overlapped_sa.append(sas[i])
                a['short_answers'] = non_overlapped_sa
                sas = non_overlapped_sa
            # 上面的排序是发现有些先标注了后面的span，后标注前面的span，造成输出的 start_token > end_token
            # 所以我们先做个排序
            # 排序是按照start_token,所以排完还应该检查这几个span是否重叠，这里暂时略过看看会不会报错
            # we -1 to make exclusive end_token to inclusive
            start_token = sas[0]["start_token"]
            end_token = sas[-1]["end_token"] - 1
            assert not _HTML_TOKENS_RE.match(e["document_tokens"][start_token])
            assert not _HTML_TOKENS_RE.match(e["document_tokens"][end_token])
            # todo here we only mark one span for all short answers which is combined and not really correct
            return a, idx, (start_token, end_token), "short"


    for a in positive_annotations:
        idx = a["long_answer"]["candidate_index"]
        start_token = e["long_answer_candidates"][idx]["start_token"]
        end_token = e["long_answer_candidates"][idx]["end_token"] - 1
        assert a["yes_no_answer"] in ("YES", "NO", "NONE")
        while _HTML_TOKENS_RE.match(e["document_tokens"][start_token]):
            start_token += 1
        while _HTML_TOKENS_RE.match(e["document_tokens"][end_token]):
            end_token -= 1
        if a["yes_no_answer"] in ("YES", "NO"):
            return a, idx, (start_token, end_token), a["yes_no_answer"].lower()
        return a, idx, (start_token, end_token), "long"

    return None, -1, (-1, -1), "unknown"


def get_context_tokens(example, span):
    """Returns the text in the example's document in the given token span."""
    token_positions = []
    tokens = []
    for i in range(span["start_token"], span["end_token"]):
        t = example["document_tokens"][i]
        if not _HTML_TOKENS_RE.match(t):
            token_positions.append(i)
            token = t.replace(" ", "")
            if len(token) > 0:
                tokens.append(token)
    return TextSpan(token_positions, tokens)


def get_span_text(s, e, doc):
    """Returns the text in the example's document in the given token span [s,e]. """
    return ' '.join(doc.split(' ')[s:e + 1])


def get_text_and_token_offset_within_candidate(e, candidate_idx, a, b):
    """Converts a token index to the token offset within the !cleaned! candidate. starting from 0!!!! """
    # a,b needs to be inclusive on both ends, the same with returned a1,b1
    # [a, b] and [a1, b1]


    # no answer case
    if (a, b) == (-1, -1):
        return '', -1, -1

    tokens = []
    c = e["long_answer_candidates"][candidate_idx]
    new_token_offset = 0
    a1, b1 = 0, 0
    for i in range(c["start_token"], c["end_token"]):
        t = e["document_tokens"][i]
        if i == a:
            a1 = new_token_offset
        if i == b:
            b1 = new_token_offset
        if not _HTML_TOKENS_RE.match(t):
            token = t.replace(" ", "")
            assert token != ''
            tokens.append(token)
            new_token_offset += 1

    return ' '.join(tokens[a1: b1 + 1]), a1, b1


def clean_html(text):
    """Converts a token index to the token offset within the !cleaned! candidate. starting from 0!!!! """
    tokens = text.split(' ')
    cleaned_tokens = [t.replace(" ", "") for t in tokens if not _HTML_TOKENS_RE.match(t)]
    return ' '.join(cleaned_tokens)


def get_candidate_type(e, idx):
    """Returns the candidate's type: Table, Paragraph, List or Other."""
    c = e["long_answer_candidates"][idx]
    first_token = e["document_tokens"][c["start_token"]]
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
        return "Unknown"


def add_candidate_types_and_positions(e):
    """Adds type and position info to each candidate in the document."""
    # counts = collections.defaultdict(int)
    for idx, c in candidates_iter(e):
        context_type = get_candidate_type(e, idx)
        # if counts[context_type] < FLAGS.max_position:
        #   counts[context_type] += 1
        # c["type_and_position"] = "[%s=%d]" % (context_type, counts[context_type])
        #   TODO
        #   not change here!
        c["type_and_position"] = "[%s]" % context_type


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
    return get_context_tokens(e, e["long_answer_candidates"][idx])


def candidates_iter(e):
    """Yield's the candidates that should not be skipped in an example."""
    for idx, c in enumerate(e["long_answer_candidates"]):
        if should_skip_context(e, idx):
            continue
        yield idx, c


def create_example_from_simplified_jsonl(e):
    """Creates an NQ example from a given line of JSON."""
    e['document_tokens'] = e['document_text'].split(' ')
    add_candidate_types_and_positions(e)
    annotation, ans_idx, (start, end), answer_type = get_first_annotation(e)
    original_ans = e['document_tokens'][start:end + 1]
    logger.debug(original_ans)

    # annotated_idx: index of the first annotated context, -1 if null.
    # annotated_sa: short answer start and end char offsets, (-1, -1) if null.
    if not start <= end:
        print(e)
        print(start)
        print(end)
        assert False
    # print(ans_idx, start, end)
    text, rel_start, rel_end = get_text_and_token_offset_within_candidate(e, ans_idx, start, end)
    answer = {
        "candidate_id": ans_idx,
        # todo here below one line not good?
        "span_text": text,
        "span_start": rel_start,
        "span_end": rel_end,
        "answer_type": answer_type,
        'orig_start': start,
        'orig_end': end
    }
    # print(answer["span_start"], answer["span_end"])
    no_ans = ans_idx == -1

    #   todo  为什么要加这个 -1 context？为什么
    # context_idxs = [-1]
    # context_list = [{"id": -1, "type": get_candidate_type_and_position(e, -1)}]
    # context_list[-1]["text_map"], context_list[-1]["tokens"] = (get_candidate_text(e, -1))
    context_idxs = []
    context_list = []
    for idx, _ in candidates_iter(e):
        # 首先明确每个context和类型
        context = {"id": idx, "type": get_candidate_type_and_position(e, idx)}
        # 然后提取出每个context的文本
        # context之间只用空格连接，这样之后加起来不应该出错。需要注意的是改一下answer span，需要加个offset
        context["text_map"], context["tokens"] = get_candidate_text(e, idx)
        context_idxs.append(idx)
        context_list.append(context)
        assert len(context["text_map"]) == len(context["tokens"])
    # if len(context_list) >= FLAGS.max_contexts:
    #   break
    if not no_ans and ans_idx not in context_idxs:
        return None
    # todo context  remove html?
    # todo important here!

    # Assemble example.
    example = {
        # "name": e['document_title'] if 'document_title' in e else 'Wikipedia',
        # the official simplify_nq script does not preserve doc title, that is why we don't use it.
        "example_id": e["example_id"],
        "document_url": e['document_url'],
        "question_text": e["question_text"],
        "answers": [answer],
        "contexts": e['document_text'],
        # "has_correct_context": annotated_idx in context_idxs
    }
    # todo 不has？？

    single_map = []
    context_tokens = []
    offset = 0
    for context in context_list:
        # context_tokens.extend(["[ContextId=%d]" % context["id"], context["type"]])
        # TODO change here
        # context_tokens.extend(["[Context]", context["type"]])
        # offset += 2
        context_tokens.extend([context["type"]])
        single_map.append(-1)
        offset += 1
        # it's wrong because [Context] shows where the long answer candidate:
        #   at present, we allow only top-level candidates.
        if context["id"] == ans_idx and not no_ans:
            answer["span_start"] += offset
            answer["span_end"] += offset

        # Many contexts are empty once the HTML tags have been stripped, so we
        # want to skip those.
        if context["tokens"]:
            context_tokens.extend(context["tokens"])
            single_map.extend(context["text_map"])
            offset += len(context["tokens"])


    if no_ans:
        expected = ''
    else:
        expected = ' '.join(context_tokens[answer["span_start"]:answer["span_end"] + 1])
    if expected != answer['span_text']:
        print(example)
        print(answer["span_start"], answer["span_end"])
    assert expected == answer["span_text"], (expected, answer["span_text"])

    # todo a different context! here we have removed html tokens and empty contexts, we added special tokens
    # we can call them cleaned_context
    example["contexts"] = " ".join(context_tokens)

    # this is the map from the cleaned context token idx to the original token idx
    example['token_map'] = single_map

    # print('preprocess')
    # print(context_idxs)
    # print(no_ans)
    # print(ans_idx)
    # print(len(context_list))
    # print(answer["span_start"], answer["span_end"])
    return example


# A special token in NQ is made of non-space chars enclosed in square brackets.
_SPECIAL_TOKENS_RE = re.compile(r"^\[[^ ]*\]$", re.UNICODE)
# TODO change here, can be moved into config or add number
# NQ_SPECIAL_TOKEN = {'bos_token': ['[Table]', '[Paragraph]', '[List]', '[Other]']}
TextSpan = collections.namedtuple("TextSpan", "token_positions text")

_HTML_TOKENS_RE = re.compile(r"^<[^ ]*>$", re.UNICODE)

# if __name__ == '__main__':
#     # with open('~/data/nq/v1.0-simplified_nq-dev-all.jsonl', 'r') as f:
#     with open('/home/sunxy-s18/data/nq/entries_dev_0_7829.jsonl', 'w') as fout:
#         with open('/home/sunxy-s18/data/nq/simplified_dev_0_7829.jsonl', 'r') as f:
#             for line in tqdm.tqdm(f):
#                 js = json.loads(line)
#                 entry = create_example_from_simplified_jsonl(js)
#                 fout.write(json.dumps(entry, ensure_ascii=False) + '\n')


def passage_sim_score(window_spans, tokenized_doc, doc, question):
    pos_span = [x for x in window_spans if x[0]]
    neg_span = [x for x in window_spans if not x[0]]
    pos_span.sort(key=lambda x: x[1])
    if not neg_span:
        return pos_span, []
    doc_words = doc.split(' ')
    ref_passage = question
    if pos_span:
        ref_passage += ' ' + ' '.join(doc_words[tokenized_doc[pos_span[0][1]].idx:
                                      tokenized_doc[pos_span[-1][2] - 1].idx + 1])
    neg_passage = [' '.join(doc_words[tokenized_doc[x[1]].idx: tokenized_doc[x[2] - 1].idx + 1])
                   for x in neg_span]
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform([ref_passage] + neg_passage)
    reference_vector = X[0]
    tfidf_matrix = X[1:]
    sim_score = cosine_similarity(reference_vector, tfidf_matrix)
    neg_span = [(sim_score[0][i], x[1], x[2]) for i, x in enumerate(neg_span)]
    neg_span.sort(key=lambda x: x[0],reverse=True)
    return pos_span, neg_span





@DatasetReader.register("bert_joint_nq_simplified")
class BertJointNQReaderSimple(DatasetReader):
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
            downsample_strategy='balanced',
            vocab_len_location="/home/sunxy-s18/data/nq/vocab.len",
            input_type='example',
            output_type='instance',
            keep_all_pos=False,
            lazy=False,
            **kwargs
    ) -> None:
        super().__init__(lazy=lazy, **kwargs)
        self._tokenizer = PretrainedTransformerTokenizer(
            transformer_model_name, add_special_tokens=False, calculate_character_offsets=True
        )
        self._token_indexers = {"tokens": PretrainedTransformerIndexer(transformer_model_name)}
        # self.vocab = PretrainedVocab(self._tokenizer.tokenizer.vocab)
        # logger.info('@'*20+' pretrained vocab of size %d '+'@'*20, len(self.vocab))
        self.length_limit = length_limit
        self.stride = stride
        self.skip_invalid_examples = skip_invalid_examples
        self.neg_pos_ratio = max(1, int(neg_pos_ratio))  # has to be an integer >= 1
        # ==1 要么本身没有负例，要么只留一个
        self.enable_downsample_strategy = enable_downsample_strategy
        self.downsample_strategy = downsample_strategy
        self.max_query_length = max_query_length
        self.non_content_type_id = max(
            self._tokenizer.tokenizer.encode_plus("left", "right", return_token_type_ids=True)[
                "token_type_ids"
            ]
        )
        self.tokenizer = self._tokenizer.tokenizer
        self.vocab_len_location = vocab_len_location
        self.input_type = input_type
        self.output_type = output_type
        self.keep_all_pos = keep_all_pos

        # workaround for a bug in the transformers library
        if "distilbert" in transformer_model_name:
            self.non_content_type_id = 0

    @overrides
    def _read(self, input_paths: str, id=None, show_all=False):
        # if `file_path` is a URL, redirect to the cache
        if self.input_type == 'text_instances':
            instances = self.read_text_instances(input_paths)
            for ins in instances:
                yield ins
            return
        elif self.input_type == 'text_entry':

            instances = self.read_text_entry(input_paths)
            for ins in instances:
                yield ins
            return

        logger.info(input_paths)
        logger.info('#' * 80)
        path_to_dir = cached_path(input_paths)
        # path_to_dir = input_paths
        # not_simplified = 'v1.0-simplified_simplified-nq-train.jsonl.gz' not in input_paths
        not_simplified = 'simplified' not in input_paths
        entries = 'entries' in input_paths

        logger.info("Using simplified version of dataset_reader.!!!!!!")
        logger.info("Reading file at %s which is%s simplified", path_to_dir, ' not' if not_simplified else '')

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

        def elmosim(instance, context, question):
            return 1
            # meta = instance.fields['metadata'].metadata
            # s = meta['context_span_orig'][0]
            # e = meta['context_span_orig'][1]
            # e1 = np.squeeze(ee.embed_sentence(question)[2, 0, :])
            # e2 = np.squeeze(ee.embed_sentence(context[s:e])[2, 0, :])
            # return sp.spatial.distance.cosine(e1, e2)

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
                for i, line in enumerate(input_file):
                    # bigjs = json.loads(input_file.read())
                    # dataset = bigjs['data']
                    # for entry in dataset:
                    js = json.loads(line, object_pairs_hook=collections.OrderedDict)
                    if id and str(js['example_id']) != str(id):
                        continue
                    logging.debug('I was here')
                    if entries:
                        entry = js
                    else:
                        if not_simplified:
                            js = simplify_nq_example(js)
                        entry = create_example_from_simplified_jsonl(js)
                    if not entry:
                        continue
                    doc_info = {
                        'id': entry['example_id'],
                        'doc_url': entry['document_url'],
                        # 'name': entry['name']
                    }
                    instances = self.make_instances(
                        doc_info=doc_info,
                        question=entry["question_text"],
                        answers=entry['answers'],
                        context=entry['contexts'],
                        first_answer_offset=entry['answers'][0]['span_start'],
                        output_type=self.output_type
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

                    if show_all:
                        logger.debug('line')
                        logger.debug(str(line)[:500])
                        logger.debug('entry')
                        logger.debug(str(entry)[:500])
                        logger.debug('instances')
                        logger.debug(str(instances))
                    # todo
                    # if we want to sample examples from multiple files"
                    # instance_this_file += instances_yielded
                    # if instance_this_file > self.max_instance / len(input_files):
                    #     break
                    # break

        _report(neg_cut, neg_kept, pos, yielded_question_count,
                questions_with_more_than_one_instance, instances_yielded_num)

    def myread(self, input_paths: str, id=None, show_all=False):
        ins = self._read(input_paths, id, show_all)
        for i in ins:
            yield i

    def text_instances_js_to_instances(self, js):
        tokenized_context_window = \
            [Token(t[0], idx=t[1], text_id=t[2], type_id=self.non_content_type_id)
             for t in js['tokenized_context_window']]
        tokenized_question = \
            [Token(t[0], idx=t[1], text_id=t[2], type_id=self.non_content_type_id)
             for t in js['tokenized_question']]
        additional_metadata = js['additional_metadata']
        question = js['question']
        context = js['context']
        answers = js['answers']
        window_token_answer_span = js['window_token_answer_span']
        additional_metadata['windows_span'] = js['windows_span']
        instance = self.text_to_instance(
            question,
            tokenized_question,
            context,
            tokenized_context_window,
            answers,
            window_token_answer_span,
            additional_metadata,
            train=True,
        )
        return instance

    def read_text_instances(self, input_paths: str):
        # path_to_dir = cached_path(input_paths)
        input_files = glob.glob(input_paths + '*')
        nopos = False
        if nopos:
            input_files = [path for path in input_files if 'nopos' in path]
        else:
            input_files = [path for path in input_files if 'nopos' not in path]
        input_files.sort()
        logger.info(input_files)
        logger.info('so many files found!')
        shuffle = False  # todo
        logger.info(f"Example entries are shuffled within each file? {shuffle}!")
        cnt = 0
        for path in input_files:
            logger.info("Reading: %s", path)
            # textins = []
            with open(path, 'r') as f:
                for line in f:
                    js = json.loads(line)
                    # textins.append(js)
                    # if js['additional_metadata']['id'] != '1875749308518572232':
                    #     continue
                # if shuffle:
                #     random.shuffle(textins)
                # for js in textins:
                    instance = self.text_instances_js_to_instances(js)
                    yield instance
                    cnt += 1
        logger.info(f'Instance yielded: {cnt}')

    def select_span_by_strategy(self, pos_span, neg_span, strategy='best'):
        selected = []
        if pos_span:
            if self.keep_all_pos:
                selected.extend(pos_span)
            else:
                selected.append(pos_span[0])
        if not neg_span:
            selected = [(x[1], x[2]) for x in selected]
            return selected
        # todo note here neg_pos_ratio is no longer a ratio, it is just the max neg instance per example
        num_neg_to_keep = self.neg_pos_ratio
        neg_span = [x for x in neg_span if x[1]<x[2]]
        if strategy == 'best':
            selected.extend(neg_span[:num_neg_to_keep])
        elif strategy == 'worst':
            selected.extend(neg_span[-num_neg_to_keep:])
        elif len(neg_span) <= num_neg_to_keep:
            selected.extend(neg_span)
        elif strategy == 'random':
            selected.extend(random.sample(neg_span, num_neg_to_keep))
        elif strategy == 'balanced':
            selected.append(neg_span[0])
            selected.extend(random.sample(neg_span[1:], num_neg_to_keep - 1))
        selected = [(x[1], x[2]) for x in selected]
        return selected

    def text_entry_js_to_instances(self, js) -> List[Instance]:
        instances = []
        tokenized_context = \
            [Token(t[0], idx=t[1], text_id=t[2], type_id=self.non_content_type_id)
             for t in js['tokenized_context']]
        tokenized_question = \
            [Token(t[0], idx=t[1], text_id=t[2], type_id=self.non_content_type_id)
             for t in js['tokenized_question']]
        additional_metadata = js['additional_metadata']
        question = js['question']
        context = js['context']
        answers = js['answers']
        token_answer_span = js['token_answer_span']
        selected_spans = self.select_span_by_strategy(js['pos_span'], js['neg_span'],
                                                      strategy=self.downsample_strategy)
        for span in selected_spans:
            tokenized_context_window = tokenized_context[span[0]:span[1]]
            if len(tokenized_context_window) == 0:
                print("wtf?")
            window_token_answer_span = (
                token_answer_span[0] - span[0],
                token_answer_span[1] - span[0],
            )
            if any(i < 0 or i >= (span[1]-span[0]) for i in window_token_answer_span):
                # The answer is not contained in the window.
                window_token_answer_span = None
            instance = self.text_to_instance(
                question,
                tokenized_question,
                context,
                tokenized_context_window,
                answers,
                window_token_answer_span,
                additional_metadata,
                train=True,
            )
            instances.append(instance)
        return instances

    def read_text_entry(self, input_paths: str):
        input_files = glob.glob(input_paths + '*')
        nopos = False
        if nopos:
            input_files = [path for path in input_files if 'nopos' in path]
        else:
            input_files = [path for path in input_files if 'nopos' not in path]
        input_files.sort()
        logger.info(input_files)
        logger.info('so many files found!')
        shuffle = False  # todo
        logger.info(f"Example entries are shuffled within each file? {shuffle}!")
        logger.info('Reading text entries!')
        logger.info(f'Negative instances are selected with the strategy: {self.downsample_strategy}')
        cnt = 0
        for path in input_files:
            logger.info("Reading: %s", path)
            with open(path, 'r') as f:
                for line in f:
                    js = json.loads(line)
                    if not js:
                        continue
                    instances = self.text_entry_js_to_instances(js)
                    for instance in instances:
                        yield instance
                        cnt += 1
        logger.info(f'Instance yielded: {cnt}')

    def generate_text_instances_from_simplified_js_line(self, line):
        js = json.loads(line, object_pairs_hook=collections.OrderedDict)
        # print(js)
        entry = create_example_from_simplified_jsonl(js)
        if not entry:
            return
        doc_info = {
            'id': entry['example_id'],
            'doc_url': entry['document_url'],
        }
        instances = self.make_instances(
            doc_info=doc_info,
            question=entry["question_text"],
            answers=entry['answers'],
            context=entry['contexts'],
            first_answer_offset=entry['answers'][0]['span_start'],
            output_type='text_entry'
        )
        return instances

    def write_text_instance(self, input_paths: str, output_paths: str, max_to_write=None):
        # if `file_path` is a URL, redirect to the cache
        logger.info(input_paths)
        logger.info('#' * 80)
        path_to_dir = cached_path(input_paths)
        # path_to_dir = input_paths
        not_simplified = 'v1.0-simplified_simplified-nq-train.jsonl.gz' not in input_paths
        entries = 'entries' in input_paths

        logger.info("Using simplified version of dataset_reader.!!!!!!")
        logger.info("Reading file at %s which is%s simplified", path_to_dir, ' not' if not_simplified else '')

        """Read a NQ json file into a list of NqEntry."""

        input_files = glob.glob(path_to_dir + '*')

        def _open(file_path):
            if file_path.endswith(".gz"):
                return gzip.GzipFile(fileobj=open(file_path, "rb"))
            else:
                return open(file_path, 'r', encoding='utf-8')

        neg_cut = 0
        neg_kept = 0
        pos = 0
        report_step = max(1, self.max_instances // max(5, len(input_files))) if self.max_instances is not None else 100

        # instance 的数目每过一个 report_step，打印一次 instance 的正负比例
        total_ins = []
        for path in input_files:
            logger.info("Reading: %s", path)
            # instance_this_file = 0
            with _open(path) as input_file:
                for line in tqdm.tqdm(input_file):
                    instances = self.generate_text_instances_from_simplified_js_line(line)
                    total_ins.extend(instances)
                    neg_kept += 1
                    if len(instances) == 2:
                        pos += 1
                    elif len(instances) != 1:
                        print('CAO' * 20)
                        print(len(instances))
                        # assert False

                    if max_to_write:
                        if len(total_ins) > max_to_write:
                            break

        with open(output_paths, 'w') as f:
            for ins in total_ins:
                f.write(json.dumps(ins, ensure_ascii=False) + '\n')

    def make_instances(
            self,
            doc_info: Dict,
            question: str,
            answers: List[Dict],
            context: str,
            first_answer_offset: Optional[int],
            output_type: str,
            train=False
    ) -> Iterable[Instance]:
        # tokenize context by spaces first, and then with the wordpiece tokenizer
        # For RoBERTa, this produces a bug where every token is marked as beginning-of-sentence. To fix it, we
        # detect whether a space comes before a word, and if so, add "a " in front of the word.
        def tokenize_slice(text_to_tokenize, space_in_front) -> List[Token]:
            # text_to_tokenize = context[start:end]
            # wordpieces = self._tokenizer.tokenize(text_to_tokenize)
            # print(wordpieces, wordpieces[0].idx)
            # return wordpieces
            if space_in_front:
                prefix = "a "  # must end in a space, and be short so we can be sure it becomes only one token
                wordpieces = self._tokenizer.tokenize(prefix + text_to_tokenize)
                # print(wordpieces, wordpieces[1].idx)
                for wordpiece in wordpieces:
                    if wordpiece.idx is not None:
                        # todo change
                        wordpiece.idx -= 1
                return wordpieces[1:]
            else:
                return self._tokenizer.tokenize(text_to_tokenize)

        instance_generated = []
        # print(context)
        tokenized_context = []
        (token_answer_span_start, token_answer_span_end) = (-1, -1)

        # print(context)
        split_context = context.split(' ')
        for i, token_text in enumerate(split_context):
            wordpiece_tokens = []
            if _SPECIAL_TOKENS_RE.match(token_text):  # or _HTML_TOKENS_RE.match(token_text):
                # # todo replace html tokens
                # if token_text not in self.tokenizer.added_tokens_encoder:
                #     self.tokenizer.add_tokens(token_text)
                #     id = self.tokenizer.convert_tokens_to_ids(token_text)
                #     new_special_token = Token(token_text, idx=i,
                #                               text_id=id,     # text_id=1,
                #                               type_id=self.non_content_type_id)
                #     wordpiece_tokens.append(new_special_token)
                # else:
                id = self.tokenizer.convert_tokens_to_ids(token_text)
                if id == 100:
                    print(token_text)
                    assert False
                special_token = Token(token_text, idx=i,
                                      text_id=id,
                                      type_id=self.non_content_type_id)
                wordpiece_tokens.append(special_token)
            else:
                wordpiece_tokens = tokenize_slice(token_text, i != 0)
                for wordpiece in wordpiece_tokens:
                    wordpiece.idx = i

            if len(wordpiece_tokens) == 0:
                wordpiece_tokens = [Token(
                    self._tokenizer.tokenizer.unk_token,
                    idx=i,
                    text_id=self._tokenizer.tokenizer.unk_token_id,
                    type_id=self.non_content_type_id, )]

            # map the answer span
            if first_answer_offset:
                if i == answers[0]['span_start']:
                    token_answer_span_start = len(tokenized_context)
                if i == answers[0]['span_end']:
                    token_answer_span_end = len(tokenized_context) + len(wordpiece_tokens) - 1

            tokenized_context.extend(wordpiece_tokens)

        if first_answer_offset and first_answer_offset >= 0:
            assert (token_answer_span_start, token_answer_span_end) != (-1, -1)

        # Tokenize the question
        tokenized_question = self._tokenizer.tokenize(question)
        tokenized_question = tokenized_question[: self.max_query_length]
        for token in tokenized_question:
            token.type_id = self.non_content_type_id
            token.idx = None

        # Stride over the context, making instances
        # Sequences are [CLS] question [SEP] [SEP] context [SEP], hence the - 4 for four special tokens.
        # This is technically not correct for anything but RoBERTa, but it does not affect the scores.
        # my_special_token = ['[NoLongAnswer']
        # space_for_context = self.length_limit - len(tokenized_question) - 4 - len(my_special_token)
        space_for_context = self.length_limit - len(tokenized_question) - 4
        stride_start = 0
        first_neg = True
        first_pos = True
        additional_metadata = doc_info
        window_spans = []
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

            if output_type == 'text_instance':
                if first_pos and window_token_answer_span is not None \
                        or first_neg and window_token_answer_span is None:
                    if window_token_answer_span is None:
                        first_neg = False
                    else:
                        first_pos = False
                    additional_metadata['context_span_orig'] = (stride_start, stride_start + space_for_context)
                    text_instance = {
                        "question": question,
                        "answers": answers,  # todo
                        "context": context,
                        "windows_span": (stride_start, stride_start + space_for_context),  # [a,b)
                        "window_token_answer_span": window_token_answer_span,
                        "additional_metadata": additional_metadata,
                        "tokenized_question": [(t.text, t.idx, t.text_id) for t in tokenized_question],
                        "tokenized_context_window": [(t.text, t.idx, t.text_id) for t in tokenized_context_window],
                    }
                    instance_generated.append(text_instance)
                if len(instance_generated) == 2 or len(instance_generated) == 1 and first_answer_offset < 0:
                    break

            elif output_type == 'instance':
                if not self.skip_invalid_examples or window_token_answer_span is not None:
                    additional_metadata['context_span_orig'] = (stride_start, stride_start + space_for_context)
                    instance = self.text_to_instance(
                        question,
                        tokenized_question,
                        context,
                        tokenized_context_window,
                        answers,
                        window_token_answer_span,
                        additional_metadata,
                        train=train,
                    )
                    # yield instance
                    instance_generated.append(instance)
            elif output_type == 'text_entry':
                # in text_entry format, we don't care if it is a positive or negative instance
                window_spans.append((int(window_token_answer_span != None),
                                     stride_start,
                                     min(stride_start + space_for_context, len(tokenized_context))
                                     ))
            else:
                raise Exception('Unknown output_type')

            # stride in the paper and all other places
            length = len(tokenized_context) - stride_start
            length = min(length, space_for_context)
            if stride_start + length >=len(tokenized_context):
                break
            stride_start += min(length, self.stride)

            # # original stride definition
            # stride_start += space_for_context
            # if stride_start >= len(tokenized_context):
            #     break
            # stride_start -= self.stride


        if output_type == 'text_instance' or output_type == 'instance':
            return instance_generated
        elif output_type == 'text_entry':
            pos_span, neg_span = passage_sim_score(window_spans, tokenized_context, context, question)
            tokenized_entry = {
                "question": question,
                "answers": answers,  # todo
                "context": context,
                "token_answer_span": (token_answer_span_start, token_answer_span_end),
                "additional_metadata": additional_metadata,
                "tokenized_question": [(t.text, t.idx, t.text_id) for t in tokenized_question],
                "tokenized_context": [(t.text, t.idx, t.text_id) for t in tokenized_context],
                "pos_span": pos_span,
                "neg_span": neg_span
            }
            return tokenized_entry

    @overrides
    def text_to_instance(
            self,  # type: ignore
            question: str,
            tokenized_question: List[Token],
            context_text: str,
            window_context_wordpiece_tokens: List[Token],
            answers: List[Dict],
            windows_token_answer_span: Optional[Tuple[int, int]],
            additional_metadata: Dict[str, Any] = None,
            train=True,
    ) -> Instance:
        '''

        :param question:
        :param tokenized_question:
        :param context_text:
        :param window_context_wordpiece_tokens:
        :param answers:
        :param windows_token_answer_span:
        :param additional_metadata:
        :return:
        根据答案在 窗口上下文 tokenized_window_context 中的位置span 计算答案在instance中的span
            就是加上前面的 cls question sep sep
        如果答案不存在当前 instance 的 window 中,
            span 标记为 [NoLongAns]所在的位置， 修改 ans_type ans_text
        metadata 存放关于 example的信息
            doc_info:
            start_of_context: start of context within question_with_context
            window_context_wordpiece_tokens：
                其中的每一个 token.idx 用于对应到 context_tokens 中的位置
            context_text 全文，cleaned版本，用空格拆开后得到 context_tokens，每一个都是 whole world token， 
                model 从这里找到对应的答案文本，用空格拼接
                    如果预测的是无答案怎么办？ - model将其替换为 ‘’
            从 wordpiece token 直接拼接很难的，要 sanitize，要知道什么地方插空格。
            ans_text 标准答案文本
            ans_type 从gpu里拉出来，不好！
            
            我们用的是 span acc， 适用于指针网络
            span based metrics适用于bio。两者等价

        '''
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
        # noans_token = Token(
        #     '[NoLongAnswer]',
        #     idx=None,
        #     text_id=9,
        #     type_id=self.non_content_type_id,
        # )

        question_with_context_field = TextField(
            (
                    [cls_token]
                    + tokenized_question
                    + [sep_token, sep_token]
                    # + [sep_token]
                    # + [noans_token]
                    + window_context_wordpiece_tokens
                    + [sep_token]
            ),
            self._token_indexers,
        )

        # qwc = ' '.join([t.text for t in ([cls_token]
        #                                  + tokenized_question
        #                                  + [sep_token, sep_token]
        #                                  + window_context_wordpiece_tokens
        #                                  + [sep_token])])
        fields["question_with_context"] = question_with_context_field
        # start_of_context = 1 + len(tokenized_question) + 2 + 1
        start_of_context = 1 + len(tokenized_question) + 2

        # make the answer span
        if windows_token_answer_span is not None:
            assert all(i >= 0 for i in windows_token_answer_span)
            assert windows_token_answer_span[0] <= windows_token_answer_span[1]
            # TODO 可以加一个答案内容匹配的assert
            ans_span = (windows_token_answer_span[0] + start_of_context,
                        windows_token_answer_span[1] + start_of_context)
            fields["answer_span"] = SpanField(
                ans_span[0],
                ans_span[1],
                question_with_context_field,
            )
            # asset checks
            ans_type = answers[0]['answer_type']
            answer_text = answers[0]['span_text']
            ans_span_context = (ans_span[0] - start_of_context, ans_span[1] - start_of_context)
            t1 = window_context_wordpiece_tokens[ans_span_context[0]]
            t2 = window_context_wordpiece_tokens[ans_span_context[1]]
            to_predict = ' '.join(context_text.split(' ')[t1.idx: t2.idx + 1])
            assert to_predict == answer_text, (to_predict, answer_text, question)
        else:
            # We have to put in something even when we don't have an answer, so that this instance can be batched
            # together with other instances that have answers.
            # TODO change to cls token here？ yes!
            fields["answer_span"] = SpanField(0, 0, question_with_context_field)
            ans_type = 'unknown'
            answer_text = ""

        answer_type_idx = make_nq_answer(ans_type)
        fields['answer_type'] = LabelField(answer_type_idx, skip_indexing=True)

        # make the context span, i.e., the span of text from which possible answers should be drawn
        fields["context_span"] = SpanField(
            start_of_context, start_of_context + len(window_context_wordpiece_tokens) - 1, question_with_context_field
        )
        # todo change the mask above to add -1

        # make the metadata if train
        if not train:
            first_part_context = ' '.join(context_text.split(' ')[window_context_wordpiece_tokens[0].idx
                                                                  :window_context_wordpiece_tokens[-1].idx + 1])
            metadata = {
                "start_of_context": start_of_context,
                "context": first_part_context,
                "window_context_wordpiece_tokens": window_context_wordpiece_tokens,
                "answer_text": answer_text,
                # 以上对计算 em-f1 有用。以下用于 debug
                # "answer_type_idx": answer_type_idx,
                # "question": question,
                "answers": answers,
                # "qwc": qwc
            }
            # TODO add doc url & ID
            if additional_metadata is not None:
                metadata.update(additional_metadata)
            fields["metadata"] = MetadataField(metadata)
        # todo remove below
        if ans_type == 'long' or ans_type == 'short':
            if answer_text == '':
                print(fields)
        return Instance(fields)

