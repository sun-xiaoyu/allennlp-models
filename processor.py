import logging
import json
import os
import sys
import gzip
import tqdm
from nq import NQPredictor
from nq.nq_text_utils import simplify_nq_example
from nq.dataset_readers.bert_joint_nq_simplified import create_example_from_simplified_jsonl, make_nq_answer
from allennlp_models.rc.transformer_qa import TransformerQAPredictor

from sklearn.metrics import confusion_matrix

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

logger = logging.getLogger(__name__)


class Processor(object):
    """
    这个类的目的是调用 predictor，预测单个例子或者一个测试数据文件 jsonl。根据 id 整合结果。
        可选开关
            1. （多个输入的情况下）加入 results 的统计，如 confusion matrix
            2. 对于NQ：
                a 找到对应的长答案
                b 生成官网所需形式的prediction
                b 调用官网 nq_eval 得到官方分数。
    """

    def __init__(self, model_path):
        prefix = '/home/sunxy-s18/data/'
        assert model_path.startswith(prefix)
        self.model_path = model_path
        self.model_name = model_path[len(prefix):].strip('/')
        logger.info(f"Using model: {self.model_name}")
        self.prediction_output_path = f'/home/sunxy-s18/data/nq/prediction_{self.model_name}.jsonl'
        logger.info(f"Predition output path will be: {self.model_name}")
        self.dev_data_path = ''
        self.results = None

    def predict_and_process(self, data_path=None, overwrite=False, unique_id=None):
        if data_path is None:
            data_path = self.dev_data_path
        if not os.path.exists(self.prediction_output_path) or overwrite:
            logger.warning('Generating prediction')
            results = self.generate_prediction_from_data(data_path, unique_id=unique_id)
            logger.info(f'Writing to {self.prediction_output_path}')
            with open(self.prediction_output_path, 'w') as fout:
                for res in results:
                    fout.write(json.dumps(res, ensure_ascii=False) + '\n')
        else:
            logger.warning("Prediction file already exists! We reload it.")
            results = []
            with open(self.prediction_output_path, 'r') as f:
                for line in f:
                    results.append(json.loads(line))
        self.results = results

    def confusion(self):
        if not self.results:
            self.predict_and_process()
        print('Confusion matrix:')
        y_true = [x['ans_type_gold'] for x in self.results]
        y_pred = [x['ans_type_pred'] for x in self.results]
        print(confusion_matrix(y_true, y_pred))

        if 'answer_type_pred_cls' in self.results[0]:
            print('Prediction with [CLS] token:')
            y_true = [x['ans_type_gold'] for x in self.results]
            y_pred = [x['answer_type_pred_cls'] for x in self.results]
            print(confusion_matrix(y_true, y_pred))

    def generate_prediction_from_data(self, data_path, unique_id):
        raise NotImplementedError


class SquadProcessor(Processor):
    def __init__(self, model_path):
        super().__init__(model_path)
        self.predictor = None
        self.dev_data_path = '/home/sunxy-s18/data/squad2.0/dev-v2.0.json'

    def generate_prediction_from_data(self, data_path, max_ids=None):
        # unique id currently not supported for squad dataset!
        results = []
        if not self.predictor:
            self.predictor = TransformerQAPredictor.from_path(self.model_path,
                                                              predictor_name='transformer_qa',
                                                              cuda_device=CUDA_DEVICE)
        with open(data_path, 'r') as f:
            dataset_json = json.load(f)
            dataset = dataset_json["data"]
            logger.info("Reading the dataset at:", self.dev_data_path)
            for article in tqdm.tqdm(dataset):
                for paragraph_json in article["paragraphs"]:
                    context = paragraph_json["context"]
                    for question_answer in paragraph_json["qas"]:
                        question_answer['context'] = context
                        res = self.predictor.predict_json(question_answer)
                        res['answers'] = question_answer['answers']
                        res['ans_type_gold'] = len(res['answers']) > 0
                        res['ans_type_pred'] = res['best_span_str'] != ''
                        results.append(res)

        logger.info(f'length of results: {len(results)}')
        return results


def compute_predictions(candidates, token_map, result):
    """Converts an example into an NQEval object for evaluation."""
    # 从来自多个窗口的候选span中排序打分，得出最好的short span，并且从list中找到包含他的long span
    # 这里的start end的选择是从整篇文章中来的。 而我自己的做法是，每个window选出一个最好的start,end 然后window之间比较    predictions = []
    score = result['best_span_scores']
    short_span_orig = result['best_span_orig']
    long_span = -1, -1
    if short_span_orig[0] == -1 or short_span_orig[1] == -1:
        short_span = -1, -1
    else:
        short_span = (token_map[short_span_orig[0]], token_map[short_span_orig[1]])
        for c in candidates:
            start = short_span[0]
            end = short_span[1]
            if c["top_level"] and c["start_token"] <= start and c["end_token"] >= end:
                long_span = c["start_token"], c["end_token"]
                break
    nq_eval = {
        "example_id": result['id'],
        "long_answer": {
            "start_token": long_span[0],
            "end_token": long_span[1],
            "start_byte": -1,
            "end_byte": -1
        },
        "long_answer_score": score,
        "short_answers": [{
            "start_token": short_span[0],
            "end_token": short_span[1],
            "start_byte": -1,
            "end_byte": -1
        }],
        "short_answers_score": score,
        "yes_no_answer": "NONE"
    }
    return nq_eval


class NqProcessor(Processor):
    def __init__(self, model_path):
        super().__init__(model_path)
        self.predictor = None
        # self.dev_data_path = '/home/sunxy-s18/data/nq/v1.0-nq-dev-all.jsonl'
        self.dev_data_path = '/home/sunxy-s18/data/nq/simplified_nq_dev_all_7830.jsonl'

    def generate_prediction_from_data(self, data_path, max_ids=None):
        if not self.predictor:
            self.predictor = NQPredictor.from_path(self.model_path, predictor_name='nq',
                                                   cuda_device=CUDA_DEVICE)
        results = []
        logger.info("Reading: %s", data_path)
        not_simplified = 'simplified' not in data_path
        with _open(data_path) as input_file:
            for line in tqdm.tqdm(input_file):
                js = json.loads(line)
                logging.debug('I was here')
                if not_simplified:
                    js = simplify_nq_example(js)
                entry = create_example_from_simplified_jsonl(js)
                if not entry:
                    continue
                doc_info = {
                    'id': entry['example_id'],
                    'doc_url': entry['document_url'],
                }
                entry['doc_info'] = doc_info
                res = {'id': entry['example_id'],
                       'doc_url': entry['document_url'],
                       'question_text': entry['question_text'],
                       'answers': entry['answers'],
                       'ans_type_gold': make_nq_answer(entry['answers'][0]['answer_type'])}
                res.update(self.predictor.predict_json(entry))
                res['ans_type_pred'] = int(res['best_span_str'] != '')
                res['nq_eval'] = compute_predictions(js["long_answer_candidates"],
                                                          entry["token_map"], res)
                results.append(res)
                if max_ids and len(results) > max_ids:
                    break

        logger.info(f'length of results: {len(results)}')
        return results


def _open(file_path):
    if file_path.endswith(".gz"):
        return gzip.GzipFile(fileobj=open(file_path, "rb"))
    else:
        return open(file_path, 'r', encoding='utf-8')


CUDA_DEVICE = 1
if __name__ == '__main__':
    overwrite = False
    args = sys.argv
    if '-o' in args:
        overwrite = True
    unique_id = None
    if '--max' in args:
        unique_id_pos = args.index('--max')
        unique_id = int(args[unique_id_pos + 1])

    # test squad2.0 ok
    # model_name = 'squad2mod_noans_bert_0307'
    # model_path = f'/home/sunxy-s18/data/{model_name}/'
    # processor = SquadProcessor(model_path)
    # processor.predict_and_process(overwrite=False)
    # processor.confusion()

    # test nq
    # model_name = 'nq_bigfinetune_0308_32_3e5'
    model_name = 'nq_bigfinetune_0310_24_3e5'
    # model_name = 'nq_bigfinetune_hasans_bert_0307_3e5_8'
    model_path = f'/home/sunxy-s18/data/{model_name}/'
    processor = NqProcessor(model_path)
    processor.predict_and_process(overwrite=overwrite, unique_id=unique_id)
    processor.confusion()
    for res in processor.results:
        if res['ans_type_pred'] != 0 or res['ans_type_pred'] != 0:
            # print(json.dumps(res, indent=2))
            sa = res['nq_eval']['short_answers'][0]
            ga = res['answers'][0]
            if ga['orig_start'] != sa['start_token'] or ga['orig_end'] != sa['end_token']:
                print(res['id'])
                print(res['doc_url'])
                print('-' * 50)
                print('Q:', res['question_text'])
                print('-' * 50)
                print('Prediction', sa['start_token'], sa['end_token'], res['ans_type_pred'], res['ans_type_pred_cls'])
                print('~~~', res['best_span_str'])
                print('-' * 30)
                print('Gold:', ga['orig_start'], ga['orig_end'], res['ans_type_gold'])
                print('~~~', ga['span_text'])
                print('=' * 70)
                print('\n')
