import logging
import json
import os
import sys
import gzip
import tqdm
from multiprocessing import Pool
import numpy as np
from nq import NQPredictor, QatypePredictor
from nq.nq_text_utils import simplify_nq_example
from nq.dataset_readers.bert_joint_nq_simplified import create_example_from_simplified_jsonl, make_nq_answer
from allennlp_models.rc.transformer_qa import TransformerQAPredictor

from sklearn.metrics import confusion_matrix, accuracy_score

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

logger = logging.getLogger(__name__)

DEBUGGGGGGGGGG = False
CUDA_DEVICE = 0
YN_TYPE_DICT = {2: 'NONE', 3: 'YES', 4: 'NO'}


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
        # we add _new_ to the following path because we use the new predictor!
        self.prediction_output_path = f'/home/sunxy-s18/data/nq/prediction_new_{self.model_name}.jsonl'
        logger.info(f"Predition output path will be: {self.prediction_output_path}")
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
        debug = DEBUGGGGGGGGGG
        if debug:
            _ = self.generate_prediction_from_data(data_path, unique_id=unique_id, debug=True)
            with open(self.prediction_output_path, 'w') as fout:
                for res in results:
                    fout.write(json.dumps(res, ensure_ascii=False) + '\n')

    def confusion(self):
        if not self.results:
            self.predict_and_process()
        print(self.model_name)
        print('Confusion matrix:')
        y_true = [x['ans_type_gold'] for x in self.results]
        y_pred = [x['ans_type_pred'] for x in self.results]
        print(confusion_matrix(y_true, y_pred))
        print(f'Gold: {min(y_true)}~{max(y_true)}')
        print(f'Pred: {min(y_pred)}~{max(y_pred)}')

        if 'ans_type_pred_cls' in self.results[0]:
            print('Prediction with [CLS] token:')
            y_true = [x['ans_type_gold'] for x in self.results]
            y_pred = [x.get('ans_type_pred_cls',0) for x in self.results]
            print(confusion_matrix(y_true, y_pred))
        print('')

    def generate_prediction_from_data(self, data_path, unique_id, debug=False):
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


def compute_predictions(candidates, token_map, result, yesno:str = "NONE"):
    """Converts an example into an NQEval object for evaluation."""
    # 从来自多个窗口的候选span中排序打分，得出最好的short span，并且从list中找到包含他的long span
    # 这里的start end的选择是从整篇文章中来的。 而我自己的做法是，每个window选出一个最好的start,end 然后window之间比较    predictions = []
    score = result['best_span_scores']
    short_span_orig = result['best_span_orig']
    long_span = -1, -1
    if short_span_orig[0] == -1 or short_span_orig[1] == -1:
        short_span = -1, -1
    else:
        assert short_span_orig[0] <= short_span_orig[1], (short_span_orig[0], short_span_orig[1])
        s = short_span_orig[0]
        e = short_span_orig[1]
        while token_map[s] == -1 and s + 1 < len(token_map):
            s += 1
        while token_map[e] == -1 and e - 1 >= 0:
            e -= 1
        if token_map[s] == -1 or token_map[e] == -1:
            short_span = -1, -1
            logger.warning(f"Can't map token back to original uncleaned text! {result['id']}, {short_span_orig} ")
        else:
            short_span = (token_map[s], token_map[e] + 1)
            assert short_span[0] < short_span[1], (short_span[0], short_span[1])
            for c in candidates:
                start = short_span[0]
                end = short_span[1]
                if c["top_level"] and c["start_token"] <= start and c["end_token"] >= end:
                    long_span = c["start_token"], c["end_token"]
                    if long_span[0] == short_span[0] - 1 and long_span[1] == short_span[1] + 1:
                        short_span = -1, -1
                        # print("it's working")
                    assert long_span[0] < long_span[1], (long_span[0], long_span[1])
                    break
    if yesno != 'NONE':
        short_span = -1, -1
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
        "yes_no_answer": yesno,
    }
    return nq_eval


def no_ans_nq_eval(id):
    return {'id': id,
            'nq_eval':{
                "example_id": id,
                "long_answer": {
                    "start_token": -1,
                    "end_token": -1,
                    "start_byte": -1,
                    "end_byte": -1
                },
                "long_answer_score": -9999,
                "short_answers": [{
                    "start_token": -1,
                    "end_token": -1,
                    "start_byte": -1,
                    "end_byte": -1
                }],
                "short_answers_score": -9999,
                "yes_no_answer": "NONE"
            },
            "ans_type_gold": 5,
            "ans_type_pred": 0
            }


class NqProcessor(Processor):
    def __init__(self, model_path, predict_yesno = True):
        super().__init__(model_path)
        self.predictor = None
        # self.dev_data_path = '/home/sunxy-s18/data/nq/v1.0-nq-dev-all.jsonl'
        self.dev_data_path = '/home/sunxy-s18/data/nq/simplified_nq_dev_all_7830.jsonl'
        self.dev_textentry_path = '/home/sunxy-s18/data/nq/train/textentry_new_dev_all.jsonl'
        # we use the new predictor! Always try to predict a plausible answer!
        # Don't predict noanswer unless noanswer in every window
        self.official_prediction_path = f'/home/sunxy-s18/data/nq/official_predictions_new_{self.model_name}'
        # length of results: 7673
        self.predict_yesno = predict_yesno
        self.yn_processor = None
        self.yesno_predictor_path = '/home/sunxy-s18/data/yesno_0321'
        self.id = 0
        if self.predict_yesno:
            self.prediction_output_path = self.prediction_output_path[:-6] + '_yesno.jsonl'
            self.official_prediction_path = self.official_prediction_path + '_yesno'
            logger.info(f"Actually, prediction output path will be: {self.prediction_output_path}")


    def generate_prediction_from_data(self, data_path, unique_id=None, debug=False):
        if not debug:
            if not self.predictor:
                self.predictor = NQPredictor.from_path(self.model_path, predictor_name='nq',
                                                       cuda_device=CUDA_DEVICE)
            if self.predict_yesno:
                if not self.yn_processor:
                    self.yn_processor = QatypeProcessor(self.yesno_predictor_path)
                    self.yn_processor.load()
        results = []
        logger.info("Reading: %s", data_path)

        # cnt = 0
        logger.info(f"prediction output path will be: {self.prediction_output_path}")
        with _open(data_path) as input_file:
            with open(self.dev_textentry_path) as f:
                textentries = f.readlines()
                lines = input_file.readlines()
                # with Pool(2) as p:
                #     results = list(tqdm.tqdm(p.imap(self.process_one, lines), total=7830))
                for i, line in enumerate(tqdm.tqdm(lines)):
                    results.append(self.process_one(line, textentries[i]))
                    # print(results[-1]['id'], results[-1]['nq_eval'])
                    if unique_id and len(results) > unique_id:
                        break


        logger.info(f'length of results: {len(results)}')
        return results

    def predict_one(self, question, html):
        if not self.predictor:
            self.load()
        entry = {
            'doc_info': {
                'id': self.id,
                'doc_url': 'url'
            },
            "question_text": question,
            "contexts": html,
        }
        res = self.predictor.predict_json(entry)
        self.id += 1
        return res
        # return "predict_one_output"

    def save_official_prediction(self):
        if not self.results:
            self.predict_and_process()
        official_predictions = {
            'predictions':[x['nq_eval'] for x in self.results]
        }
        # if not os.path.exists(self.official_prediction_path):
        with open(self.official_prediction_path, 'w') as fout:
            fout.write(json.dumps(official_predictions, ensure_ascii=False))
        logger.info('official prediction saved at:' + self.official_prediction_path)

    def load(self):
        if not self.predictor:
            self.predictor = NQPredictor.from_path(self.model_path, predictor_name='nq',
                                                   cuda_device=CUDA_DEVICE)
        if self.predict_yesno:
            if not self.yn_processor:
                self.yn_processor = QatypeProcessor(self.yesno_predictor_path)
                self.yn_processor.load()

    def process_one(self, line, textentry):
        js = json.loads(line.strip('\n'))
        textentry = json.loads(textentry.strip('\n'))
        logging.debug('I was here')
        # todo we use simplified data
        # if not_simplified:
        #     js = simplify_nq_example(js)
        entry = create_example_from_simplified_jsonl(js)
        if not entry:
            res = no_ans_nq_eval(js['example_id'])
            return res
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
        # res.update(self.predictor.predict_json(entry))
        res.update(self.predictor.predict_one_text_entry(textentry))
        res['ans_type_pred'] = int(res['best_span_str'] != '')
        yesno = 'NONE'
        if self.predict_yesno:
            yesno = self.yn_processor.predict_yn(res['question_text'], res['best_span_str'])
        res['nq_eval'] = compute_predictions(js["long_answer_candidates"],
                                             entry["token_map"], res, yesno=yesno)

        return res



def _open(file_path):
    if file_path.endswith(".gz"):
        return gzip.GzipFile(fileobj=open(file_path, "rb"))
    else:
        return open(file_path, 'r', encoding='utf-8')


class QatypeProcessor(Processor):
    def __init__(self, model_path):
        super().__init__(model_path)
        self.predictor = None
        self.dev_data_path = '/home/sunxy-s18/data/nq/qatype_dev.jsonl'
        if 'yesno' in model_path:
            self.dev_data_path = '/home/sunxy-s18/data/nq/yesno_dev.jsonl'
        self.with_answer = False
        if 'noans' not in model_path:
            self.with_answer = True

    def generate_prediction_from_data(self, data_path, unique_id, debug=False):
        # unique id currently not supported for squad dataset!
        results = []
        if not self.predictor:
            self.predictor = QatypePredictor.from_path(self.model_path,
                                                            predictor_name='qatype',
                                                            cuda_device=CUDA_DEVICE)
        with open(data_path, 'r') as f:
            dataset_json = json.load(f)
            dataset = dataset_json["data"]
            logger.info("Reading the dataset at:" + self.dev_data_path)
            for example in tqdm.tqdm(dataset):
                label = example["answer_type"]
                question = example["question"]
                answer = example["answer_text"]
                # no no-answer!
                if label == 0:
                    continue
                if not self.with_answer:
                    if label == 4:
                        label = 3
                    res = self.predictor.predict_qa(question, None)
                else:
                    res = self.predictor.predict_qa(question, answer)
                res['question'] = question
                res['answer_text'] = answer
                res['ans_type_gold'] = label
                res['ans_type_pred'] = int(res["label"])
                del res['label']
                results.append(res)
                # if len(results) > 500:
                #     break
                if unique_id and len(results) > unique_id:
                    break

        logger.info(f'length of results: {len(results)}')
        return results

    def find_threshold(self):
        print(self.model_name)
        print('Confusion matrix:')
        y_true = [x['ans_type_gold'] for x in self.results]
        y_pred = [x['ans_type_pred'] for x in self.results]
        print(confusion_matrix(y_true, y_pred))
        print(accuracy_score(y_true, y_pred))
        print(f'Gold: {min(y_true)}~{max(y_true)}')
        print(f'Pred: {min(y_pred)}~{max(y_pred)}')
        thresholds = np.linspace(0,1, 1001)

        accuracy_scores = []
        for thre in thresholds:
            max_yesno = [max(x['probs'][3], x['probs'][4]) for x in self.results]
            y_pred_clip = [x['ans_type_pred'] if max_yesno[i] > thre else 2 for i, x in enumerate(self.results)]
            # precision, recall, thresholds = precision_recall_curve(y_test, y_test_predicted_probas)
            accuracy_scores.append(accuracy_score(y_true, y_pred_clip))
            # mat = confusion_matrix(y_true, y_pred_clip)
            # print(mat)
            # print(thre)

        accuracies = np.array(accuracy_scores)
        max_accuracy = max(accuracies)
        min_acc = min(accuracies)
        max_accuracy_threshold = thresholds[accuracies.argmax()]
        print(max_accuracy, max_accuracy_threshold, min_acc)

    def predict_yn(self, question, answer):
        if not self.predictor:
            self.predictor = QatypePredictor.from_path(self.model_path,
                                                            predictor_name='qatype',
                                                            cuda_device=CUDA_DEVICE)
        res = self.predictor.predict_qa(question, answer)
        return YN_TYPE_DICT[int(res["label"])]

    def load(self):
        if not self.predictor:
            self.predictor = QatypePredictor.from_path(self.model_path,
                                                       predictor_name='qatype',
                                                       cuda_device=CUDA_DEVICE)


