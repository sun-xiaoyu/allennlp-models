import json
import logging
from typing import List, Dict, Any

from allennlp.models import Model
from overrides import overrides

from allennlp.common.util import JsonDict, sanitize
from allennlp.data import Instance, DatasetReader
from allennlp.predictors.predictor import Predictor


@Predictor.register("nq")
class NQPredictor(Predictor):
    """
    Predictor for the :class:`~allennlp_rc.models.TransformerQA` model, and any
    other model that takes a question and passage as input.
    """

    def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
        super(NQPredictor, self).__init__(model, dataset_reader)
        # self._next_qid = 0

    def set_raw(self, raw):
        self.raw_prediction = raw

    def predict(self, question: str, passage: str) -> JsonDict:
        """
        Make a machine comprehension prediction on the supplied input.
        See https://rajpurkar.github.io/SQuAD-explorer/ for more information about the machine comprehension task.

        Parameters
        ----------
        question : ``str``
            A question about the content in the supplied paragraph.  The question must be answerable by a
            span in the paragraph.
        passage : ``str``
            A paragraph of information relevant to the question.

        Returns
        -------
        A dictionary that represents the prediction made by the system.  The answer string will be under the
        "best_span_str" key.
        """
        return self.predict_json({"context": passage, "question": question})

    def predict_json(self, inputs: JsonDict) -> JsonDict:
        results = self.predict_batch_json([inputs])
        assert len(results) == 1
        return results[0]

    def predict_one_text_entry(self, inputs: JsonDict) -> JsonDict:
        instances, _, _ = self._dataset_reader.text_entry_js_to_instances(inputs, train=False, strategy='all')
        result = self.predict_batch_instance(instances)
        assert len(result) == 1
        return result[0]

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        raise NotImplementedError(
            "This predictor maps a question to multiple instances. "
            "Please use _json_to_instances instead."
        )

    def _json_to_instances(self, json_dict: JsonDict) -> List[Instance]:
        result = list(
            self._dataset_reader.make_instances(
                doc_info=json_dict['doc_info'],
                question=json_dict["question_text"],
                answers=[],
                context=json_dict["contexts"],
                first_answer_offset=None,
                output_type='instance',
                train=False
            )
        )
        # self._next_qid += 1
        return result

    def set_model_must_have_ans(self):
        self._model.must_have_ans = True

    @overrides
    def _batch_json_to_instances(self, json_dicts: List[JsonDict]) -> List[Instance]:
        instances = []
        for json_dict in json_dicts:
            instances.extend(self._json_to_instances(json_dict))
        return instances

    @overrides
    def predict_batch_json(self, inputs: List[JsonDict]) -> List[JsonDict]:
        instances = self._batch_json_to_instances(inputs)
        result = self.predict_batch_instance(instances)
        assert len(result) == len(inputs)
        return result

    @overrides
    def predict_batch_instance(self, instances: List[Instance], batch_size=24) -> List[JsonDict]:
        if len(instances) == 0:
            pass
            a = 1
        if len(instances) < batch_size:
            outputs = self._model.forward_on_instances(instances)
        else:
            n = len(instances)
            batch_start = 0
            n_batch = len(instances) // batch_size + 1
            outputs = []
            for i in range(n_batch):
                batch_ins = instances[batch_start: min(n, batch_start + batch_size)]
                if not batch_ins:
                    break
                res = self._model.forward_on_instances(batch_ins)
                outputs.extend(res)
                batch_start += batch_size

        if self.raw_prediction:
            print("Raw prediction! we don't try to avoid no-answer")
            return [sanitize(outputs[0])]

        # scores = [x['best_span_scores'] for x in outputs]
        outputs.sort(key=lambda x:x['best_span_scores'], reverse=True)
        for output in outputs:
            if output['best_span_str'] != '':
                return [sanitize(output)]
        return [sanitize(outputs[-1])]
        # todo important assumption!
        # 这里我们假设调用这个函数的都是同一个example生成的instance。所以用了上面的代码注释掉了下面的
        # group outputs with the same question id
        # qid_to_output: Dict[str, Dict[str, Any]] = {}
        # for instance, output in zip(instances, outputs):
        #     qid = instance["metadata"]["id"]
        #     output["id"] = qid
        #     output["answers"] = instance["metadata"]["answers"]
        #     if qid in qid_to_output:
        #         old_output = qid_to_output[qid]
        #         if old_output["best_span_scores"] < output["best_span_scores"]:
        #             qid_to_output[qid] = output
        #     else:
        #         qid_to_output[qid] = output
        #
        # return [sanitize(o) for o in qid_to_output.values()]

    def predict_textins(self, input_paths):
        ins_gen = self._dataset_reader.read_text_instances(input_paths)
        instances = []
        for ins in ins_gen:
            instances.append(ins)
        n = len(instances)
        batch_size = 24
        batch_start = 0
        n_batch = len(instances) // batch_size + 1
        result = []
        for i in range(n_batch):
            batch_ins = instances[batch_start: min(n, batch_start+batch_size)]
            res = self.predict_batch_instance(batch_ins)
            result.extend(res)
            batch_start += batch_size
        return result



