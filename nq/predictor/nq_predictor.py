import json
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
        self._next_qid = 0

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

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        raise NotImplementedError(
            "This predictor maps a question to multiple instances. "
            "Please use _json_to_instances instead."
        )

    def _json_to_instances(self, json_dict: JsonDict) -> List[Instance]:
        result = list(
            self._dataset_reader.make_instances(
                doc_info={
                    'id': str(self._next_qid),
                    'doc_url': '',
                },
                question=json_dict["question"],
                answers=[],
                context=json_dict["context"],
                first_answer_offset=None,
                output_type='instances'
            )
        )
        self._next_qid += 1
        return result

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
    def predict_batch_instance(self, instances: List[Instance]) -> List[JsonDict]:
        outputs = self._model.forward_on_instances(instances)

        # group outputs with the same question id
        qid_to_output: Dict[str, Dict[str, Any]] = {}
        for instance, output in zip(instances, outputs):
            qid = instance["metadata"]["id"]
            output["id"] = qid
            output["answers"] = instance["metadata"]["answers"]
            if qid in qid_to_output:
                old_output = qid_to_output[qid]
                if old_output["best_span_scores"] < output["best_span_scores"]:
                    qid_to_output[qid] = output
            else:
                qid_to_output[qid] = output

        return [sanitize(o) for o in qid_to_output.values()]

    def predict_textins(self, input_paths):
        ins_gen = self._dataset_reader.read_text_instances(input_paths)
        instances = []
        for ins in ins_gen:
            instances.append(ins)
        print(len(instances))
        print(instances[0])
        n = len(instances)
        batch_size = 32
        batch_start = 0
        n_batch = len(instances) // batch_size + 1
        result = []
        for i in range(n_batch):
            batch_ins = instances[batch_start: min(n, batch_start+batch_size)]
            res = self.predict_batch_instance(batch_ins)
            result.extend(res)
            batch_start += batch_size
        return result



