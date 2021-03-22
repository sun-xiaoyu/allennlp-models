from typing import List, Dict

import numpy
from overrides import overrides

from allennlp.common.util import JsonDict, sanitize
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor
from allennlp.data.fields import LabelField


@Predictor.register("qatype")
class QatypePredictor(Predictor):
    """
    Predictor for the [`DecomposableAttention`](../models/decomposable_attention.md) model.

    Registered as a `Predictor` with name "textual_entailment".
    """


    def predict_qa(self, question: str, answer: str = None) -> JsonDict:
        """
        Predicts whether the hypothesis is entailed by the premise text.

        # Parameters

        premise : `str`
            A passage representing what is assumed to be true.

        hypothesis : `str`
            A sentence that may be entailed by the premise.

        # Returns

        `JsonDict`
            A dictionary where the key "label_probs" determines the probabilities of each of
            [entailment, contradiction, neutral].
        """
        return self.predict_json({"question": question, "answer": answer})

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like `{"premise": "...", "hypothesis": "..."}`.
        """
        question = json_dict["question"]
        answer = json_dict["answer"]
        return self._dataset_reader.text_to_instance(None, question, answer)

    @overrides
    def predict_instance(self, instance: Instance) -> JsonDict:
        result = self._model.forward_on_instance(instance)
        return sanitize(result)

    @overrides
    def predictions_to_labeled_instances(
        self, instance: Instance, outputs: Dict[str, numpy.ndarray]
    ) -> List[Instance]:
        new_instance = instance.duplicate()
        label = numpy.argmax(outputs["label_logits"])
        # Skip indexing, we have integer representations of the strings "entailment", etc.
        new_instance.add_field("label", LabelField(int(label), skip_indexing=True))
        return [new_instance]
