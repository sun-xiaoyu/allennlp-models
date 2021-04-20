import logging
from typing import Any, Dict, List, Optional

import random
import numpy as np
import torch
import pickle
from allennlp.common.util import sanitize_wordpiece
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import PretrainedTransformerEmbedder
from allennlp.modules import FeedForward
from allennlp.modules.seq2vec_encoders import CnnEncoder, BertPooler
from allennlp.nn.util import get_token_ids_from_text_field_tensors
from torch import nn

from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.nn import util, Activation
from allennlp.training.metrics import BooleanAccuracy, CategoricalAccuracy
from torch.nn.functional import cross_entropy

from nq.models_utils import get_best_span
from nq.squad_em_and_f1 import SquadEmAndF1
from allennlp_models.lm.bidirectional_lm_transformer import MultiHeadedAttention, make_model

logger = logging.getLogger(__name__)


@Model.register("bert_joint")
class BertJointNQ(Model):
    """
    This class implements a reading comprehension model patterned after the proposed model in
    https://arxiv.org/abs/1810.04805 (Devlin et al), with improvements borrowed from the SQuAD model in the
    transformers project.

    It predicts start tokens and end tokens with a linear layer on top of word piece embeddings.
    输出每个位置作为 开始和结束的概率

    重要！模型显示的 metrics 是 per-instance 而不是 per-example
    也就是说 不是官方的分数，不可全信
    Note that the metrics that the model produces are calculated on a per-instance basis only. Since there could
    be more than one instance per question, these metrics are not the official numbers on the SQuAD task. To get
    official numbers, run the script in scripts/transformer_qa_eval.py.

    Parameters
    ----------
    vocab : ``Vocabulary``
    transformer_model_name : ``str``, optional (default=``bert-base-cased``)
        This model chooses the embedder according to this setting. You probably want to make sure this is set to
        the same thing as the reader.
    """

    def __init__(
        self, vocab: Vocabulary, classifier_feedforward: Optional[FeedForward] = None,
            transformer_model_name: str = "bert-base-cased",
            vocab_len_location="/home/sunxy-s18/data/nq/vocab.len",
            option: str = 'original',
            # init_all: bool = False
            **kwargs,
    ) -> None:
        super().__init__(vocab, **kwargs)
        self._text_field_embedder = BasicTextFieldEmbedder(
            {"tokens": PretrainedTransformerEmbedder(transformer_model_name)}
        )

        logger.info('wtf' * 20)
        # if str(vocab_len_location).isdigit():
        #     vocab_len = int(vocab_len_location)
        # else:
        #     with open(vocab_len_location, 'r') as f:
        #         vocab_len = int(f.read())
        # self._text_field_embedder._token_embedders['tokens'].transformer_model.resize_token_embeddings(vocab_len)
        # logger.info(f'Model embedding matrix resized to length: {vocab_len}')
        logger.info('yeah' * 20)

        output_dim = self._text_field_embedder.get_output_dim()
        self._linear_layer = nn.Linear(output_dim, 2)
        '''
        cnn: 用cnn对指针网络、是否答案做特征抽取
        transformer: 
        combine: 加了三层网络 输入771
        combine1: 只保留一层linear网络 输入771
        combine2：理论上应该和combine1一样，只不过用的网络不是output_layer而是 classifier_feedforward
        combine3: 先把cls的768维压缩到12维，然后和3维并到一起。15维一起输入一个[15,3]的linear
        '''

        init_all = False
        options = ['original', 'cnn', 'transformer', 'combine2', 'no_type_loss', 'combine3', 'default_pooler']
        self.option = option
        if self.option == 'original' or self.option == 'default_pooler' or init_all:
            if classifier_feedforward:
                self.classifier_feedforward = classifier_feedforward
                logger.warning('We are using fine-tuned model!')
                logger.info(str(classifier_feedforward))
            else:
                self.classifier_feedforward = FeedForward(input_dim=output_dim, num_layers=1, hidden_dims=[5],
                                                 activations=[Activation.by_name("linear")()])
            if self.option == 'default_pooler':
                self.pooler = BertPooler(pretrained_model=transformer_model_name, )
        if self.option == 'cnn' or init_all:
            self.cnn = CnnEncoder(embedding_dim=4, num_filters=3, ngram_filter_sizes=(2,))
            self.cnn_output_linear = nn.Linear(3, 3, bias=True)
        if self.option == 'transformer' or init_all:
            self.cnn = None
            self.ans_type_predict_layer = make_model(num_layers=1, input_size=16, hidden_size=42, heads=4)
            self.pre_attention = FeedForward(input_dim=4, num_layers=1, hidden_dims=16,
                                             activations=Activation.by_name("relu")(), dropout=0.1)
            self.transformer_output_linear = nn.Linear(16, 3, bias=True)
        if self.option == 'combine2' or init_all:
            self.cnn = CnnEncoder(embedding_dim=4, num_filters=3, ngram_filter_sizes=(2,))
            if classifier_feedforward:
                self.combine2_output_linear = classifier_feedforward
            else:
                self.combine2_output_linear = FeedForward(input_dim=output_dim+3, num_layers=1, hidden_dims=[5],
                                                      activations=Activation.by_name("linear")())
        if self.option == 'combine3' or init_all:
            self.cnn = CnnEncoder(embedding_dim=4, num_filters=3, ngram_filter_sizes=(2,))
            self.combine3_cls_pooler = FeedForward(input_dim=output_dim, num_layers=1, hidden_dims=[12],
                                                   activations=Activation.by_name("relu")(), dropout=0.1)
            self.combine3_output_linear = FeedForward(input_dim=15, num_layers=1, hidden_dims=[5],
                                                      activations=Activation.by_name("linear")())
            # self.output_linear = FeedForward(input_dim=771, num_layers=3, hidden_dims=[128,16,3],
            #                              activations=[Activation.by_name("relu")(), Activation.by_name("relu")(),
            #                                           Activation.by_name("linear")()], dropout=0.1)
        #     self.output_linear = FeedForward(input_dim=771, num_layers=1, hidden_dims=[5],
        #                                  activations=[Activation.by_name("linear")()])
        # else:
        #     self.output_linear = nn.Linear(3,3,bias=True)


        self._span_start_accuracy = CategoricalAccuracy()
        self._span_end_accuracy = CategoricalAccuracy()
        self._span_accuracy = BooleanAccuracy()
        self._ans_type_accuracy = CategoricalAccuracy()
        self._per_instance_metrics = SquadEmAndF1()

        self.no_type_loss = self.option == 'no_type_loss'
        logger.info(f'No type_loss**: {self.no_type_loss}')
        logger.info(f'We use model architecture as: {self.option}')

        self.must_have_ans = False


    def forward(  # type: ignore
        self,
        question_with_context: Dict[str, Dict[str, torch.LongTensor]],
        context_span: torch.IntTensor,
        sptk: Optional[torch.IntTensor] = None,
        answer_span: Optional[torch.IntTensor] = None,
        answer_type: Optional[torch.IntTensor] = None,
        metadata: List[Dict[str, Any]] = None,
    ) -> Dict[str, torch.Tensor]:

        """
        Parameters
        ----------
        question_with_context : Dict[str, torch.LongTensor]
            From a ``TextField``. The model assumes that this text field contains the context followed by the
            question. It further assumes that the tokens have type ids set such that any token that can be part of
            the answer (i.e., tokens from the context) has type id 0, and any other token (including [CLS] and
            [SEP]) has type id 1.
        context_span : ``torch.IntTensor``
            From a ``SpanField``. This marks the span of word pieces in ``question`` from which answers can come.
        answer_span : ``torch.IntTensor``, optional
            From a ``SpanField``. This is the 1st thing we are trying to predict - the span of text that marks the
            answer. If given, we compute a loss that gets included in the output directory.
        answer_type : torch.IntTensor, optional (default = None)
            From a `LabelField`. This is the 2nd thing we are trying to predict - the answer type.
        metadata : ``List[Dict[str, Any]]``, optional
            If present, this should contain the question id, and the original texts of context, question, tokenized
            version of both, and a list of possible answers. The length of the ``metadata`` list should be the
            batch size, and each dictionary should have the keys ``id``, ``question``, ``context``,
            ``question_tokens``, ``context_tokens``, and ``answers``.

        Returns
        -------
        An output dictionary consisting of:
        span_start_logits : torch.FloatTensor
            A tensor of shape ``(batch_size, passage_length)`` representing unnormalized log
            probabilities of the span start position.
        span_start_probs : torch.FloatTensor
            The result of ``softmax(span_start_logits)``.
        span_end_logits : torch.FloatTensor
            A tensor of shape ``(batch_size, passage_length)`` representing unnormalized log
            probabilities of the span end position (inclusive).
        span_end_probs : torch.FloatTensor
            The result of ``softmax(span_end_logits)``.
        best_span : torch.IntTensor
            The result of a constrained inference over ``span_start_logits`` and
            ``span_end_logits`` to find the most probable span.  Shape is ``(batch_size, 2)``
            and each offset is a token index.
        best_span_scores : torch.FloatTensor
            The score for each of the best spans.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.
        best_span_str : List[str]
            If sufficient metadata was provided for the instances in the batch, we also return the
            string from the original passage that the model thinks is the best answer to the
            question.
            :param sptk:
        """
        # print(question_with_context)
        # print(question_with_context['tokens'])
        # print(question_with_context['tokens']['token_ids'].shape)
        # print(question_with_context['tokens']['mask'].shape)
        # print(question_with_context['tokens']['type_ids'].shape)
        # print('\n\n')
        # pickle.dump(question_with_context['tokens']['token_ids'], open('/home/sunxy-s18/data/bad_tensor','wb'))
        logger.debug(f'Batch size: {context_span.size()[0]}')
        embedded_question = self._text_field_embedder(question_with_context) # TODO ERROR HERE


        # print("embedded_question", embedded_question.shape)
        # raise Exception('unacceptable!')
        # shape: (batch_size, sequence_length, 2)
        logits = self._linear_layer(embedded_question)
        # shape: (batch_size, sequence_length, 1)
        span_start_logits, span_end_logits = logits.split(1, dim=-1)
        # shape: (batch_size, sequence_length)
        span_start_logits = span_start_logits.squeeze(-1)
        # shape: (batch_size, sequence_length)
        span_end_logits = span_end_logits.squeeze(-1)
        # print('span_start_logits', span_start_logits.shape)
        # print('span_end_logits', span_end_logits.shape)

        # Create a mask for `question_with_context` to mask out tokens that are not part
        # of the context.
        # shape: (batch_size, sequence_length)
        possible_answer_mask = torch.zeros_like(
            get_token_ids_from_text_field_tensors(question_with_context), dtype=torch.bool
        )

        # here we use mask:
        # we replace all logits whose position are now within the context_span with 0
        # that way, there is no possibility that answer fall in the [cls][question tokens][sep] part

        for i, (start, end) in enumerate(context_span):
            possible_answer_mask[i, start: end + 1] = True
            # Also unmask the [CLS] token since that token is used to indicate that
            # the question is impossible.
            possible_answer_mask[i, 0] = True
            if self.must_have_ans:
                possible_answer_mask[i, 0] = False

        no_ans_score = span_start_logits[:,0].unsqueeze(1) + span_end_logits[:,0].unsqueeze(1)

        # Replace the masked values with a very negative constant since we're in log-space.
        # shape: (batch_size, sequence_length)
        span_start_logits = util.replace_masked_values(
            span_start_logits, possible_answer_mask, -1e32
        )
        span_end_logits = util.replace_masked_values(span_end_logits, possible_answer_mask, -1e32)


        span_start_probs = torch.nn.functional.softmax(span_start_logits, dim=-1)
        span_end_probs = torch.nn.functional.softmax(span_end_logits, dim=-1)
        # Now calculate the best span.
        # shape: (batch_size, 2)
        best_spans = get_best_span(span_start_logits, span_end_logits)

        # Sum the span start score with the span end score to get an overall score for the span.
        # shape: (batch_size,)
        best_span_scores = torch.gather(
            span_start_logits, 1, best_spans[:, 0].unsqueeze(1)
        ) + torch.gather(span_end_logits, 1, best_spans[:, 1].unsqueeze(1)) - no_ans_score
                           # span_start_logits[:,0].unsqueeze(1)- span_end_logits[:,0].unsqueeze(1)
        best_span_scores = best_span_scores.squeeze(1)
        bert_cls_vec = embedded_question[:, 0, :]
        # output_dict = {
        #     "span_start_logits": span_start_logits,
        #     # "span_start_probs": span_start_probs,
        #     "span_end_logits": span_end_logits,
        #     # "span_end_probs": span_end_probs,
        # }
        output_dict = {}
        output_dict['best_span'] = best_spans
        output_dict["best_span_scores"] = best_span_scores


        # get extra feature here
        if not self.no_type_loss:
            if self.option in ['cnn', 'transformer', 'combine2', 'combine3']:
                input_ids = question_with_context['tokens']['token_ids']
                batch_size, seq_length = input_ids.shape
                device = input_ids.device
                if sptk is None:
                    sptk = torch.zeros([batch_size, seq_length, 1], dtype=torch.float, device=device)
                else:
                    sptk = sptk.unsqueeze(-1)

                # todo change
                # compute pooled for
                if self.option == 'transformer':
                    position_ids = torch.arange(seq_length, dtype=torch.float, device=device)
                    position_ids = position_ids.unsqueeze(0).expand(input_ids.shape).unsqueeze(-1)
                    type_vec_for_transformer = torch.cat((logits, position_ids, sptk), dim=2)
                    att_mask = torch.zeros([batch_size, seq_length, seq_length], dtype=torch.bool, device=device)
                    for i, (start, end) in enumerate(context_span):
                        att_mask[i, start: end + 1, :] = True
                        att_mask[i, :, start: end + 1] = True
                        # Also unmask the [CLS] token since that token is used to indicate that
                        # the question is impossible.
                        att_mask[i, 0, :] = True
                        att_mask[i, :, 0] = True

                    type_vec = self.pre_attention(type_vec_for_transformer)
                    transformer_logits = self.ans_type_predict_layer(type_vec, att_mask)
                    pooled = transformer_logits[:, 0, :].squeeze(1)
                    type_logits = self.transformer_output_linear(pooled)
                else:
                    cls_marker = torch.zeros([batch_size, seq_length, 1], dtype=torch.float, device=device)
                    cls_marker[:, 0] = True
                    type_vec_for_cnn = torch.cat((logits, sptk, cls_marker), dim=2)
                    pooled = self.cnn(type_vec_for_cnn, possible_answer_mask)
                    if self.option == 'combine2':
                        bert_cls_vec = torch.cat([bert_cls_vec, pooled], dim=1)
                        type_logits = self.combine2_output_linear(bert_cls_vec)
                    elif self.option == 'combine3':
                        cls_pooled = self.combine3_cls_pooler(bert_cls_vec)
                        pooled = torch.cat([cls_pooled, pooled], dim=-1)
                        type_logits = self.combine3_output_linear(pooled)
                    else:
                        type_logits = self.cnn_output_linear(pooled)

            elif self.option == 'default_pooler':
                pooled = self.pooler(embedded_question)
                type_logits = self.classifier_feedforward(pooled)
            else: # self.option == 'original':
                type_logits = self.classifier_feedforward(bert_cls_vec)

                # todo end change

            type_probs = torch.nn.functional.softmax(type_logits, dim=-1)
            argmax_indices = torch.argmax(type_probs, dim=-1)
            output_dict['answer_type_logits'] = type_logits
            output_dict["answer_type_probs"] = type_probs
            output_dict["ans_type_pred_cls"] = argmax_indices


        # todo it was here


        # Compute the loss for training.
        if answer_span is not None:
            span_start = answer_span[:, 0]
            span_end = answer_span[:, 1]
            # span_mask = span_start != -1
            self._span_accuracy(
                best_spans, answer_span,  # span_mask.unsqueeze(-1).expand_as(best_spans)
            )

            ignored_index = -100
            if self.must_have_ans:
                ignored_index = 0
            start_loss = cross_entropy(span_start_logits, span_start, ignore_index=ignored_index)  # ignore_index=-1)
            if torch.any(start_loss > 1e9):
                logger.critical("Start loss too high (%r)", start_loss)
                logger.critical("span_start_logits: %r", span_start_logits)
                logger.critical("span_start: %r", span_start)
                assert False

            end_loss = cross_entropy(span_end_logits, span_end, ignore_index=ignored_index)  # ignore_index=-1)
            if torch.any(end_loss > 1e9):
                logger.critical("End loss too high (%r)", end_loss)
                logger.critical("span_end_logits: %r", span_end_logits)
                logger.critical("span_end: %r", span_end)
                assert False

            if self.no_type_loss:
                loss = start_loss + end_loss
            else:
                type_loss = cross_entropy(type_logits, answer_type)
                loss = start_loss + end_loss + type_loss
                self._ans_type_accuracy(type_logits, answer_type)

            self._span_start_accuracy(span_start_logits, span_start)
            self._span_end_accuracy(span_end_logits, span_end)


            output_dict["loss"] = loss

        # # debug here
        # if answer_type is not None:
        #     for i, label in enumerate(output_dict['answer_type_pred']):
        #         if metadata[i]['answer_type_pred'] != answer_type[i]:
        #             logger.info(metadata[i])
        #             logger.info(output_dict[''])

        # Compute the EM and F1 on SQuAD and add the tokenized input to the output.
        # get best span str & best span orig
        if metadata is not None:
            best_spans = best_spans.detach().cpu().numpy()
            if not self.no_type_loss and 'ans_type_pred_cls' in output_dict:
                answer_type_preds = output_dict['ans_type_pred_cls'].detach().cpu().numpy()
            output_dict["best_span_str"] = []
            output_dict["best_span_orig"] = []
            for i, (metadata_entry, best_span) in enumerate(zip(metadata, best_spans)):
                if 'window_context_wordpiece_tokens' in metadata_entry:
                    window_context_wordpiece_tokens = metadata_entry["window_context_wordpiece_tokens"]

                    # 注释 we remove the offsets and only focus on the context part
                    if best_span[0] == 0:
                        # Predicting [CLS] is interpreted as predicting the question as unanswerable.
                        best_span_string = ""
                        # NOTE: even though we've "detached" 'best_spans' above, this still
                        # modifies the original tensor in-place.
                        best_span[0], best_span[1] = -1, -1
                        orig_start_token, orig_end_token = -1, -1
                    else:
                        best_span -= metadata_entry['start_of_context']
                        assert np.all(best_span >= 0)

                        predicted_start, predicted_end = tuple(best_span)
                        orig_start_token = window_context_wordpiece_tokens[predicted_start].idx
                        orig_end_token = window_context_wordpiece_tokens[predicted_end].idx
                        start_of_orig_context = window_context_wordpiece_tokens[0].idx
                        # 下面这一段好像意思是，如果预测到的点在一个词的中间，那么这个对应的token应该idx是none，
                        # 如果是答案开始，我们往前找到第一个有index的，如果是答案结束我们往后找第一个有index的
                        # while (
                        #     predicted_start >= 0
                        #     and context_tokens_for_question[predicted_start].idx is None
                        # ):
                        #     predicted_start -= 1
                        # if predicted_start < 0:
                        #     logger.warning(
                        #         f"Could not map the token '{window_context_wordpiece_tokens[best_span[0]].text}' at index "
                        #         f"'{best_span[0]}' to an offset in the original text."
                        #     )
                        #     orig_start_token = 0
                        # else:
                        #     orig_start_token = window_context_wordpiece_tokens[predicted_start].idx

                        # while (
                        #     predicted_end < len(context_tokens_for_question)
                        #     and context_tokens_for_question[predicted_end].idx is None
                        # ):
                        #     predicted_end += 1
                        # if predicted_end >= len(window_context_wordpiece_tokens):
                        #     logger.warning(
                        #         f"Could not map the token '{window_context_wordpiece_tokens[best_span[1]].text}' at index "
                        #         f"'{best_span[1]}' to an offset in the original text."
                        #     )
                        #     orig_end_token = len(metadata_entry["context"])
                        # else:
                        #     orig_end_token = window_context_wordpiece_tokens[predicted_end].idx

                        # TODO 那么问题来了，我为什么要改下面这行呢？我为什么之前要改 best_span_string 的计算方式呢
                        # 下面两行，第一行是20年11月发现的问题，当时没改。第二行是21年2月改过来的
                        best_span_string = ' '.join(metadata_entry["context"].split(' ')[
                                                    orig_start_token - start_of_orig_context:
                                                    orig_end_token+1 - start_of_orig_context])
                            # best_span_string = metadata_entry["context"][orig_start_token:orig_end_token]

                    answer_text = metadata_entry.get("answer_text")
                    if len(answer_text) > 0:
                        self._per_instance_metrics(best_span_string, [answer_text])
                    # TODO 问题出在上面这一行。和原来的 model 文件相比，best_span_string的计算方法变了，所以实际传进去的东西不一样。

                    output_dict["best_span_str"].append(best_span_string)
                    output_dict["best_span_orig"].append((orig_start_token, orig_end_token))
                    display_bad_case = False
                    # display_bad_case = True
                    if display_bad_case:
                        answer_type_pred = answer_type_preds[i]
                        answer_type_idx = metadata_entry['answer_type_idx']
                        # if answer_type_pred != answer_type_idx or answer_type_pred != 0 and best_span_string != answer_text:
                        if best_span_string != answer_text and answer_type_idx != 0:
                            # if random.random() < 0.05:
                                print(metadata_entry['id'])
                                print(metadata_entry['doc_url'])
                                print('-' * 50)
                                print('Q:', metadata_entry['question'])
                                print('-' * 50)
                                print('Prediction', answer_type_pred, orig_start_token, orig_end_token)
                                print(best_span)
                                print('~~~', best_span_string)
                                print('-'*30)
                                print('Gold:', answer_type_idx)
                                print('~~~', answer_text)
                                # for i in range(10):
                                #     print(context_tokens_for_question[i])
                                # print(context_tokens_for_question)
                                # print(metadata_entry['qwc'])
                                # print(metadata_entry['context'])
                                print('='*70)
                                print('')
                                print('')
                else:
                    # character level
                    context_tokens_for_question = metadata_entry["context_tokens"]
                    if best_span[0] == 0:
                        # Predicting [CLS] is interpreted as predicting the question as unanswerable.
                        best_span_string = ""
                        # NOTE: even though we've "detached" 'best_spans' above, this still
                        # modifies the original tensor in-place.
                        best_span[0], best_span[1] = -1, -1
                    else:
                        best_span -= 1 + len(metadata_entry["question_tokens"]) + 2
                        assert np.all(best_span >= 0)

                        predicted_start, predicted_end = tuple(best_span)

                        while (
                                predicted_start >= 0
                                and context_tokens_for_question[predicted_start].idx is None
                        ):
                            predicted_start -= 1
                        if predicted_start < 0:
                            logger.warning(
                                f"Could not map the token '{context_tokens_for_question[best_span[0]].text}' at index "
                                f"'{best_span[0]}' to an offset in the original text."
                            )
                            character_start = 0
                        else:
                            character_start = context_tokens_for_question[predicted_start].idx

                        while (
                                predicted_end < len(context_tokens_for_question)
                                and context_tokens_for_question[predicted_end].idx is None
                        ):
                            predicted_end += 1
                        if predicted_end >= len(context_tokens_for_question):
                            logger.warning(
                                f"Could not map the token '{context_tokens_for_question[best_span[1]].text}' at index "
                                f"'{best_span[1]}' to an offset in the original text."
                            )
                            character_end = len(metadata_entry["context"])
                        else:
                            end_token = context_tokens_for_question[predicted_end]
                            character_end = end_token.idx + len(sanitize_wordpiece(end_token.text))

                        best_span_string = metadata_entry["context"][character_start:character_end]
                    output_dict["best_span_str"].append(best_span_string)

                    answers = metadata_entry.get("answers")
                    if len(answers) > 0:
                        self._per_instance_metrics(best_span_string, answers)

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        exact_match, f1_score = self._per_instance_metrics.get_metric(reset)
        return {
            "start_acc": self._span_start_accuracy.get_metric(reset),
            "end_acc": self._span_end_accuracy.get_metric(reset),
            "span_acc": self._span_accuracy.get_metric(reset),
            "answer_type_acc": self._ans_type_accuracy.get_metric(reset),
            "per_instance_em": exact_match,
            "per_instance_f1": f1_score,
        }
