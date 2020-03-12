from typing import List, Tuple

import pytest

from allennlp.common.util import ensure_list
from allennlp.common.testing import AllenNlpTestCase

from allennlp_models.coref import PrecoReader
from tests import FIXTURES_ROOT


class TestPrecoReader:
    span_width = 5

    @pytest.mark.parametrize("remove_singleton_clusters", (True, False))
    def test_read_from_file(self, remove_singleton_clusters):
        conll_reader = PrecoReader(
            max_span_width=self.span_width, remove_singleton_clusters=remove_singleton_clusters
        )
        instances = ensure_list(
            conll_reader.read(str(FIXTURES_ROOT / "coref" / "preco.jsonl"))
        )

        assert len(instances) == 3

        fields = instances[2].fields
        text = [x.text for x in fields["text"].tokens]

        assert text == [
            "When",
            "you",
            "watch",
            "TV",
            "and",
            "play",
            "video",
            "games",
            "you",
            "make",
            "global",
            "warming",
            "worse",
            "!",
            "It",
            "may",
            "seem",
            "hard",
            "to",
            "believe",
            ",",
            "but",
            "when",
            "electricity",
            "is",
            "made",
            ",",
            "so",
            "are",
            "greenhouse",
            "gases",
            ".",
            "This",
            "means",
            "that",
            "every",
            "time",
            "you",
            "use",
            "electricity",
            "you",
            "help",
            "make",
            "global",
            "warming",
            "worse",
            "!",
            "Cars",
            "are",
            "also",
            "making",
            "global",
            "warming",
            "worse",
            ".",
            "They",
            "burn",
            "fossil",
            "fuels",
            "in",
            "their",
            "engines",
            ",",
            "and",
            "send",
            "lots",
            "of",
            "greenhouse",
            "gases",
            "into",
            "the",
            "air",
            ".",
            "Global",
            "warming",
            "may",
            "be",
            "a",
            "big",
            "problem",
            ",",
            "but",
            "we",
            "can",
            "all",
            "help",
            "stop",
            "it",
            ".",
            "People",
            "can",
            "try",
            "to",
            "drive",
            "their",
            "cars",
            "less",
            ".",
            "Or",
            "even",
            "get",
            "ones",
            "that",
            "run",
            "on",
            "sunlight",
            "!",
            "You",
            "can",
            "also",
            "help",
            ".",
            "Let",
            "'s",
            "try",
            "one",
            "of",
            "these",
            "top",
            "ideas",
            ":",
            "(",
            "1",
            ")",
            "Try",
            "to",
            "use",
            "less",
            "electricity",
            ".",
            "Turn",
            "off",
            "lights",
            ",",
            "your",
            "television",
            ",",
            "and",
            "your",
            "computer",
            "when",
            "you",
            "'ve",
            "stopped",
            "using",
            "them",
            ".",
            "To",
            "make",
            "electricity",
            ",",
            "fossil",
            "fuels",
            "are",
            "burned",
            "in",
            "big",
            "factories",
            ".",
            "But",
            "burning",
            "fossil",
            "fuels",
            "also",
            "makes",
            "greenhouse",
            "gases",
            ".",
            "You",
            "should",
            "also",
            "try",
            "to",
            "watch",
            "less",
            "TV",
            ".",
            "(",
            "2",
            ")",
            "Plant",
            "trees",
            ".",
            "Not",
            "only",
            "is",
            "it",
            "a",
            "fun",
            "thing",
            "to",
            "do",
            ",",
            "but",
            "it",
            "is",
            "also",
            "a",
            "great",
            "way",
            "to",
            "lower",
            "the",
            "number",
            "of",
            "greenhouse",
            "gases",
            "in",
            "the",
            "air",
            ".",
            "Trees",
            "take",
            "carbon",
            "dioxide",
            "out",
            "of",
            "the",
            "air",
            "when",
            "they",
            "grow",
            ".",
            "(",
            "3",
            ")",
            "Do",
            "n't",
            "throw",
            "away",
            "your",
            "rubbish",
            ",",
            "try",
            "to",
            "recycle",
            "it",
            ".",
            "If",
            "rubbish",
            "is",
            "not",
            "recycled",
            ",",
            "it",
            "is",
            "put",
            "in",
            "the",
            "ground",
            ".",
            "There",
            "it",
            "rots",
            "and",
            "makes",
            "a",
            "greenhouse",
            "gas",
            "called",
            "methane",
            ".",
            "So",
            "try",
            "to",
            "recycle",
            "cans",
            ",",
            "bottles",
            ",",
            "plastic",
            "bags",
            "and",
            "newspapers",
            ".",
            "It",
            "'ll",
            "make",
            "you",
            "feel",
            "great",
            "!",
            "And",
            "it",
            "'ll",
            "help",
            "the",
            "Earth",
            ".",
        ]

        spans = fields["spans"].field_list
        span_starts, span_ends = zip(*[(field.span_start, field.span_end) for field in spans])

        candidate_mentions = self.check_candidate_mentions_are_well_defined(
            span_starts, span_ends, text
        )

        gold_span_labels = fields["span_labels"]
        gold_indices_with_ids = [(i, x) for i, x in enumerate(gold_span_labels.labels) if x != -1]
        gold_mentions_with_ids: List[Tuple[List[str], int]] = [
            (candidate_mentions[i], x) for i, x in gold_indices_with_ids
        ]

        assert (["you"], 0) in gold_mentions_with_ids
        gold_mentions_with_ids.remove((["you"], 0))
        assert (["you"], 0) in gold_mentions_with_ids

        if not remove_singleton_clusters:
            # Singleton mention
            assert (["video", "games"], 2) in gold_mentions_with_ids
            gold_mentions_with_ids.remove((["video", "games"], 2))
            assert not any(_ for _, id_ in gold_mentions_with_ids if id_ == 2)

            assert (["them"], 24) in gold_mentions_with_ids
            # This is a span which exceeds our max_span_width, so it should not be considered.
            assert (
                ["lights", ",", "your", "television", ",", "and", "your", "computer"],
                24,
            ) not in gold_mentions_with_ids
        else:
            assert (["video", "games"], 2) not in gold_mentions_with_ids

    def check_candidate_mentions_are_well_defined(self, span_starts, span_ends, text):
        candidate_mentions = []
        for start, end in zip(span_starts, span_ends):
            # Spans are inclusive.
            text_span = text[start : end + 1]
            candidate_mentions.append(text_span)

        # Check we aren't considering zero length spans and all
        # candidate spans are less than what we specified
        assert all(self.span_width >= len(x) > 0 for x in candidate_mentions)
        return candidate_mentions
