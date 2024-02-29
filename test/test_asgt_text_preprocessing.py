# coding: utf-8
import unittest
import numpy as np
import transformers
from onnxruntime_extensions import PyOrtFunction, AsgtTextPreprocessing, util

vocab = {"[UNK]": 0, "[PAD]": 1, "hello": 2, "world": 3}


def _run_basic_case(input):
    t2stc = PyOrtFunction.from_customop(
        AsgtTextPreprocessing, vocab=vocab
    )
    result = t2stc(input)
    print(result)


class TestBertTokenizer(unittest.TestCase):
    def test_text_to_case1(self):
        print("\n\n****** Starting input ids, token type ids, and attention mask tests. ******\n")

        _run_basic_case(
            input=["Hello ww", "World", "Nothing"],
        )


        print("\n*** Offset mapping tests complete. ***\n")


if __name__ == "__main__":
    unittest.main()
