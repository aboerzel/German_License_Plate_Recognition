import itertools
import numpy as np


class LabelCodec:
    ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZÄÖÜ0123456789- "

    # Translation of characters to unique numerical classes
    @staticmethod
    def encode_number(number):
        return list(map(lambda c: LabelCodec.ALPHABET.index(c), number))

    # Reverse translation of numerical classes back to characters
    @staticmethod
    def decode_number(label):
        return ''.join(list(map(lambda x: LabelCodec.ALPHABET[int(x)], label)))

    @staticmethod
    def decode_prediction(prediction):
        out_best = list(np.argmax(prediction, axis=1))
        out_best = [k for k, g in itertools.groupby(out_best)]
        outstr = ''
        for c in out_best:
            if c < len(LabelCodec.ALPHABET):
                outstr += LabelCodec.ALPHABET[c]
        return outstr
