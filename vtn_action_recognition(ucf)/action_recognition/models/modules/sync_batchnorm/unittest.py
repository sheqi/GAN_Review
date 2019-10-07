# -*- coding: utf-8 -*-

import unittest

import numpy as np
from torch.autograd import Variable


def as_numpy(v):
    if isinstance(v, Variable):
        v = v.data
    return v.cpu().numpy()


class TorchTestCase(unittest.TestCase):
    def assertTensorClose(self, a, b, atol=1e-3, rtol=1e-3):
        npa, npb = as_numpy(a), as_numpy(b)
        self.assertTrue(
                np.allclose(npa, npb, atol=atol),
                'Tensor close check failed\n{}\n{}\nadiff={}, rdiff={}'.format(a, b, np.abs(npa - npb).max(), np.abs((npa - npb) / np.fmax(npa, 1e-5)).max())
        )
