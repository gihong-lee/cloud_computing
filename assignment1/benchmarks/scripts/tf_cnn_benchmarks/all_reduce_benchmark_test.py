# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for all_reduce_benchmark.py."""

import tensorflow.compat.v1 as tf

import all_reduce_benchmark
import benchmark_cnn
import test_util


class AllReduceBenchmarkTest(tf.test.TestCase):
  """Tests the all-reduce benchmark."""

  def _test_run_benchmark(self, params):
    """Tests that run_benchmark() runs successfully with the params."""
    logs = []
    with test_util.monkey_patch(all_reduce_benchmark,
                                log_fn=test_util.print_and_add_to_list(logs)):
      bench_cnn = benchmark_cnn.BenchmarkCNN(params)
      all_reduce_benchmark.run_benchmark(bench_cnn, num_iters=5)
      self.assertRegex(logs[-1], '^Average time per step: [0-9.]+$')

  def test_run_benchmark(self):
    """Tests that run_benchmark() runs successfully."""
    params = benchmark_cnn.make_params(num_batches=10,
                                       variable_update='replicated',
                                       num_gpus=2)
    self._test_run_benchmark(params)
    params = params._replace(hierarchical_copy=True, gradient_repacking=8,
                             num_gpus=8)
    self._test_run_benchmark(params)

if __name__ == '__main__':
  tf.disable_v2_behavior()
  tf.test.main()