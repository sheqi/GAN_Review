#ifndef CAFFE_MATRIX_MULTIPLICATION_LAYER_HPP_
#define CAFFE_MATRIX_MULTIPLICATION_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief compute matrix multiplication of two input X and Y and output $Z=XY$
 *
 * Input:
 * X: <B1xB2...xBnxMxK> or <MxK>
 * Y: <B1xB2...xBnxKxN> or <KxN>
 * Output:
 * Z: <B1xB2...xBnxMxN> or <MxN>
 *
 * If X shape is <BxMxK> while Y shape is <KxN>, then $Z=\{X_0*Y, X_1*Y, ..., X_{B-1}*Y\}$ by broadcasting Y.
 * And similar for the other case by broadcasting X.
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
template <typename Dtype>
class MatrixMultiplicationLayer : public Layer<Dtype> {
 public:
  explicit MatrixMultiplicationLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "MatrixMultiplication"; }
  virtual inline int ExactNumBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int M_; //X <MxK>, Y<KxN>, Z<MxN>
  int K_;
  int N_;
};

}  // namespace caffe

#endif  // CAFFE_MATRIX_MULTIPLICATION_LAYER_HPP_
