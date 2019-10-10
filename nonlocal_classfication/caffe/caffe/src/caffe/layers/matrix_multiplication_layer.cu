#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/matrix_multiplication_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void MatrixMultiplicationLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  const Dtype* X_data = bottom[0]->gpu_data();
  const Dtype* Y_data = bottom[1]->gpu_data();
  Dtype* Z_data = top[0]->mutable_gpu_data();

  const bool X_hasbatch = (bottom[0]->num_axes()>2);
  const bool Y_hasbatch = (bottom[1]->num_axes()>2);
  const bool Z_hasbatch = (top[0]->num_axes()>2);
  const int B = Z_hasbatch ? top[0]->count(0, top[0]->CanonicalAxisIndex(-2)) : 1;
  const int X_stride = M_ * K_;
  const int Y_stride = K_ * N_;
  const int Z_stride = M_ * N_;
  for(int b=0; b<B; ++b) {//TODO: parfor by OpenMP?
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
      M_, N_, K_,
      (Dtype)1.,
      X_data+b*X_stride*int(X_hasbatch), Y_data+b*Y_stride*int(Y_hasbatch),
      (Dtype)0.,
      Z_data+b*Z_stride*int(Z_hasbatch));
  }
}

template <typename Dtype>
void MatrixMultiplicationLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

  const Dtype* Z_diff = top[0]->gpu_diff();
  const Dtype* Y_data = bottom[1]->gpu_data();
  Dtype* X_diff = bottom[0]->mutable_gpu_diff();
  const Dtype* X_data = bottom[0]->gpu_data();
  Dtype* Y_diff = bottom[1]->mutable_gpu_diff();

  const bool X_hasbatch = (bottom[0]->num_axes()>2);
  const bool Y_hasbatch = (bottom[1]->num_axes()>2);
  const bool Z_hasbatch = (top[0]->num_axes()>2);
  const int B = Z_hasbatch ? top[0]->count(0, top[0]->CanonicalAxisIndex(-2)) : 1;
  const bool X_needbroadcast = (bottom[0]->num_axes() < bottom[1]->num_axes());
  const bool Y_needbroadcast = (bottom[1]->num_axes() < bottom[0]->num_axes());
  if (X_needbroadcast) {
    caffe_gpu_set<Dtype>(bottom[0]->count(), (Dtype)0., X_diff);
  }
  if (Y_needbroadcast) {
    caffe_gpu_set<Dtype>(bottom[1]->count(), (Dtype)0., Y_diff);
  }
  const int X_stride = M_ * K_;
  const int Y_stride = K_ * N_;
  const int Z_stride = M_ * N_;
  for(int b=0; b<B; ++b) {//TODO: parfor by OpenMP?
    if (propagate_down[0]) {
      // dl/dX' = dl/d(XY)' * Y', i.e., bottom[0].diff = top[0].diff * bottom[1].data'
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
        M_, K_, N_,
        (Dtype)1.,
        Z_diff+b*Z_stride*int(Z_hasbatch), Y_data+b*Y_stride*int(Y_hasbatch),
        (Dtype)(X_needbroadcast? 1. : 0.),
        X_diff+b*X_stride*int(X_hasbatch));
    }
    if (propagate_down[1]) {
      // dl/dY' = X' * dl/d(XY)', i.e., bottom[1].diff = bottom[0].data' * top[0].diff
      caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
        K_, N_, M_,
        (Dtype)1.,
        X_data+b*X_stride*int(X_hasbatch), Z_diff+b*Z_stride*int(Z_hasbatch),
        (Dtype)(Y_needbroadcast? 1. : 0.),
        Y_diff+b*Y_stride*int(Y_hasbatch));
    }
  } //for b
}

INSTANTIATE_LAYER_GPU_FUNCS(MatrixMultiplicationLayer);

}  // namespace caffe
