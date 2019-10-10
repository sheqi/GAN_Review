#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/matrix_multiplication_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void MatrixMultiplicationLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  Reshape(bottom, top);
}

template <typename Dtype>
void MatrixMultiplicationLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int xaxes = bottom[0]->num_axes();
  CHECK(xaxes>=2)
    << "X blob must be of shape (B1xB2...xBn,M,K) or (M,K)!";
  const int yaxes = bottom[1]->num_axes();
  CHECK(yaxes>=2)
    << "Y blob must be of shape (B1xB2...xBn,K,N) or (K,N)!";
  if (xaxes>2 && yaxes>2) {
    CHECK(xaxes==yaxes)
      << "X and Y blob must be of same shape when both of them have more than 2 dimensions!";
  }

  const int Rx = bottom[0]->shape(-2);
  const int Cx = bottom[0]->shape(-1);
  const int Ry = bottom[1]->shape(-2);
  const int Cy = bottom[1]->shape(-1);
  CHECK_EQ(Cx, Ry)
    << "Input X and Y have incompatible dimensions ("<<Rx<<"x"<<Cx<<" vs. "<<Ry<<"x"<<Cy<<").";
  M_ = Rx;
  K_ = Cx;
  N_ = Cy;
  
  vector<int> top_shape = bottom[ xaxes>=yaxes?0:1 ]->shape();
  top_shape[top_shape.size()-2]=M_;
  top_shape[top_shape.size()-1]=N_;
  top[0]->Reshape(top_shape); //B1x...xBnxMxN
}

template <typename Dtype>
void MatrixMultiplicationLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  const Dtype* X_data = bottom[0]->cpu_data();
  const Dtype* Y_data = bottom[1]->cpu_data();
  Dtype* Z_data = top[0]->mutable_cpu_data();

  const bool X_hasbatch = (bottom[0]->num_axes()>2);
  const bool Y_hasbatch = (bottom[1]->num_axes()>2);
  const bool Z_hasbatch = (top[0]->num_axes()>2);
  const int B = Z_hasbatch ? top[0]->count(0, top[0]->CanonicalAxisIndex(-2)) : 1;
  const int X_stride = M_ * K_;
  const int Y_stride = K_ * N_;
  const int Z_stride = M_ * N_;
  for(int b=0; b<B; ++b) {//TODO: parfor by OpenMP?
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
      M_, N_, K_,
      (Dtype)1.,
      X_data+b*X_stride*int(X_hasbatch), Y_data+b*Y_stride*int(Y_hasbatch),
      (Dtype)0.,
      Z_data+b*Z_stride*int(Z_hasbatch));
  }
}

template <typename Dtype>
void MatrixMultiplicationLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

  const Dtype* Z_diff = top[0]->cpu_diff();
  const Dtype* Y_data = bottom[1]->cpu_data();
  Dtype* X_diff = bottom[0]->mutable_cpu_diff();
  const Dtype* X_data = bottom[0]->cpu_data();
  Dtype* Y_diff = bottom[1]->mutable_cpu_diff();

  const bool X_hasbatch = (bottom[0]->num_axes()>2);
  const bool Y_hasbatch = (bottom[1]->num_axes()>2);
  const bool Z_hasbatch = (top[0]->num_axes()>2);
  const int B = Z_hasbatch ? top[0]->count(0, top[0]->CanonicalAxisIndex(-2)) : 1;
  const bool X_needbroadcast = (bottom[0]->num_axes() < bottom[1]->num_axes());
  const bool Y_needbroadcast = (bottom[1]->num_axes() < bottom[0]->num_axes());
  if (X_needbroadcast) {
    caffe_set<Dtype>(bottom[0]->count(), (Dtype)0., X_diff);
  }
  if (Y_needbroadcast) {
    caffe_set<Dtype>(bottom[1]->count(), (Dtype)0., Y_diff);
  }
  const int X_stride = M_ * K_;
  const int Y_stride = K_ * N_;
  const int Z_stride = M_ * N_;
  for(int b=0; b<B; ++b) {//TODO: parfor by OpenMP?
    if (propagate_down[0]) {
      // dl/dX' = dl/d(XY)' * Y', i.e., bottom[0].diff = top[0].diff * bottom[1].data'
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
        M_, K_, N_,
        (Dtype)1.,
        Z_diff+b*Z_stride*int(Z_hasbatch), Y_data+b*Y_stride*int(Y_hasbatch),
        (Dtype)(X_needbroadcast? 1. : 0.),
        X_diff+b*X_stride*int(X_hasbatch));
    }
    if (propagate_down[1]) {
      // dl/dY' = X' * dl/d(XY)', i.e., bottom[1].diff = bottom[0].data' * top[0].diff
      caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
        K_, N_, M_,
        (Dtype)1.,
        X_data+b*X_stride*int(X_hasbatch), Z_diff+b*Z_stride*int(Z_hasbatch),
        (Dtype)(Y_needbroadcast? 1. : 0.),
        Y_diff+b*Y_stride*int(Y_hasbatch));
    }
  } //for b
}

#ifdef CPU_ONLY
STUB_GPU(MatrixMultiplicationLayer);
#endif

INSTANTIATE_CLASS(MatrixMultiplicationLayer);
REGISTER_LAYER_CLASS(MatrixMultiplication);

}  // namespace caffe
