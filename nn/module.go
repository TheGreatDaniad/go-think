package nn

import "go-think/tensor"

// Module is the interface for all neural network layers
type Module interface {
	Forward(input *tensor.TensorCPU) *tensor.TensorCPU
	Parameters() []*tensor.TensorCPU
}
