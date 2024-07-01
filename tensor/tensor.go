package tensor

import (
	"fmt"
)

// core is interface representing different computing devices such as cpu, cuda, etc
type Core[T any] interface {
	NewTensorWithZeros(shape []int) Tensor
	NewTensorWithOnes(shape []int) Tensor
	NewTensorWithConstant(shape []int, value T) Tensor
	NewTensorWithRand(shape []int) Tensor
	NewTensorWithRandn(shape []int) Tensor
	NewTensorArange(start, end, step T) Tensor
	NewTensorLinspace(start, end T, num int) Tensor
	NewTensorEye(n int) Tensor
}

// Tensor interface with all tensor operations and creation functions.
type Tensor interface {
	Reshape(newShape []int) error
	GetShape() []int
}

// NewCore creates a new Core instance based on the device type.
func NewCore(deviceType string) (*CoreCPU, error) {
	switch deviceType {
	case "cpu":
		return &CoreCPU{}, nil
	case "cuda":
		return nil, fmt.Errorf("CUDA support is not implemented yet")
	case "opencl":
		return nil, fmt.Errorf("OpenCL support is not implemented yet")
	default:
		return nil, fmt.Errorf("unknown device type: %s", deviceType)
	}
}
