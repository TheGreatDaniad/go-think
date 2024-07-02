package tensor

import (
	"fmt"
)

// core is interface representing different computing devices such as cpu, cuda, etc

type Core interface {
	NewTensorWithZeros(shape []int) *TensorCPU
	NewTensorWithOnes(shape []int) *TensorCPU
	NewTensorWithConstant(shape []int, value float32) *TensorCPU
	NewTensorWithRand(shape []int) *TensorCPU
	NewTensorWithRandn(shape []int) *TensorCPU
	NewTensorArange(start, end, step float32) *TensorCPU
	NewTensorLinspace(start, end float32, num int) *TensorCPU
	NewTensorEye(n int) *TensorCPU

	// Element-wise operations
	Add(a, b *TensorCPU) (*TensorCPU, error)
	Subtract(a, b *TensorCPU) (*TensorCPU, error)
	Multiply(a, b *TensorCPU) (*TensorCPU, error)
	Divide(a, b *TensorCPU) (*TensorCPU, error)

	// Matrix operations
	Matmul(a, b *TensorCPU) (*TensorCPU, error)
	Dot(a, b *TensorCPU) (float32, error)

	// Reduction operations
	Sum(t *TensorCPU) float32
	Mean(t *TensorCPU) float32
	Max(t *TensorCPU) float32
	Min(t *TensorCPU) float32
	Prod(t *TensorCPU) float32

	// Element-wise functions
	Sqrt(t *TensorCPU) *TensorCPU
	Log(t *TensorCPU) *TensorCPU
	Exp(t *TensorCPU) *TensorCPU
	Pow(t *TensorCPU, power float32) *TensorCPU
}

// Tensor interface with all tensor operations and creation functions.
type Tensor interface {
	Reshape(newShape []int) error
	GetShape() []int
	Transpose() (*TensorCPU, error)
	Determinant() (float32, error)
	Flatten() *TensorCPU
	Slice(ranges ...[2]int) (*TensorCPU, error)
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
