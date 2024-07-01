package tensor

import (
	"errors"
	"math/rand"
	"time"
)

func init() {
	rand.Seed(time.Now().UnixNano())
}


// Tensor struct with specific float32 type
type TensorCPU struct {
	Shape   []int
	Strides []int
	Data    []float32
}

// CoreCPU struct implementing tensor operations
type CoreCPU struct{}

// NewTensorWithConstant creates a new tensor with all elements initialized to a specific value.
func NewTensorWithConstant(shape []int, value float32) *TensorCPU {
	total := 1
	for _, dim := range shape {
		total *= dim
	}
	data := make([]float32, total)
	for i := range data {
		data[i] = value
	}
	return &TensorCPU{
		Shape:   shape,
		Strides: calculateStrides(shape),
		Data:    data,
	}
}

// Reshape changes the shape of the tensor.
func (t *TensorCPU) Reshape(newShape []int) error {
	newTotal := 1
	for _, size := range newShape {
		newTotal *= size
	}
	if newTotal != len(t.Data) {
		return errors.New("new shape must contain the same total number of elements as the tensor")
	}
	t.Shape = newShape
	t.Strides = calculateStrides(newShape)
	return nil
}

// calculateStrides computes the strides for a given shape.
func calculateStrides(shape []int) []int {
	strides := make([]int, len(shape))
	stride := 1
	for i := len(shape) - 1; i >= 0; i-- {
		strides[i] = stride
		stride *= shape[i]
	}
	return strides
}

// GetShape returns the current shape of the tensor.
func (t *TensorCPU) GetShape() []int {
	return t.Shape
}

// NewTensorWithZeros creates a tensor filled with zeros.
func NewTensorWithZeros(shape []int) *TensorCPU {
	total := calculateTotalElements(shape)
	return &TensorCPU{
		Shape:   shape,
		Strides: calculateStrides(shape),
		Data:    make([]float32, total),
	}
}

// NewTensorWithOnes creates a tensor filled with ones.
func NewTensorWithOnes(shape []int) *TensorCPU {
	total := calculateTotalElements(shape)
	data := make([]float32, total)
	for i := range data {
		data[i] = 1
	}
	return &TensorCPU{
		Shape:   shape,
		Strides: calculateStrides(shape),
		Data:    data,
	}
}

// NewTensorWithRand creates a tensor with uniformly distributed random values between 0 and 1.
func NewTensorWithRand(shape []int) *TensorCPU {
	total := calculateTotalElements(shape)
	data := make([]float32, total)
	for i := range data {
		data[i] = float32(rand.Float64())
	}
	return &TensorCPU{
		Shape:   shape,
		Strides: calculateStrides(shape),
		Data:    data,
	}
}

// NewTensorWithRandn creates a tensor with normally distributed random values (mean = 0, stddev = 1).
func NewTensorWithRandn(shape []int) *TensorCPU {
	total := calculateTotalElements(shape)
	data := make([]float32, total)
	for i := range data {
		data[i] = float32(rand.NormFloat64())
	}
	return &TensorCPU{
		Shape:   shape,
		Strides: calculateStrides(shape),
		Data:    data,
	}
}

// NewTensorArange creates a tensor with values from start to end with a step increment.
func NewTensorArange(start, end, step float32) *TensorCPU {
	var data []float32
	for i := start; i < end; i += step {
		data = append(data, i)
	}
	shape := []int{len(data)}
	return &TensorCPU{
		Shape:   shape,
		Strides: calculateStrides(shape),
		Data:    data,
	}
}

// NewTensorLinspace creates a tensor with linearly spaced elements between start and end, inclusive.
func NewTensorLinspace(start, end float32, num int) *TensorCPU {
	data := make([]float32, num)
	if num == 1 {
		data[0] = start
	} else {
		delta := (end - start) / float32(num-1)
		for i := range data {
			data[i] = start + float32(i)*delta
		}
	}
	shape := []int{num}
	return &TensorCPU{
		Shape:   shape,
		Strides: calculateStrides(shape),
		Data:    data,
	}
}

// NewTensorEye creates an identity matrix of size n.
func NewTensorEye(n int) *TensorCPU {
	data := make([]float32, n*n)
	stride := n + 1
	for i := 0; i < len(data); i += stride {
		data[i] = 1
	}
	shape := []int{n, n}
	return &TensorCPU{
		Shape:   shape,
		Strides: calculateStrides(shape),
		Data:    data,
	}
}

// calculateTotalElements calculates the total number of elements that a tensor with the given shape would hold.
func calculateTotalElements(shape []int) int {
	total := 1
	for _, dim := range shape {
		if dim < 1 {
			return 0 // this case handles any non-positive dimension size, effectively ensuring all dimensions are valid
		}
		total *= dim
	}
	return total
}

// Implement the CoreCPU methods

func (c *CoreCPU) Reshape(t *TensorCPU, newShape []int) error {
	return t.Reshape(newShape)
}

func (c *CoreCPU) GetShape(t *TensorCPU) []int {
	return t.GetShape()
}

func (c *CoreCPU) NewTensorWithZeros(shape []int) *TensorCPU {
	return NewTensorWithZeros(shape)
}

func (c *CoreCPU) NewTensorWithOnes(shape []int) *TensorCPU {
	return NewTensorWithOnes(shape)
}

func (c *CoreCPU) NewTensorWithConstant(shape []int, value float32) *TensorCPU {
	return NewTensorWithConstant(shape, value)
}

func (c *CoreCPU) NewTensorWithRand(shape []int) *TensorCPU {
	return NewTensorWithRand(shape)
}

func (c *CoreCPU) NewTensorWithRandn(shape []int) *TensorCPU {
	return NewTensorWithRandn(shape)
}

func (c *CoreCPU) NewTensorArange(start, end, step float32) *TensorCPU {
	return NewTensorArange(start, end, step)
}

func (c *CoreCPU) NewTensorLinspace(start, end float32, num int) *TensorCPU {
	return NewTensorLinspace(start, end, num)
}

func (c *CoreCPU) NewTensorEye(n int) *TensorCPU {
	return NewTensorEye(n)
}
