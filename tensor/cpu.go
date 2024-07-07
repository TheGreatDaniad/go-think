package tensor

import (
	"errors"
	"math"
	"math/rand"
	"time"
)

func init() {
	rand.Seed(time.Now().UnixNano())
}

// Tensor struct with specific float32 type
type TensorCPU struct {
	Shape        []int
	Strides      []int
	Data         []float32
	Grad         []float32 // Change type to float32 for consistency
	RequiresGrad bool
	GradFn       func(grad *TensorCPU)
}

// Transpose returns the transpose of a 2D tensor.
func (t *TensorCPU) Transpose() (*TensorCPU, error) {
	if len(t.Shape) != 2 {
		return nil, errors.New("transpose is only defined for 2D tensors")
	}
	rows, cols := t.Shape[0], t.Shape[1]
	data := make([]float32, len(t.Data))
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			data[j*rows+i] = t.Data[i*cols+j]
		}
	}
	return &TensorCPU{
		Shape:   []int{cols, rows},
		Strides: calculateStrides([]int{cols, rows}),
		Data:    data,
	}, nil
}

// Determinant calculates the determinant of a 2D tensor.
func (t *TensorCPU) Determinant() (float32, error) {
	if len(t.Shape) != 2 || t.Shape[0] != t.Shape[1] {
		return 0, errors.New("determinant is only defined for square 2D tensors")
	}
	return determinantRecursive(t.Data, t.Shape[0]), nil
}

func determinantRecursive(data []float32, n int) float32 {
	if n == 1 {
		return data[0]
	}
	if n == 2 {
		return data[0]*data[3] - data[1]*data[2]
	}
	var det float32
	sign := float32(1.0)
	subMatrix := make([]float32, (n-1)*(n-1))
	for i := 0; i < n; i++ {
		getSubMatrix(data, subMatrix, n, 0, i)
		det += sign * data[i] * determinantRecursive(subMatrix, n-1)
		sign = -sign
	}
	return det
}

func getSubMatrix(data, subMatrix []float32, n, row, col int) {
	subRow, subCol := 0, 0
	for i := 0; i < n; i++ {
		if i == row {
			continue
		}
		subCol = 0
		for j := 0; j < n; j++ {
			if j == col {
				continue
			}
			subMatrix[subRow*(n-1)+subCol] = data[i*n+j]
			subCol++
		}
		subRow++
	}
}

// Flatten returns a flattened 1D tensor.
func (t *TensorCPU) Flatten() *TensorCPU {
	return &TensorCPU{
		Shape:   []int{len(t.Data)},
		Strides: []int{1},
		Data:    t.Data,
	}
}

// Slice returns a sub-tensor based on the provided ranges.
func (t *TensorCPU) Slice(ranges ...[2]int) (*TensorCPU, error) {
	if len(ranges) != len(t.Shape) {
		return nil, errors.New("number of ranges must match the number of dimensions")
	}
	newShape := make([]int, len(t.Shape))
	for i, r := range ranges {
		if r[0] < 0 || r[1] > t.Shape[i] || r[0] >= r[1] {
			return nil, errors.New("invalid slice range")
		}
		newShape[i] = r[1] - r[0]
	}

	newData := make([]float32, calculateTotalElements(newShape))
	copySlice(newData, t.Data, newShape, t.Strides, 0, ranges, t.Shape)
	return &TensorCPU{
		Shape:   newShape,
		Strides: calculateStrides(newShape),
		Data:    newData,
	}, nil
}

func copySlice(dst, src []float32, shape, strides []int, offset int, ranges [][2]int, srcShape []int) {
	if len(shape) == 1 {
		srcStart := offset + ranges[0][0]*strides[0]
		srcEnd := srcStart + shape[0]*strides[0]
		dstIndex := 0
		for i := srcStart; i < srcEnd; i += strides[0] {
			dst[dstIndex] = src[i]
			dstIndex++
		}
		return
	}
	for i := 0; i < shape[0]; i++ {
		newOffset := offset + (ranges[0][0]+i)*strides[0]
		copySlice(dst[i*shape[1]:], src, shape[1:], strides[1:], newOffset, ranges[1:], srcShape[1:])
	}
}

// Helper function to calculate total elements in a shape
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
func (c *CoreCPU) NewTensorWithZeros(shape []int) *TensorCPU {
	total := calculateTotalElements(shape)
	return &TensorCPU{
		Shape:   shape,
		Strides: calculateStrides(shape),
		Data:    make([]float32, total),
	}
}

// NewTensorWithOnes creates a tensor filled with ones.
func (c *CoreCPU) NewTensorWithOnes(shape []int) *TensorCPU {
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
func (c *CoreCPU) NewTensorWithRand(shape []int) *TensorCPU {
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
func (c *CoreCPU) NewTensorWithRandn(shape []int) *TensorCPU {
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
func (c *CoreCPU) NewTensorArange(start, end, step float32) *TensorCPU {
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
func (c *CoreCPU) NewTensorLinspace(start, end float32, num int) *TensorCPU {
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
func (c *CoreCPU) NewTensorEye(n int) *TensorCPU {
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
func AddCPU(a, b *TensorCPU) (*TensorCPU, error) {
	if !equalShapes(a.Shape, b.Shape) {
		return nil, errors.New("shapes do not match")
	}
	data := make([]float32, len(a.Data))
	for i := range data {
		data[i] = a.Data[i] + b.Data[i]
	}
	result := &TensorCPU{
		Shape:        a.Shape,
		Strides:      a.Strides,
		Data:         data,
		RequiresGrad: a.RequiresGrad || b.RequiresGrad,
	}
	if result.RequiresGrad {
		result.GradFn = func(grad *TensorCPU) {
			if a.RequiresGrad {
				a.Backward(grad)
			}
			if b.RequiresGrad {
				b.Backward(grad)
			}
		}
	}
	return result, nil
}

func SubtractCPU(a, b *TensorCPU) (*TensorCPU, error) {
	if !equalShapes(a.Shape, b.Shape) {
		return nil, errors.New("shapes do not match")
	}
	data := make([]float32, len(a.Data))
	for i := range data {
		data[i] = a.Data[i] - b.Data[i]
	}
	result := &TensorCPU{
		Shape:        a.Shape,
		Strides:      a.Strides,
		Data:         data,
		RequiresGrad: a.RequiresGrad || b.RequiresGrad,
	}
	if result.RequiresGrad {
		result.GradFn = func(grad *TensorCPU) {
			if a.RequiresGrad {
				a.Backward(grad)
			}
			if b.RequiresGrad {
				negGrad := &TensorCPU{
					Shape:   grad.Shape,
					Strides: grad.Strides,
					Data:    make([]float32, len(grad.Data)),
				}
				for i := range grad.Data {
					negGrad.Data[i] = -grad.Data[i]
				}
				b.Backward(negGrad)
			}
		}
	}
	return result, nil
}

func MultiplyCPU(a, b *TensorCPU) (*TensorCPU, error) {
	if !equalShapes(a.Shape, b.Shape) {
		return nil, errors.New("shapes do not match")
	}
	data := make([]float32, len(a.Data))
	for i := range data {
		data[i] = a.Data[i] * b.Data[i]
	}
	result := &TensorCPU{
		Shape:        a.Shape,
		Strides:      a.Strides,
		Data:         data,
		RequiresGrad: a.RequiresGrad || b.RequiresGrad,
	}
	if result.RequiresGrad {
		result.GradFn = func(grad *TensorCPU) {
			if a.RequiresGrad {
				aGrad := &TensorCPU{
					Shape:   grad.Shape,
					Strides: grad.Strides,
					Data:    make([]float32, len(grad.Data)),
				}
				for i := range grad.Data {
					aGrad.Data[i] = grad.Data[i] * b.Data[i]
				}
				a.Backward(aGrad)
			}
			if b.RequiresGrad {
				bGrad := &TensorCPU{
					Shape:   grad.Shape,
					Strides: grad.Strides,
					Data:    make([]float32, len(grad.Data)),
				}
				for i := range grad.Data {
					bGrad.Data[i] = grad.Data[i] * a.Data[i]
				}
				b.Backward(bGrad)
			}
		}
	}
	return result, nil
}

func DivideCPU(a, b *TensorCPU) (*TensorCPU, error) {
	if !equalShapes(a.Shape, b.Shape) {
		return nil, errors.New("shapes do not match")
	}
	data := make([]float32, len(a.Data))
	for i := range data {
		data[i] = a.Data[i] / b.Data[i]
	}
	result := &TensorCPU{
		Shape:        a.Shape,
		Strides:      a.Strides,
		Data:         data,
		RequiresGrad: a.RequiresGrad || b.RequiresGrad,
	}
	if result.RequiresGrad {
		result.GradFn = func(grad *TensorCPU) {
			if a.RequiresGrad {
				aGrad := &TensorCPU{
					Shape:   grad.Shape,
					Strides: grad.Strides,
					Data:    make([]float32, len(grad.Data)),
				}
				for i := range grad.Data {
					aGrad.Data[i] = grad.Data[i] / b.Data[i]
				}
				a.Backward(aGrad)
			}
			if b.RequiresGrad {
				bGrad := &TensorCPU{
					Shape:   grad.Shape,
					Strides: grad.Strides,
					Data:    make([]float32, len(grad.Data)),
				}
				for i := range grad.Data {
					bGrad.Data[i] = -grad.Data[i] * a.Data[i] / (b.Data[i] * b.Data[i])
				}
				b.Backward(bGrad)
			}
		}
	}
	return result, nil
}
func (t *TensorCPU) Backward(grad *TensorCPU) {
	if t.Grad == nil {
		t.Grad = make([]float32, len(t.Data))
	}
	for i := range t.Grad {
		t.Grad[i] += grad.Data[i]
	}
	if t.GradFn != nil {
		t.GradFn(grad)
	}
}

// Matrix operations

func MatmulCPU(a, b *TensorCPU) (*TensorCPU, error) {
	if len(a.Shape) != 2 || len(b.Shape) != 2 {
		return nil, errors.New("matrices must be 2-dimensional")
	}
	if a.Shape[1] != b.Shape[0] {
		return nil, errors.New("inner dimensions do not match")
	}
	m, n, p := a.Shape[0], a.Shape[1], b.Shape[1]
	data := make([]float32, m*p)
	for i := 0; i < m; i++ {
		for j := 0; j < p; j++ {
			sum := float32(0)
			for k := 0; k < n; k++ {
				sum += a.Data[i*n+k] * b.Data[k*p+j]
			}
			data[i*p+j] = sum
		}
	}
	result := &TensorCPU{
		Shape:        []int{m, p},
		Strides:      calculateStrides([]int{m, p}),
		Data:         data,
		RequiresGrad: a.RequiresGrad || b.RequiresGrad,
	}
	if result.RequiresGrad {
		result.GradFn = func(grad *TensorCPU) {
			if a.RequiresGrad {
				aGrad := &TensorCPU{
					Shape:   a.Shape,
					Strides: a.Strides,
					Data:    make([]float32, len(a.Data)),
				}
				for i := 0; i < m; i++ {
					for k := 0; k < n; k++ {
						for j := 0; j < p; j++ {
							aGrad.Data[i*n+k] += grad.Data[i*p+j] * b.Data[k*p+j]
						}
					}
				}
				a.Backward(aGrad)
			}
			if b.RequiresGrad {
				bGrad := &TensorCPU{
					Shape:   b.Shape,
					Strides: b.Strides,
					Data:    make([]float32, len(b.Data)),
				}
				for k := 0; k < n; k++ {
					for j := 0; j < p; j++ {
						for i := 0; i < m; i++ {
							bGrad.Data[k*p+j] += grad.Data[i*p+j] * a.Data[i*n+k]
						}
					}
				}
				b.Backward(bGrad)
			}
		}
	}
	return result, nil
}

func DotCPU(a, b *TensorCPU) (float32, error) {
	if len(a.Shape) != 1 || len(b.Shape) != 1 {
		return 0, errors.New("vectors must be 1-dimensional")
	}
	if a.Shape[0] != b.Shape[0] {
		return 0, errors.New("vector lengths do not match")
	}
	sum := float32(0)
	for i := range a.Data {
		sum += a.Data[i] * b.Data[i]
	}
	return sum, nil
}

// Reduction operations

func SumCPU(t *TensorCPU) float32 {
	sum := float32(0)
	for _, v := range t.Data {
		sum += v
	}
	return sum
}

func MeanCPU(t *TensorCPU) float32 {
	return SumCPU(t) / float32(len(t.Data))
}

func MaxCPU(t *TensorCPU) float32 {
	maxVal := t.Data[0]
	for _, v := range t.Data {
		if v > maxVal {
			maxVal = v
		}
	}
	return maxVal
}

func MinCPU(t *TensorCPU) float32 {
	minVal := t.Data[0]
	for _, v := range t.Data {
		if v < minVal {
			minVal = v
		}
	}
	return minVal
}

func ProdCPU(t *TensorCPU) float32 {
	prod := float32(1)
	for _, v := range t.Data {
		prod *= v
	}
	return prod
}

// Element-wise functions
func SqrtCPU(t *TensorCPU) *TensorCPU {
	data := make([]float32, len(t.Data))
	for i, v := range t.Data {
		data[i] = float32(math.Sqrt(float64(v)))
	}
	result := &TensorCPU{
		Shape:        t.Shape,
		Strides:      t.Strides,
		Data:         data,
		RequiresGrad: t.RequiresGrad,
	}
	if result.RequiresGrad {
		result.GradFn = func(grad *TensorCPU) {
			if t.RequiresGrad {
				tGrad := &TensorCPU{
					Shape:   grad.Shape,
					Strides: grad.Strides,
					Data:    make([]float32, len(grad.Data)),
				}
				for i := range grad.Data {
					tGrad.Data[i] = grad.Data[i] / (2 * data[i])
				}
				t.Backward(tGrad)
			}
		}
	}
	return result
}

func LogCPU(t *TensorCPU) *TensorCPU {
	data := make([]float32, len(t.Data))
	for i, v := range t.Data {
		data[i] = float32(math.Log(float64(v)))
	}
	result := &TensorCPU{
		Shape:        t.Shape,
		Strides:      t.Strides,
		Data:         data,
		RequiresGrad: t.RequiresGrad,
	}
	if result.RequiresGrad {
		result.GradFn = func(grad *TensorCPU) {
			if t.RequiresGrad {
				tGrad := &TensorCPU{
					Shape:   grad.Shape,
					Strides: grad.Strides,
					Data:    make([]float32, len(grad.Data)),
				}
				for i := range grad.Data {
					tGrad.Data[i] = grad.Data[i] / t.Data[i]
				}
				t.Backward(tGrad)
			}
		}
	}
	return result
}

func ExpCPU(t *TensorCPU) *TensorCPU {
	data := make([]float32, len(t.Data))
	for i, v := range t.Data {
		data[i] = float32(math.Exp(float64(v)))
	}
	result := &TensorCPU{
		Shape:        t.Shape,
		Strides:      t.Strides,
		Data:         data,
		RequiresGrad: t.RequiresGrad,
	}
	if result.RequiresGrad {
		result.GradFn = func(grad *TensorCPU) {
			if t.RequiresGrad {
				tGrad := &TensorCPU{
					Shape:   grad.Shape,
					Strides: grad.Strides,
					Data:    make([]float32, len(grad.Data)),
				}
				for i := range grad.Data {
					tGrad.Data[i] = grad.Data[i] * data[i]
				}
				t.Backward(tGrad)
			}
		}
	}
	return result
}

func PowCPU(t *TensorCPU, power float32) *TensorCPU {
	data := make([]float32, len(t.Data))
	for i, v := range t.Data {
		data[i] = float32(math.Pow(float64(v), float64(power)))
	}
	result := &TensorCPU{
		Shape:        t.Shape,
		Strides:      t.Strides,
		Data:         data,
		RequiresGrad: t.RequiresGrad,
	}
	if result.RequiresGrad {
		result.GradFn = func(grad *TensorCPU) {
			if t.RequiresGrad {
				tGrad := &TensorCPU{
					Shape:   grad.Shape,
					Strides: grad.Strides,
					Data:    make([]float32, len(grad.Data)),
				}
				for i := range grad.Data {
					tGrad.Data[i] = grad.Data[i] * power * float32(math.Pow(float64(t.Data[i]), float64(power-1)))
				}
				t.Backward(tGrad)
			}
		}
	}
	return result
}

// Helper functions
func equalShapes(shape1, shape2 []int) bool {
	if len(shape1) != len(shape2) {
		return false
	}
	for i := range shape1 {
		if shape1[i] != shape2[i] {
			return false
		}
	}
	return true
}
