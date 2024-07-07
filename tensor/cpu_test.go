package tensor

import (
	"testing"
)

func TestTranspose(t *testing.T) {
	// Test case 1: Transpose of a 2x3 tensor
	data := []float32{
		1, 2, 3,
		4, 5, 6,
	}
	tensor := &TensorCPU{
		Shape:   []int{2, 3},
		Strides: calculateStrides([]int{2, 3}),
		Data:    data,
	}
	expectedData := []float32{
		1, 4,
		2, 5,
		3, 6,
	}
	expectedShape := []int{3, 2}

	transposeTensor, err := tensor.Transpose()
	if err != nil {
		t.Errorf("Error during transpose: %v", err)
	}
	if !equalShapes(transposeTensor.Shape, expectedShape) {
		t.Errorf("Expected shape %v, got %v", expectedShape, transposeTensor.Shape)
	}
	for i := range transposeTensor.Data {
		if transposeTensor.Data[i] != expectedData[i] {
			t.Errorf("Expected data %v, got %v", expectedData, transposeTensor.Data)
		}
	}

	// Test case 2: Transpose of a 3x2 tensor
	data = []float32{
		1, 2,
		3, 4,
		5, 6,
	}
	tensor = &TensorCPU{
		Shape:   []int{3, 2},
		Strides: calculateStrides([]int{3, 2}),
		Data:    data,
	}
	expectedData = []float32{
		1, 3, 5,
		2, 4, 6,
	}
	expectedShape = []int{2, 3}

	transposeTensor, err = tensor.Transpose()
	if err != nil {
		t.Errorf("Error during transpose: %v", err)
	}
	if !equalShapes(transposeTensor.Shape, expectedShape) {
		t.Errorf("Expected shape %v, got %v", expectedShape, transposeTensor.Shape)
	}
	for i := range transposeTensor.Data {
		if transposeTensor.Data[i] != expectedData[i] {
			t.Errorf("Expected data %v, got %v", expectedData, transposeTensor.Data)
		}
	}
}

func TestDeterminant(t *testing.T) {
	// Test case 1: Determinant of a 2x2 tensor
	data := []float32{
		1, 2,
		3, 4,
	}
	tensor := &TensorCPU{
		Shape:   []int{2, 2},
		Strides: calculateStrides([]int{2, 2}),
		Data:    data,
	}
	expectedDet := float32(-2)

	det, err := tensor.Determinant()
	if err != nil {
		t.Errorf("Error during determinant calculation: %v", err)
	}
	if det != expectedDet {
		t.Errorf("Expected determinant %v, got %v", expectedDet, det)
	}

	// Test case 2: Determinant of a 3x3 tensor
	data = []float32{
		1, 2, 3,
		0, 1, 4,
		5, 6, 0,
	}
	tensor = &TensorCPU{
		Shape:   []int{3, 3},
		Strides: calculateStrides([]int{3, 3}),
		Data:    data,
	}
	expectedDet = float32(1)

	det, err = tensor.Determinant()
	if err != nil {
		t.Errorf("Error during determinant calculation: %v", err)
	}
	if det != expectedDet {
		t.Errorf("Expected determinant %v, got %v", expectedDet, det)
	}
}

func TestFlatten(t *testing.T) {
	// Test case 1: Flatten a 2x2 tensor
	data := []float32{
		1, 2,
		3, 4,
	}
	tensor := &TensorCPU{
		Shape:   []int{2, 2},
		Strides: calculateStrides([]int{2, 2}),
		Data:    data,
	}
	expectedData := []float32{1, 2, 3, 4}
	expectedShape := []int{4}

	flattenedTensor := tensor.Flatten()
	if !equalShapes(flattenedTensor.Shape, expectedShape) {
		t.Errorf("Expected shape %v, got %v", expectedShape, flattenedTensor.Shape)
	}
	for i := range flattenedTensor.Data {
		if flattenedTensor.Data[i] != expectedData[i] {
			t.Errorf("Expected data %v, got %v", expectedData, flattenedTensor.Data)
		}
	}

	// Test case 2: Flatten a 3x2 tensor
	data = []float32{
		1, 2,
		3, 4,
		5, 6,
	}
	tensor = &TensorCPU{
		Shape:   []int{3, 2},
		Strides: calculateStrides([]int{3, 2}),
		Data:    data,
	}
	expectedData = []float32{1, 2, 3, 4, 5, 6}
	expectedShape = []int{6}

	flattenedTensor = tensor.Flatten()
	if !equalShapes(flattenedTensor.Shape, expectedShape) {
		t.Errorf("Expected shape %v, got %v", expectedShape, flattenedTensor.Shape)
	}
	for i := range flattenedTensor.Data {
		if flattenedTensor.Data[i] != expectedData[i] {
			t.Errorf("Expected data %v, got %v", expectedData, flattenedTensor.Data)
		}
	}
}

func TestSlice(t *testing.T) {
	// Test case 1: Slice a 3x3 tensor to get a 2x2 sub-tensor
	data := []float32{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
	}
	tensor := &TensorCPU{
		Shape:   []int{3, 3},
		Strides: calculateStrides([]int{3, 3}),
		Data:    data,
	}
	expectedData := []float32{
		5, 6,
		8, 9,
	}
	expectedShape := []int{2, 2}

	slicedTensor, err := tensor.Slice([2]int{1, 3}, [2]int{1, 3})
	if err != nil {
		t.Errorf("Error during slicing: %v", err)
	}
	if !equalShapes(slicedTensor.Shape, expectedShape) {
		t.Errorf("Expected shape %v, got %v", expectedShape, slicedTensor.Shape)
	}
	for i := range slicedTensor.Data {
		if slicedTensor.Data[i] != expectedData[i] {
			t.Errorf("Expected data %v, got %v", expectedData, slicedTensor.Data)
		}
	}

	// Test case 2: Slice a 3x3 tensor to get a 1x3 sub-tensor
	expectedData = []float32{
		4, 5, 6,
	}
	expectedShape = []int{1, 3}

	slicedTensor, err = tensor.Slice([2]int{1, 2}, [2]int{0, 3})
	if err != nil {
		t.Errorf("Error during slicing: %v", err)
	}
	if !equalShapes(slicedTensor.Shape, expectedShape) {
		t.Errorf("Expected shape %v, got %v", expectedShape, slicedTensor.Shape)
	}
	for i := range slicedTensor.Data {
		if slicedTensor.Data[i] != expectedData[i] {
			t.Errorf("Expected data %v, got %v", expectedData, slicedTensor.Data)
		}
	}
}

func TestNewTensorWithConstant(t *testing.T) {
	// Test case 1: Creating a 2x2 tensor with all elements set to 3.14
	shape := []int{2, 2}
	value := float32(3.14)
	expectedData := []float32{3.14, 3.14, 3.14, 3.14}

	tensor := NewTensorWithConstant(shape, value)

	if !equalShapes(tensor.Shape, shape) {
		t.Errorf("Expected shape %v, got %v", shape, tensor.Shape)
	}
	for i, v := range tensor.Data {
		if v != expectedData[i] {
			t.Errorf("Expected data %v, got %v", expectedData, tensor.Data)
		}
	}

	// Test case 2: Creating a 3x3 tensor with all elements set to 1.23
	shape = []int{3, 3}
	value = float32(1.23)
	expectedData = []float32{1.23, 1.23, 1.23, 1.23, 1.23, 1.23, 1.23, 1.23, 1.23}

	tensor = NewTensorWithConstant(shape, value)

	if !equalShapes(tensor.Shape, shape) {
		t.Errorf("Expected shape %v, got %v", shape, tensor.Shape)
	}
	for i, v := range tensor.Data {
		if v != expectedData[i] {
			t.Errorf("Expected data %v, got %v", expectedData, tensor.Data)
		}
	}
}
func TestGetShape(t *testing.T) {
	// Test case 1: Check shape of a 2x2 tensor
	tensor := &TensorCPU{
		Shape:   []int{2, 2},
		Strides: calculateStrides([]int{2, 2}),
		Data:    []float32{1, 2, 3, 4},
	}
	expectedShape := []int{2, 2}

	resultShape := tensor.GetShape()
	if !equalShapes(resultShape, expectedShape) {
		t.Errorf("Expected shape %v, got %v", expectedShape, resultShape)
	}

	// Test case 2: Check shape of a 3x3 tensor
	tensor = &TensorCPU{
		Shape:   []int{3, 3},
		Strides: calculateStrides([]int{3, 3}),
		Data:    []float32{1, 2, 3, 4, 5, 6, 7, 8, 9},
	}
	expectedShape = []int{3, 3}

	resultShape = tensor.GetShape()
	if !equalShapes(resultShape, expectedShape) {
		t.Errorf("Expected shape %v, got %v", expectedShape, resultShape)
	}
}

func TestReshape(t *testing.T) {
	// Test case 1: Reshape a 2x2 tensor to 1x4
	data := []float32{1, 2, 3, 4}
	tensor := &TensorCPU{
		Shape:   []int{2, 2},
		Strides: calculateStrides([]int{2, 2}),
		Data:    data,
	}
	newShape := []int{1, 4}

	err := tensor.Reshape(newShape)
	if err != nil {
		t.Errorf("Error during reshape: %v", err)
	}
	if !equalShapes(tensor.Shape, newShape) {
		t.Errorf("Expected shape %v, got %v", newShape, tensor.Shape)
	}

	// Test case 2: Reshape a 3x3 tensor to 1x9
	data = []float32{1, 2, 3, 4, 5, 6, 7, 8, 9}
	tensor = &TensorCPU{
		Shape:   []int{3, 3},
		Strides: calculateStrides([]int{3, 3}),
		Data:    data,
	}
	newShape = []int{1, 9}

	err = tensor.Reshape(newShape)
	if err != nil {
		t.Errorf("Error during reshape: %v", err)
	}
	if !equalShapes(tensor.Shape, newShape) {
		t.Errorf("Expected shape %v, got %v", newShape, tensor.Shape)
	}
}

func TestNewTensorWithZeros(t *testing.T) {
	// Test case 1: Creating a 2x2 tensor filled with zeros
	shape := []int{2, 2}
	expectedData := []float32{0, 0, 0, 0}
	core, _ := NewCore("cpu")
	tensor := core.NewTensorWithZeros(shape)

	if !equalShapes(tensor.Shape, shape) {
		t.Errorf("Expected shape %v, got %v", shape, tensor.Shape)
	}
	for i, v := range tensor.Data {
		if v != expectedData[i] {
			t.Errorf("Expected data %v, got %v", expectedData, tensor.Data)
		}
	}

	// Test case 2: Creating a 3x3 tensor filled with zeros
	shape = []int{3, 3}
	expectedData = []float32{0, 0, 0, 0, 0, 0, 0, 0, 0}

	tensor = core.NewTensorWithZeros(shape)

	if !equalShapes(tensor.Shape, shape) {
		t.Errorf("Expected shape %v, got %v", shape, tensor.Shape)
	}
	for i, v := range tensor.Data {
		if v != expectedData[i] {
			t.Errorf("Expected data %v, got %v", expectedData, tensor.Data)
		}
	}
}

func TestNewTensorWithOnes(t *testing.T) {
	// Test case 1: Creating a 2x2 tensor filled with ones
	shape := []int{2, 2}
	expectedData := []float32{1, 1, 1, 1}
	core, _ := NewCore("cpu")

	tensor := core.NewTensorWithOnes(shape)

	if !equalShapes(tensor.Shape, shape) {
		t.Errorf("Expected shape %v, got %v", shape, tensor.Shape)
	}
	for i, v := range tensor.Data {
		if v != expectedData[i] {
			t.Errorf("Expected data %v, got %v", expectedData, tensor.Data)
		}
	}

	// Test case 2: Creating a 3x3 tensor filled with ones
	shape = []int{3, 3}
	expectedData = []float32{1, 1, 1, 1, 1, 1, 1, 1, 1}

	tensor = core.NewTensorWithOnes(shape)

	if !equalShapes(tensor.Shape, shape) {
		t.Errorf("Expected shape %v, got %v", shape, tensor.Shape)
	}
	for i, v := range tensor.Data {
		if v != expectedData[i] {
			t.Errorf("Expected data %v, got %v", expectedData, tensor.Data)
		}
	}
}

func TestNewTensorWithRand(t *testing.T) {
	// Test case 1: Creating a 2x2 tensor with random values
	shape := []int{2, 2}
	core, _ := NewCore("cpu")

	tensor := core.NewTensorWithRand(shape)

	if !equalShapes(tensor.Shape, shape) {
		t.Errorf("Expected shape %v, got %v", shape, tensor.Shape)
	}
	for _, v := range tensor.Data {
		if v < 0 || v > 1 {
			t.Errorf("Expected value between 0 and 1, got %v", v)
		}
	}

	// Test case 2: Creating a 3x3 tensor with random values
	shape = []int{3, 3}
	tensor = core.NewTensorWithRand(shape)

	if !equalShapes(tensor.Shape, shape) {
		t.Errorf("Expected shape %v, got %v", shape, tensor.Shape)
	}
	for _, v := range tensor.Data {
		if v < 0 || v > 1 {
			t.Errorf("Expected value between 0 and 1, got %v", v)
		}
	}
}

func TestNewTensorWithRandn(t *testing.T) {
	// Test case 1: Creating a 2x2 tensor with normally distributed random values
	shape := []int{2, 2}
	core, _ := NewCore("cpu")

	tensor := core.NewTensorWithRandn(shape)

	if !equalShapes(tensor.Shape, shape) {
		t.Errorf("Expected shape %v, got %v", shape, tensor.Shape)
	}
	for _, v := range tensor.Data {
		// Just check if the value is a float32, the range could be anything since it's normal distribution
		if v == float32(int(v)) {
			t.Errorf("Expected normally distributed float32 value, got %v", v)
		}
	}

	// Test case 2: Creating a 3x3 tensor with normally distributed random values
	shape = []int{3, 3}

	tensor = core.NewTensorWithRandn(shape)

	if !equalShapes(tensor.Shape, shape) {
		t.Errorf("Expected shape %v, got %v", shape, tensor.Shape)
	}
	for _, v := range tensor.Data {
		// Just check if the value is a float32, the range could be anything since it's normal distribution
		if v == float32(int(v)) {
			t.Errorf("Expected normally distributed float32 value, got %v", v)
		}
	}
}

func TestNewTensorArange(t *testing.T) {
	// Test case 1: Creating a tensor with values from 0 to 5 with step 1
	start, end, step := float32(0), float32(5), float32(1)
	expectedData := []float32{0, 1, 2, 3, 4}
	expectedShape := []int{5}
	core, _ := NewCore("cpu")

	tensor := core.NewTensorArange(start, end, step)

	if !equalShapes(tensor.Shape, expectedShape) {
		t.Errorf("Expected shape %v, got %v", expectedShape, tensor.Shape)
	}
	for i, v := range tensor.Data {
		if v != expectedData[i] {
			t.Errorf("Expected data %v, got %v", expectedData, tensor.Data)
		}
	}

	// Test case 2: Creating a tensor with values from 1 to 4 with step 0.5
	start, end, step = float32(1), float32(4), float32(0.5)
	expectedData = []float32{1, 1.5, 2, 2.5, 3, 3.5}
	expectedShape = []int{6}

	tensor = core.NewTensorArange(start, end, step)

	if !equalShapes(tensor.Shape, expectedShape) {
		t.Errorf("Expected shape %v, got %v", expectedShape, tensor.Shape)
	}
	for i, v := range tensor.Data {
		if v != expectedData[i] {
			t.Errorf("Expected data %v, got %v", expectedData, tensor.Data)
		}
	}
}

func TestNewTensorLinspace(t *testing.T) {
	// Test case 1: Creating a tensor with linearly spaced elements between 0 and 10, inclusive, with 5 elements
	start, end, num := float32(0), float32(10), 5
	expectedData := []float32{0, 2.5, 5, 7.5, 10}
	expectedShape := []int{5}
	core, _ := NewCore("cpu")

	tensor := core.NewTensorLinspace(start, end, num)
	if !equalShapes(tensor.Shape, expectedShape) {
		t.Errorf("Expected shape %v, got %v", expectedShape, tensor.Shape)
	}
	for i, v := range tensor.Data {
		if v != expectedData[i] {
			t.Errorf("Expected data %v, got %v", expectedData, tensor.Data)
		}
	}

	// Test case 2: Creating a tensor with linearly spaced elements between 1 and 2, inclusive, with 3 elements
	start, end, num = float32(1), float32(2), 3
	expectedData = []float32{1, 1.5, 2}
	expectedShape = []int{3}

	tensor = core.NewTensorLinspace(start, end, num)

	if !equalShapes(tensor.Shape, expectedShape) {
		t.Errorf("Expected shape %v, got %v", expectedShape, tensor.Shape)
	}
	for i, v := range tensor.Data {
		if v != expectedData[i] {
			t.Errorf("Expected data %v, got %v", expectedData, tensor.Data)
		}
	}
}
func TestNewTensorEye(t *testing.T) {
	// Test case 1: Creating a 3x3 identity matrix
	n := 3
	expectedData := []float32{
		1, 0, 0,
		0, 1, 0,
		0, 0, 1,
	}
	expectedShape := []int{3, 3}
	core, _ := NewCore("cpu")

	tensor := core.NewTensorEye(n)

	if !equalShapes(tensor.Shape, expectedShape) {
		t.Errorf("Expected shape %v, got %v", expectedShape, tensor.Shape)
	}
	for i, v := range tensor.Data {
		if v != expectedData[i] {
			t.Errorf("Expected data %v, got %v", expectedData, tensor.Data)
		}
	}

	// Test case 2: Creating a 2x2 identity matrix
	n = 2
	expectedData = []float32{
		1, 0,
		0, 1,
	}
	expectedShape = []int{2, 2}

	tensor = core.NewTensorEye(n)

	if !equalShapes(tensor.Shape, expectedShape) {
		t.Errorf("Expected shape %v, got %v", expectedShape, tensor.Shape)
	}
	for i, v := range tensor.Data {
		if v != expectedData[i] {
			t.Errorf("Expected data %v, got %v", expectedData, tensor.Data)
		}
	}
}

func TestAdd(t *testing.T) {

	// Test case 1: Adding two 2x2 tensors
	a := &TensorCPU{
		Shape:   []int{2, 2},
		Strides: calculateStrides([]int{2, 2}),
		Data:    []float32{1, 2, 3, 4},
	}
	b := &TensorCPU{
		Shape:   []int{2, 2},
		Strides: calculateStrides([]int{2, 2}),
		Data:    []float32{5, 6, 7, 8},
	}
	expectedData := []float32{6, 8, 10, 12}

	result, err := AddCPU(a, b)
	if err != nil {
		t.Errorf("Error during addition: %v", err)
	}
	if !equalShapes(result.Shape, a.Shape) {
		t.Errorf("Expected shape %v, got %v", a.Shape, result.Shape)
	}
	for i, v := range result.Data {
		if v != expectedData[i] {
			t.Errorf("Expected data %v, got %v", expectedData, result.Data)
		}
	}

	// Test case 2: Adding two 3x3 tensors
	a = &TensorCPU{
		Shape:   []int{3, 3},
		Strides: calculateStrides([]int{3, 3}),
		Data:    []float32{1, 2, 3, 4, 5, 6, 7, 8, 9},
	}
	b = &TensorCPU{
		Shape:   []int{3, 3},
		Strides: calculateStrides([]int{3, 3}),
		Data:    []float32{9, 8, 7, 6, 5, 4, 3, 2, 1},
	}
	expectedData = []float32{10, 10, 10, 10, 10, 10, 10, 10, 10}

	result, err = AddCPU(a, b)
	if err != nil {
		t.Errorf("Error during addition: %v", err)
	}
	if !equalShapes(result.Shape, a.Shape) {
		t.Errorf("Expected shape %v, got %v", a.Shape, result.Shape)
	}
	for i, v := range result.Data {
		if v != expectedData[i] {
			t.Errorf("Expected data %v, got %v", expectedData, result.Data)
		}
	}
}

func TestSubtract(t *testing.T) {

	// Test case 1: Subtracting two 2x2 tensors
	a := &TensorCPU{
		Shape:   []int{2, 2},
		Strides: calculateStrides([]int{2, 2}),
		Data:    []float32{5, 6, 7, 8},
	}
	b := &TensorCPU{
		Shape:   []int{2, 2},
		Strides: calculateStrides([]int{2, 2}),
		Data:    []float32{1, 2, 3, 4},
	}
	expectedData := []float32{4, 4, 4, 4}

	result, err := SubtractCPU(a, b)
	if err != nil {
		t.Errorf("Error during subtraction: %v", err)
	}
	if !equalShapes(result.Shape, a.Shape) {
		t.Errorf("Expected shape %v, got %v", a.Shape, result.Shape)
	}
	for i, v := range result.Data {
		if v != expectedData[i] {
			t.Errorf("Expected data %v, got %v", expectedData, result.Data)
		}
	}

	// Test case 2: Subtracting two 3x3 tensors
	a = &TensorCPU{
		Shape:   []int{3, 3},
		Strides: calculateStrides([]int{3, 3}),
		Data:    []float32{9, 8, 7, 6, 5, 4, 3, 2, 1},
	}
	b = &TensorCPU{
		Shape:   []int{3, 3},
		Strides: calculateStrides([]int{3, 3}),
		Data:    []float32{1, 2, 3, 4, 5, 6, 7, 8, 9},
	}
	expectedData = []float32{8, 6, 4, 2, 0, -2, -4, -6, -8}

	result, err = SubtractCPU(a, b)
	if err != nil {
		t.Errorf("Error during subtraction: %v", err)
	}
	if !equalShapes(result.Shape, a.Shape) {
		t.Errorf("Expected shape %v, got %v", a.Shape, result.Shape)
	}
	for i, v := range result.Data {
		if v != expectedData[i] {
			t.Errorf("Expected data %v, got %v", expectedData, result.Data)
		}
	}
}

func TestMultiply(t *testing.T) {

	// Test case 1: Multiplying two 2x2 tensors
	a := &TensorCPU{
		Shape:   []int{2, 2},
		Strides: calculateStrides([]int{2, 2}),
		Data:    []float32{1, 2, 3, 4},
	}
	b := &TensorCPU{
		Shape:   []int{2, 2},
		Strides: calculateStrides([]int{2, 2}),
		Data:    []float32{5, 6, 7, 8},
	}
	expectedData := []float32{5, 12, 21, 32}

	result, err := MultiplyCPU(a, b)
	if err != nil {
		t.Errorf("Error during multiplication: %v", err)
	}
	if !equalShapes(result.Shape, a.Shape) {
		t.Errorf("Expected shape %v, got %v", a.Shape, result.Shape)
	}
	for i, v := range result.Data {
		if v != expectedData[i] {
			t.Errorf("Expected data %v, got %v", expectedData, result.Data)
		}
	}

	// Test case 2: Multiplying two 3x3 tensors
	a = &TensorCPU{
		Shape:   []int{3, 3},
		Strides: calculateStrides([]int{3, 3}),
		Data:    []float32{1, 2, 3, 4, 5, 6, 7, 8, 9},
	}
	b = &TensorCPU{
		Shape:   []int{3, 3},
		Strides: calculateStrides([]int{3, 3}),
		Data:    []float32{9, 8, 7, 6, 5, 4, 3, 2, 1},
	}
	expectedData = []float32{9, 16, 21, 24, 25, 24, 21, 16, 9}

	result, err = MultiplyCPU(a, b)
	if err != nil {
		t.Errorf("Error during multiplication: %v", err)
	}
	if !equalShapes(result.Shape, a.Shape) {
		t.Errorf("Expected shape %v, got %v", a.Shape, result.Shape)
	}
	for i, v := range result.Data {
		if v != expectedData[i] {
			t.Errorf("Expected data %v, got %v", expectedData, result.Data)
		}
	}
}

func TestDivide(t *testing.T) {
	// Test case 1: Dividing two 2x2 tensors
	a := &TensorCPU{
		Shape:   []int{2, 2},
		Strides: calculateStrides([]int{2, 2}),
		Data:    []float32{6, 8, 10, 12},
	}
	b := &TensorCPU{
		Shape:   []int{2, 2},
		Strides: calculateStrides([]int{2, 2}),
		Data:    []float32{2, 4, 5, 3},
	}
	expectedData := []float32{3, 2, 2, 4}

	result, err := DivideCPU(a, b)
	if err != nil {
		t.Errorf("Error during division: %v", err)
	}
	if !equalShapes(result.Shape, a.Shape) {
		t.Errorf("Expected shape %v, got %v", a.Shape, result.Shape)
	}
	for i, v := range result.Data {
		if v != expectedData[i] {
			t.Errorf("Expected data %v, got %v", expectedData, result.Data)
		}
	}

	// Test case 2: Dividing two 3x3 tensors
	a = &TensorCPU{
		Shape:   []int{3, 3},
		Strides: calculateStrides([]int{3, 3}),
		Data:    []float32{9, 8, 7, 6, 5, 4, 3, 2, 1},
	}
	b = &TensorCPU{
		Shape:   []int{3, 3},
		Strides: calculateStrides([]int{3, 3}),
		Data:    []float32{1, 2, 1, 2, 1, 2, 1, 2, 1},
	}
	expectedData = []float32{9, 4, 7, 3, 5, 2, 3, 1, 1}

	result, err = DivideCPU(a, b)
	if err != nil {
		t.Errorf("Error during division: %v", err)
	}
	if !equalShapes(result.Shape, a.Shape) {
		t.Errorf("Expected shape %v, got %v", a.Shape, result.Shape)
	}
	for i, v := range result.Data {
		if v != expectedData[i] {
			t.Errorf("Expected data %v, got %v", expectedData, result.Data)
		}
	}
}

func TestMatmul(t *testing.T) {
	// Test case 1: Multiplying two 2x2 matrices
	a := &TensorCPU{
		Shape:   []int{2, 2},
		Strides: calculateStrides([]int{2, 2}),
		Data:    []float32{1, 2, 3, 4},
	}
	b := &TensorCPU{
		Shape:   []int{2, 2},
		Strides: calculateStrides([]int{2, 2}),
		Data:    []float32{5, 6, 7, 8},
	}
	expectedData := []float32{19, 22, 43, 50}

	result, err := MatmulCPU(a, b)
	if err != nil {
		t.Errorf("Error during matrix multiplication: %v", err)
	}
	expectedShape := []int{2, 2}
	if !equalShapes(result.Shape, expectedShape) {
		t.Errorf("Expected shape %v, got %v", expectedShape, result.Shape)
	}
	for i, v := range result.Data {
		if v != expectedData[i] {
			t.Errorf("Expected data %v, got %v", expectedData, result.Data)
		}
	}

	// Test case 2: Multiplying a 2x3 matrix by a 3x2 matrix
	a = &TensorCPU{
		Shape:   []int{2, 3},
		Strides: calculateStrides([]int{2, 3}),
		Data:    []float32{1, 2, 3, 4, 5, 6},
	}
	b = &TensorCPU{
		Shape:   []int{3, 2},
		Strides: calculateStrides([]int{3, 2}),
		Data:    []float32{7, 8, 9, 10, 11, 12},
	}
	expectedData = []float32{58, 64, 139, 154}

	result, err = MatmulCPU(a, b)
	if err != nil {
		t.Errorf("Error during matrix multiplication: %v", err)
	}
	expectedShape = []int{2, 2}
	if !equalShapes(result.Shape, expectedShape) {
		t.Errorf("Expected shape %v, got %v", expectedShape, result.Shape)
	}
	for i, v := range result.Data {
		if v != expectedData[i] {
			t.Errorf("Expected data %v, got %v", expectedData, result.Data)
		}
	}
}

func TestDot(t *testing.T) {
	// Test case 1: Dot product of two vectors
	a := &TensorCPU{
		Shape:   []int{3},
		Strides: calculateStrides([]int{3}),
		Data:    []float32{1, 2, 3},
	}
	b := &TensorCPU{
		Shape:   []int{3},
		Strides: calculateStrides([]int{3}),
		Data:    []float32{4, 5, 6},
	}
	expectedDot := float32(32)

	result, err := DotCPU(a, b)
	if err != nil {
		t.Errorf("Error during dot product: %v", err)
	}
	if result != expectedDot {
		t.Errorf("Expected dot product %v, got %v", expectedDot, result)
	}

	// Test case 2: Dot product of another two vectors
	a = &TensorCPU{
		Shape:   []int{4},
		Strides: calculateStrides([]int{4}),
		Data:    []float32{1, 2, 3, 4},
	}
	b = &TensorCPU{
		Shape:   []int{4},
		Strides: calculateStrides([]int{4}),
		Data:    []float32{5, 6, 7, 8},
	}
	expectedDot = float32(70)

	result, err = DotCPU(a, b)
	if err != nil {
		t.Errorf("Error during dot product: %v", err)
	}
	if result != expectedDot {
		t.Errorf("Expected dot product %v, got %v", expectedDot, result)
	}
}

func TestSum(t *testing.T) {
	// Test case 1: Sum of a 2x2 tensor
	tensor := &TensorCPU{
		Shape:   []int{2, 2},
		Strides: calculateStrides([]int{2, 2}),
		Data:    []float32{1, 2, 3, 4},
	}
	expectedSum := float32(10)

	result := SumCPU(tensor)
	if result != expectedSum {
		t.Errorf("Expected sum %v, got %v", expectedSum, result)
	}

	// Test case 2: Sum of a 3x3 tensor
	tensor = &TensorCPU{
		Shape:   []int{3, 3},
		Strides: calculateStrides([]int{3, 3}),
		Data:    []float32{1, 2, 3, 4, 5, 6, 7, 8, 9},
	}
	expectedSum = float32(45)

	result = SumCPU(tensor)
	if result != expectedSum {
		t.Errorf("Expected sum %v, got %v", expectedSum, result)
	}
}

func TestMax(t *testing.T) {
	// Test case 1: Max of a 2x2 tensor
	tensor := &TensorCPU{
		Shape:   []int{2, 2},
		Strides: calculateStrides([]int{2, 2}),
		Data:    []float32{1, 2, 3, 4},
	}
	expectedMax := float32(4)

	result := MaxCPU(tensor)
	if result != expectedMax {
		t.Errorf("Expected max %v, got %v", expectedMax, result)
	}

	// Test case 2: Max of a 3x3 tensor
	tensor = &TensorCPU{
		Shape:   []int{3, 3},
		Strides: calculateStrides([]int{3, 3}),
		Data:    []float32{1, 2, 3, 4, 5, 6, 7, 8, 9},
	}
	expectedMax = float32(9)

	result = MaxCPU(tensor)
	if result != expectedMax {
		t.Errorf("Expected max %v, got %v", expectedMax, result)
	}
}

func TestMin(t *testing.T) {
	// Test case 1: Min of a 2x2 tensor
	tensor := &TensorCPU{
		Shape:   []int{2, 2},
		Strides: calculateStrides([]int{2, 2}),
		Data:    []float32{1, 2, 3, 4},
	}
	expectedMin := float32(1)

	result := MinCPU(tensor)
	if result != expectedMin {
		t.Errorf("Expected min %v, got %v", expectedMin, result)
	}

	// Test case 2: Min of a 3x3 tensor
	tensor = &TensorCPU{
		Shape:   []int{3, 3},
		Strides: calculateStrides([]int{3, 3}),
		Data:    []float32{9, 8, 7, 6, 5, 4, 3, 2, 1},
	}
	expectedMin = float32(1)

	result = MinCPU(tensor)
	if result != expectedMin {
		t.Errorf("Expected min %v, got %v", expectedMin, result)
	}
}

func TestProd(t *testing.T) {

	// Test case 1: Product of a 2x2 tensor
	tensor := &TensorCPU{
		Shape:   []int{2, 2},
		Strides: calculateStrides([]int{2, 2}),
		Data:    []float32{1, 2, 3, 4},
	}
	expectedProd := float32(24)

	result := ProdCPU(tensor)
	if result != expectedProd {
		t.Errorf("Expected product %v, got %v", expectedProd, result)
	}

	// Test case 2: Product of a 3x3 tensor
	tensor = &TensorCPU{
		Shape:   []int{3, 3},
		Strides: calculateStrides([]int{3, 3}),
		Data:    []float32{1, 2, 3, 4, 5, 6, 7, 8, 9},
	}
	expectedProd = float32(362880)

	result = ProdCPU(tensor)
	if result != expectedProd {
		t.Errorf("Expected product %v, got %v", expectedProd, result)
	}
}

func TestSqrt(t *testing.T) {

	// Test case 1: Square root of a 2x2 tensor
	tensor := &TensorCPU{
		Shape:   []int{2, 2},
		Strides: calculateStrides([]int{2, 2}),
		Data:    []float32{1, 4, 9, 16},
	}
	expectedData := []float32{1, 2, 3, 4}

	result := SqrtCPU(tensor)
	if !equalShapes(result.Shape, tensor.Shape) {
		t.Errorf("Expected shape %v, got %v", tensor.Shape, result.Shape)
	}
	for i, v := range result.Data {
		if v != expectedData[i] {
			t.Errorf("Expected data %v, got %v", expectedData, result.Data)
		}
	}

	// Test case 2: Square root of a 3x3 tensor
	tensor = &TensorCPU{
		Shape:   []int{3, 3},
		Strides: calculateStrides([]int{3, 3}),
		Data:    []float32{1, 4, 9, 16, 25, 36, 49, 64, 81},
	}
	expectedData = []float32{1, 2, 3, 4, 5, 6, 7, 8, 9}

	result = SqrtCPU(tensor)
	if !equalShapes(result.Shape, tensor.Shape) {
		t.Errorf("Expected shape %v, got %v", tensor.Shape, result.Shape)
	}
	for i, v := range result.Data {
		if v != expectedData[i] {
			t.Errorf("Expected data %v, got %v", expectedData, result.Data)
		}
	}
}

func TestLog(t *testing.T) {

	// Test case 1: Logarithm of a 2x2 tensor
	tensor := &TensorCPU{
		Shape:   []int{2, 2},
		Strides: calculateStrides([]int{2, 2}),
		Data:    []float32{1, 2, 3, 4},
	}
	expectedData := []float32{0, 0.6931472, 1.0986123, 1.3862944}

	result := LogCPU(tensor)
	if !equalShapes(result.Shape, tensor.Shape) {
		t.Errorf("Expected shape %v, got %v", tensor.Shape, result.Shape)
	}
	for i, v := range result.Data {
		if v != expectedData[i] {
			t.Errorf("Expected data %v, got %v", expectedData, result.Data)
		}
	}

	// Test case 2: Logarithm of a 3x3 tensor
	tensor = &TensorCPU{
		Shape:   []int{3, 3},
		Strides: calculateStrides([]int{3, 3}),
		Data:    []float32{1, 2, 3, 4, 5, 6, 7, 8, 9},
	}
	expectedData = []float32{
		0, 0.6931472, 1.0986123,
		1.3862944, 1.609438, 1.7917595,
		1.9459101, 2.0794415, 2.1972246,
	}

	result = LogCPU(tensor)
	if !equalShapes(result.Shape, tensor.Shape) {
		t.Errorf("Expected shape %v, got %v", tensor.Shape, result.Shape)
	}
	for i, v := range result.Data {
		if v != expectedData[i] {
			t.Errorf("Expected data %v, got %v", expectedData, result.Data)
		}
	}
}

func TestExp(t *testing.T) {
	// Test case 1: Exponential of a 2x2 tensor
	tensor := &TensorCPU{
		Shape:   []int{2, 2},
		Strides: calculateStrides([]int{2, 2}),
		Data:    []float32{0, 1, 2, 3},
	}
	expectedData := []float32{1, 2.7182817, 7.389056, 20.085537}

	result := ExpCPU(tensor)
	if !equalShapes(result.Shape, tensor.Shape) {
		t.Errorf("Expected shape %v, got %v", tensor.Shape, result.Shape)
	}
	for i, v := range result.Data {
		if v != expectedData[i] {
			t.Errorf("Expected data %v, got %v", expectedData, result.Data)
		}
	}

	// Test case 2: Exponential of a 3x3 tensor
	tensor = &TensorCPU{
		Shape:   []int{3, 3},
		Strides: calculateStrides([]int{3, 3}),
		Data:    []float32{0, 1, 2, 3, 4, 5, 6, 7, 8},
	}
	expectedData = []float32{
		1, 2.7182817, 7.389056,
		20.085537, 54.59815, 148.41316,
		403.4288, 1096.6332, 2980.958,
	}

	result = ExpCPU(tensor)
	if !equalShapes(result.Shape, tensor.Shape) {
		t.Errorf("Expected shape %v, got %v", tensor.Shape, result.Shape)
	}
	for i, v := range result.Data {
		if v != expectedData[i] {
			t.Errorf("Expected data %v, got %v", expectedData, result.Data)
		}
	}
}

func TestPow(t *testing.T) {

	// Test case 1: Power of 2 of a 2x2 tensor
	tensor := &TensorCPU{
		Shape:   []int{2, 2},
		Strides: calculateStrides([]int{2, 2}),
		Data:    []float32{1, 2, 3, 4},
	}
	power := float32(2)
	expectedData := []float32{1, 4, 9, 16}

	result := PowCPU(tensor, power)
	if !equalShapes(result.Shape, tensor.Shape) {
		t.Errorf("Expected shape %v, got %v", tensor.Shape, result.Shape)
	}
	for i, v := range result.Data {
		if v != expectedData[i] {
			t.Errorf("Expected data %v, got %v", expectedData, result.Data)
		}
	}

	// Test case 2: Power of 3 of a 3x3 tensor
	tensor = &TensorCPU{
		Shape:   []int{3, 3},
		Strides: calculateStrides([]int{3, 3}),
		Data:    []float32{1, 2, 3, 4, 5, 6, 7, 8, 9},
	}
	power = float32(3)
	expectedData = []float32{1, 8, 27, 64, 125, 216, 343, 512, 729}

	result = PowCPU(tensor, power)
	if !equalShapes(result.Shape, tensor.Shape) {
		t.Errorf("Expected shape %v, got %v", tensor.Shape, result.Shape)
	}
	for i, v := range result.Data {
		if v != expectedData[i] {
			t.Errorf("Expected data %v, got %v", expectedData, result.Data)
		}
	}
}

func TestGetSubMatrix(t *testing.T) {
	// Test case 1: 3x3 matrix removing row 1 and column 1
	data := []float32{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
	}
	subMatrix := make([]float32, 4)
	getSubMatrix(data, subMatrix, 3, 1, 1)
	expectedData := []float32{
		1, 3,
		7, 9,
	}
	for i, v := range subMatrix {
		if v != expectedData[i] {
			t.Errorf("Expected subMatrix %v, got %v", expectedData, subMatrix)
		}
	}

	// Test case 2: 4x4 matrix removing row 2 and column 2
	data = []float32{
		1, 2, 3, 4,
		5, 6, 7, 8,
		9, 10, 11, 12,
		13, 14, 15, 16,
	}
	subMatrix = make([]float32, 9)
	getSubMatrix(data, subMatrix, 4, 2, 2)
	expectedData = []float32{
		1, 2, 4,
		5, 6, 8,
		13, 14, 16,
	}
	for i, v := range subMatrix {
		if v != expectedData[i] {
			t.Errorf("Expected subMatrix %v, got %v", expectedData, subMatrix)
		}
	}
}

func TestCalculateTotalElements(t *testing.T) {
	// Test case 1: 2x2 matrix
	shape := []int{2, 2}
	expectedTotal := 4
	total := calculateTotalElements(shape)
	if total != expectedTotal {
		t.Errorf("Expected total elements %v, got %v", expectedTotal, total)
	}

	// Test case 2: 3x3x3 matrix
	shape = []int{3, 3, 3}
	expectedTotal = 27
	total = calculateTotalElements(shape)
	if total != expectedTotal {
		t.Errorf("Expected total elements %v, got %v", expectedTotal, total)
	}

	// Test case 3: Invalid shape
	shape = []int{3, 0, 3}
	expectedTotal = 0
	total = calculateTotalElements(shape)
	if total != expectedTotal {
		t.Errorf("Expected total elements %v, got %v", expectedTotal, total)
	}
}

func TestCalculateStrides(t *testing.T) {
	// Test case 1: 2x2 matrix
	shape := []int{2, 2}
	expectedStrides := []int{2, 1}
	strides := calculateStrides(shape)
	for i, v := range strides {
		if v != expectedStrides[i] {
			t.Errorf("Expected strides %v, got %v", expectedStrides, strides)
		}
	}

	// Test case 2: 3x3x3 matrix
	shape = []int{3, 3, 3}
	expectedStrides = []int{9, 3, 1}
	strides = calculateStrides(shape)
	for i, v := range strides {
		if v != expectedStrides[i] {
			t.Errorf("Expected strides %v, got %v", expectedStrides, strides)
		}
	}
}

func TestEqualShapes(t *testing.T) {
	// Test case 1: Same shapes
	shape1 := []int{2, 2}
	shape2 := []int{2, 2}
	expected := true
	result := equalShapes(shape1, shape2)
	if result != expected {
		t.Errorf("Expected %v, got %v", expected, result)
	}

	// Test case 2: Different shapes
	shape1 = []int{2, 2}
	shape2 = []int{2, 3}
	expected = false
	result = equalShapes(shape1, shape2)
	if result != expected {
		t.Errorf("Expected %v, got %v", expected, result)
	}

	// Test case 3: Different lengths
	shape1 = []int{2, 2}
	shape2 = []int{2, 2, 2}
	expected = false
	result = equalShapes(shape1, shape2)
	if result != expected {
		t.Errorf("Expected %v, got %v", expected, result)
	}
}
