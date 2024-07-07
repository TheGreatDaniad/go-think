package nn

import (
	"go-think/tensor"
	"math/rand"
)

type Linear struct {
	InputSize  int
	OutputSize int
	Weights    *tensor.TensorCPU
	Bias       *tensor.TensorCPU
}

func NewLinear(inputSize, outputSize int) *Linear {
	weights := &tensor.TensorCPU{
		Shape:        []int{inputSize, outputSize},
		Data:         make([]float32, inputSize*outputSize),
		RequiresGrad: true,
	}
	bias := &tensor.TensorCPU{
		Shape:        []int{outputSize},
		Data:         make([]float32, outputSize),
		RequiresGrad: true,
	}
	// Initialize weights and bias with random values
	for i := range weights.Data {
		weights.Data[i] = float32(rand.NormFloat64())
	}
	for i := range bias.Data {
		bias.Data[i] = float32(rand.NormFloat64())
	}
	return &Linear{inputSize, outputSize, weights, bias}
}

func (l *Linear) Forward(input *tensor.TensorCPU) *tensor.TensorCPU {
	// Matrix multiplication input * weights + bias
	output, _ := tensor.MatmulCPU(input, l.Weights)
	for i := 0; i < len(output.Data); i++ {
		output.Data[i] += l.Bias.Data[i%l.OutputSize]
	}
	if output.RequiresGrad {
		output.GradFn = func(grad *tensor.TensorCPU) {
			// Compute gradients for weights and bias
			weightsTranspose, err := l.Weights.Transpose()
			if err != nil {
				panic(err)
			}
			inputGrad, _ := tensor.MatmulCPU(grad, weightsTranspose)
			inputTranspose, err := input.Transpose()
			if err != nil {
				panic(err)
			}
			weightsGrad, _ := tensor.MatmulCPU(inputTranspose, grad)
			l.Weights.Backward(weightsGrad)
			l.Bias.Backward(grad)
			input.Backward(inputGrad)
		}
	}
	return output
}

func (l *Linear) Parameters() []*tensor.TensorCPU {
	return []*tensor.TensorCPU{l.Weights, l.Bias}
}
