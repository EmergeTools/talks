//
//  Resnet50+Layers.swift
//  NSSpain23
//
//  Created by Noah Martin on 9/1/23.
//

import Foundation
import MetalPerformanceShadersGraph

extension Resnet50 {
  static func addMovingVariable(
    _ graph: MPSGraph,
    newValue: MPSGraphTensor,
    existingValue: MPSGraphTensor) -> MPSGraphOperation
  {
    let momentumTensor = graph.constant(0.99, dataType: .float32)
    let subtractMomentumTensor = graph.constant(1-0.99, dataType: .float32)
    let scaledNew = graph.multiplication(newValue, subtractMomentumTensor, name: nil)
    let scaledExisting = graph.multiplication(existingValue, momentumTensor, name: nil)
    let newValue = graph.addition(scaledNew, scaledExisting, name: nil)
    return graph.assign(existingValue, tensor: newValue, name: nil)
  }

  func batchNormalization(_ input: MPSGraphTensor) -> MPSGraphTensor {
    let outputFeatures = input.shape![3].intValue
    let beta = graph.uniformWeights(shape: [outputFeatures as NSNumber], value: 0)
    let gamma = graph.uniformWeights(shape: [outputFeatures as NSNumber], value: 1)
    weightTensors += [beta, gamma]
    let shape = [1, 1, 1, NSNumber(value: outputFeatures)]
    let movingMean = graph.uniformWeights(shape: shape, value: 0)
    let movingVar = graph.uniformWeights(shape: shape, value: 1)

    return graph.if(isTrainingPlaceholder, then: {
      let axes: [NSNumber] = [0, 1, 2]
      let mean = self.graph.mean(of: input, axes: axes, name: nil)
      let variance = self.graph.variance(of: input, mean: mean, axes: axes, name: nil)
      let meanAssign = Self.addMovingVariable(self.graph, newValue: mean, existingValue: movingMean)
      let varAssign = Self.addMovingVariable(self.graph, newValue: variance, existingValue: movingVar)
      return self.graph.controlDependency(with: [meanAssign, varAssign], dependentBlock: {
        return [self.graph.normalize(input, mean: mean, variance: variance, gamma: gamma, beta: beta, epsilon: 0.001, name: nil)]
      }, name: nil)
    }, else: {
      let epsilonTensor = [Float](repeating: 0.001, count: 1).constantTensor(graph: self.graph, shape: [1, 1, 1, 1])
      let subtractedMean = self.graph.subtraction(input, movingMean, name: nil)
      let varPlusEpsilon = self.graph.addition(movingVar, epsilonTensor, name: nil)
      let stdDev = self.graph.squareRoot(with: varPlusEpsilon, name: nil)
      let normalizedInput = self.graph.division(subtractedMean, stdDev, name: nil)
      let scaled = self.graph.multiplication(normalizedInput, gamma, name: nil)
      return [self.graph.addition(scaled, beta, name: nil)]
    }, name: nil)[0]
  }

  // Always use bias on FC layer
  func fullyConnected(_ input: MPSGraphTensor, outputFeatures: Int, name: String? = nil) -> MPSGraphTensor {
    assert(input.shape!.count == 2)

    let weightsShape = [input.shape![1], NSNumber(value: outputFeatures)]

    let fanIn = input.shape![0].intValue * weightsShape[0].intValue
    let fanOut = input.shape![0].intValue * weightsShape[1].intValue
    let fcWeights = graph.randomWeights(shape: weightsShape, fanIn: fanIn, fanOut: fanOut)

    let fcBiases = graph.uniformWeights(shape: [outputFeatures as NSNumber], value: 0)

    let fcTensor = graph.matrixMultiplication(primary: input,
                                                secondary: fcWeights,
                                                name: nil)

    let fcBiasTensor = graph.addition(fcTensor,
                                        fcBiases,
                                        name: nil)

    weightTensors += [fcWeights, fcBiases]

    return fcBiasTensor
  }

  func maxPooling2D(
    _ input: MPSGraphTensor,
    kernelSize: (width: Int, height: Int),
    strideSize: (width: Int, height: Int),
    padding: MPSGraphPaddingStyle = .TF_VALID) -> MPSGraphTensor
  {
    let descriptor = MPSGraphPooling2DOpDescriptor(
      kernelWidth: kernelSize.width,
      kernelHeight: kernelSize.height,
      strideInX: strideSize.width,
      strideInY: strideSize.height,
      paddingStyle: padding,
      dataLayout: .NHWC)!
    return graph.maxPooling2D(withSourceTensor: input, descriptor: descriptor, name: nil)
  }

  func avgPooling2D(
    _ input: MPSGraphTensor,
    kernelSize: (width: Int, height: Int),
    strideSize: (width: Int, height: Int)? = nil,
    padding: MPSGraphPaddingStyle = .TF_VALID) -> MPSGraphTensor {
    let descriptor = MPSGraphPooling2DOpDescriptor(
      kernelWidth: kernelSize.width,
      kernelHeight: kernelSize.height,
      strideInX: strideSize?.width ?? kernelSize.width,
      strideInY: strideSize?.height ?? kernelSize.height,
      paddingStyle: padding,
      dataLayout: .NHWC)!
    return graph.avgPooling2D(withSourceTensor: input, descriptor: descriptor, name: nil)
  }

  func kernelOutputSize(
    in sourceShape: [NSNumber],
    kernel: (width: Int, height: Int),
    stride: (width: Int, height: Int),
    padding: MPSGraphPaddingStyle) -> (width: Int, height: Int)
  {
    var outputWidth = Int(Float(sourceShape[1].intValue - kernel.width)/Float(stride.width) + 1)
    var outputHeight = Int(Float(sourceShape[2].intValue - kernel.height)/Float(stride.height) + 1)
    if padding == MPSGraphPaddingStyle.TF_SAME {
      outputWidth = Int(ceil(Float(sourceShape[1].intValue) / Float(stride.width)))
      outputHeight = Int(ceil(Float(sourceShape[2].intValue) / Float(stride.height)))
    }
    return (outputWidth, outputHeight)
  }

  // No bias on conv layer
  func conv2D(
    _ sourceTensor: MPSGraphTensor,
    filters: Int,
    kernelSize: (width: Int, height: Int),
    strideSize: (width: Int, height: Int) = (1, 1),
    name: String? = nil,
    padding: MPSGraphPaddingStyle = .TF_VALID) -> MPSGraphTensor
  {
    let desc = MPSGraphConvolution2DOpDescriptor(
      strideInX: strideSize.width,
      strideInY: strideSize.height,
      dilationRateInX: 1,
      dilationRateInY: 1,
      groups: 1,
      paddingStyle: padding,
      dataLayout: .NHWC,
      weightsLayout: .HWIO)!

    let fanIn = sourceTensor.shape!.totalElements
    let output: (width: Int, height: Int)
    output = kernelOutputSize(in: sourceTensor.shape!, kernel: kernelSize, stride: strideSize, padding: padding)
    let fanOut = sourceTensor.shape![0].intValue * output.width * output.height * filters

    let convWeights = graph.randomWeights(
      shape: [kernelSize.height as NSNumber, kernelSize.width as NSNumber, sourceTensor.shape![3], filters as NSNumber],
      fanIn: fanIn,
      fanOut: fanOut)
    weightTensors += [convWeights]
    return graph.convolution2D(sourceTensor, weights: convWeights, descriptor: desc, name: name)
  }
}
