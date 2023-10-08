//
//  Resnet50.swift
//  NSSpain23
//
//  Created by Noah Martin on 8/30/23.
//

import Foundation
import MetalPerformanceShadersGraph

let gDevice = MTLCreateSystemDefaultDevice()!

class Resnet50 {

  private let modelQueue = DispatchQueue(label: "com.noah.modelqueue")

  init() {
    modelQueue.async {
      self.buildModel()
    }
  }

  func train(with batch: MPSGraphTensorData, batchLabels: MPSGraphTensorData, completion: @escaping (Float) -> Void) {
    modelQueue.async { [weak self] in
      guard let self else { return }

      guard let output, let updateOps, let loss, let inputPlaceholder, let outputPlaceholder else { return }

      let desc = MPSGraphExecutionDescriptor()
      desc.completionHandler = { results, error in
        var lossFloat: Float = 0
        results[loss]!.mpsndarray().readBytes(&lossFloat, strideBytes: nil)

        var outputFloat = [Float](repeating: 0, count: 10*64)
        results[output]!.mpsndarray().readBytes(&outputFloat, strideBytes: nil)
        completion(lossFloat)
      }
      let isTraining = MPSGraphTensorData(
        device: MPSGraphDevice(mtlDevice: gDevice),
        data: Data(bytes: [1], count: 1),
        shape: [1],
        dataType: .bool)

      graph.runAsync(
        feeds: [
          inputPlaceholder: batch,
          isTrainingPlaceholder: isTraining,
          outputPlaceholder: batchLabels,
        ],
        targetTensors: [loss, output],
        targetOperations: updateOps,
        executionDescriptor: desc)
    }
  }

  let graph = MPSGraph()
  private var inputPlaceholder: MPSGraphTensor?
  private var outputPlaceholder: MPSGraphTensor?
  private var output: MPSGraphTensor?
  private var loss: MPSGraphTensor?
  private var updateOps: [MPSGraphOperation]?
  lazy var isTrainingPlaceholder = graph.placeholder(shape: [1], dataType: .bool, name: nil)
  var weightTensors = [MPSGraphTensor]()

  private func identityBlock(_ sourceTensor: MPSGraphTensor, stride: Int, filters: (f1: Int, f2: Int)) -> MPSGraphTensor {
    let (f1, f2) = filters
    var x = conv2D(sourceTensor, filters: f1, kernelSize: (1, 1), strideSize: (1, 1), padding: .TF_VALID)
    x = batchNormalization(x)
    x = graph.reLU(with: x, name: nil)

    x = conv2D(x, filters: f1, kernelSize: (3, 3), strideSize: (1, 1), padding: .TF_SAME)
    x = batchNormalization(x)
    x = graph.reLU(with: x, name: nil)

    x = conv2D(x, filters: f2, kernelSize: (1, 1), strideSize: (1, 1), padding: .TF_VALID)
    x = batchNormalization(x)

    x = graph.addition(x, sourceTensor, name: nil)
    x = graph.reLU(with: x, name: nil)
    return x
  }

  private func convBlock(_ sourceTensor: MPSGraphTensor, stride: Int, filters: (f1: Int, f2: Int)) -> MPSGraphTensor {
    let (f1, f2) = filters
    var x = conv2D(sourceTensor, filters: f1, kernelSize: (1, 1), strideSize: (stride, stride), padding: .TF_VALID)
    x = batchNormalization(x)
    x = graph.reLU(with: x, name: nil)

    x = conv2D(x, filters: f1, kernelSize: (3, 3), strideSize: (1, 1), padding: .TF_SAME)
    x = batchNormalization(x)
    x = graph.reLU(with: x, name: nil)

    x = conv2D(x, filters: f2, kernelSize: (1, 1), strideSize: (1, 1), padding: .TF_VALID)
    x = batchNormalization(x)

    var x_skip = conv2D(sourceTensor, filters: f2, kernelSize: (1, 1), strideSize: (stride, stride), padding: .TF_VALID)
    x_skip = batchNormalization(x_skip)

    x = graph.addition(x, x_skip, name: nil)
    return graph.reLU(with: x, name: nil)
  }

  private func buildModel() {
    let batchSize = 64
    let inputPlaceholder = graph.placeholder(shape: [batchSize as NSNumber, 32, 32, 3], dataType: .float32, name: nil)
    let outputPlaceholder = graph.placeholder(shape: [batchSize as NSNumber, 10], dataType: .float32, name: nil)
    var x = conv2D(inputPlaceholder, filters: 2, kernelSize: (7, 7))
    x = batchNormalization(x)
    x = graph.reLU(with: x, name: nil)
    x = maxPooling2D(x, kernelSize: (3, 3), strideSize: (2, 2))

    x = convBlock(x, stride: 1, filters: (64, 256))
    x = identityBlock(x, stride: 1, filters: (64, 256))
    x = identityBlock(x, stride: 1, filters: (64, 256))
    x = convBlock(x, stride: 2, filters: (128, 512))
    x = identityBlock(x, stride: 1, filters: (128, 512))
    x = identityBlock(x, stride: 1, filters: (128, 512))
    x = identityBlock(x, stride: 1, filters: (128, 512))
    x = convBlock(x, stride: 2, filters: (256, 1024))
    x = identityBlock(x, stride: 1, filters: (256, 1024))
    x = identityBlock(x, stride: 1, filters: (256, 1024))
    x = identityBlock(x, stride: 1, filters: (256, 1024))
    x = identityBlock(x, stride: 1, filters: (256, 1024))
    x = identityBlock(x, stride: 1, filters: (256, 1024))
    x = convBlock(x, stride: 2, filters: (512, 2048))
    x = identityBlock(x, stride: 1, filters: (512, 2048))
    x = identityBlock(x, stride: 1, filters: (512, 2048))

    x = avgPooling2D(x, kernelSize: (2, 2), padding: .TF_SAME)
    let dataSize = x.shape![1...].totalElements
    x = graph.reshape(x, shape: [batchSize as NSNumber, dataSize as NSNumber], name: nil)
    x = fullyConnected(x, outputFeatures: 10)
    var (output, loss) = graph.fixedSoftmaxLoss(x, labels: outputPlaceholder)
    self.output = output
    loss = graph.division(loss, graph.constant(Double(batchSize), dataType: .float32), name: nil)

    let gradients = graph.gradients(of: loss, with: weightTensors, name: nil)
    let learningRate = graph.constant(0.01, dataType: .float32)
    var updateOps = [MPSGraphOperation]()
    for (key, value) in gradients {
      let newValue = graph.stochasticGradientDescent(learningRate: learningRate, values: key, gradient: value, name: nil)
      updateOps.append(graph.assign(key, tensor: newValue, name: nil))
    }

    self.inputPlaceholder = inputPlaceholder
    self.outputPlaceholder = outputPlaceholder
    self.loss = loss
    self.updateOps = updateOps
  }
}
