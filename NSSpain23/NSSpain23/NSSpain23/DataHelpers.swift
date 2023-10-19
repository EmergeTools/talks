//
//  DataHelpers.swift
//  NSSpain23
//
//  Created by Noah Martin on 8/30/23.
//

import Foundation
import MetalPerformanceShadersGraph

struct RandomNumberGeneratorWithSeed: RandomNumberGenerator {
    init(seed: Int) {
        // Set the random seed
        srand48(seed)
    }

    func next() -> UInt64 {
        // drand48() returns a Double, transform to UInt64
        return withUnsafeBytes(of: drand48()) { bytes in
            bytes.load(as: UInt64.self)
        }
    }
}

func getRandomData(numValues: Int, minimum: Float, maximum: Float) -> [Float] {
  var rng = RandomNumberGeneratorWithSeed(seed: 1234)
  return (1...numValues).map { _ in Float.random(in: minimum..<maximum, using: &rng) }
}

extension Sequence where Element == NSNumber {
  var totalElements: Int {
    self.reduce(1) { $0 * $1.intValue }
  }
}

extension Array where Element == Float {
  var data: Data {
    self.withUnsafeBytes { pointer in
      Data(bytes: pointer.baseAddress!, count: count * 4)
    }
  }

  func constantTensor(graph: MPSGraph, shape: [NSNumber]) -> MPSGraphTensor {
    assert(shape.totalElements == count)

    return graph.constant(data, shape: shape, dataType: .float32)
  }

  func variableTensor(graph: MPSGraph, shape: [NSNumber], name: String? = nil) -> MPSGraphTensor {
    assert(shape.totalElements == count)

    return graph.variable(with: data, shape: shape, dataType: .float32, name: name)
  }
}

extension MPSGraph {
  func uniformWeights(shape: [NSNumber], value: Float) -> MPSGraphTensor {
    return [Float](repeating: value, count: shape.totalElements).variableTensor(graph: self, shape: shape)
  }

  func randomWeights(shape: [NSNumber], fanIn: Int, fanOut: Int) -> MPSGraphTensor {
    let limit = sqrt(6/Float(fanIn + fanOut))
    return getRandomData(numValues: shape.totalElements, minimum: -limit, maximum: limit)
      .variableTensor(graph: self, shape: shape)
  }
}
