//
//  CIFARDataSet.swift
//  NSSpain23
//
//  Created by Noah Martin on 9/4/23.
//

import Foundation
import MetalPerformanceShaders
import ZIPFoundation
import QuartzCore

class CIFARDataSet {

  let batchSize: Int
  static let inputDims = 32
  static let numClasses = 10

  var dataSet = [Float]()
  var labels = [Float]()
  private var data: Data
  private let device: MTLDevice

  let bytesPerImage = 3073

  private let trainingImageCount: Int

  var stepsPerEpoch: Int {
    let result = (Float(trainingImageCount) / Float(batchSize))
    return Int(result)
  }
  var order: [Int]
  var currentBatch = 0

  static func getDocumentsDirectory() -> URL {
      let paths = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)
      let documentsDirectory = paths[0]
      return documentsDirectory
  }

  init(device: MTLDevice, batchSize: Int = 64) {
    self.device = device
    self.batchSize = batchSize

    var dataURL = Self.getDocumentsDirectory().appending(component: "cifar-dataset")
    if !FileManager.default.fileExists(atPath: dataURL.path) {
      print("unzipping")
      // Copy files to documents
      let zipPath = URL(filePath: Bundle.main.path(forResource: "cifar-10-batches-bin", ofType: "zip")!)

      try! FileManager.default.unzipItem(at: zipPath, to: dataURL)
    }
    dataURL = dataURL.appending(component: "cifar-10-batches-bin")

    print(try! FileManager.default.contentsOfDirectory(atPath: dataURL.path))

    var data = Data()
    data.append(FileManager.default.contents(atPath: dataURL.appending(component: "data_batch_1.bin").path)!)
    data.append(FileManager.default.contents(atPath: dataURL.appending(component: "data_batch_2.bin").path)!)
    data.append(FileManager.default.contents(atPath: dataURL.appending(component: "data_batch_3.bin").path)!)
    data.append(FileManager.default.contents(atPath: dataURL.appending(component: "data_batch_4.bin").path)!)
    data.append(FileManager.default.contents(atPath: dataURL.appending(component: "data_batch_5.bin").path)!)
    self.data = data
    trainingImageCount = data.count / bytesPerImage

    order = Array(0..<trainingImageCount)
  }

  func prepare() {
    var rng = RandomNumberGeneratorWithSeed(seed: 1234)
    order.shuffle(using: &rng)

    let channels = 3
    labels = [Float](repeating: 0, count: Self.numClasses * trainingImageCount)
    dataSet = [Float](repeating: 0, count: Self.inputDims * Self.inputDims * channels * trainingImageCount)
    let pixelsInColor = Self.inputDims * Self.inputDims
    let pixelsInImage = pixelsInColor * channels
    let start = CACurrentMediaTime()
    for i in 0..<trainingImageCount {
      let dataStart = order[i] * bytesPerImage
      let dataSetStart = i * pixelsInImage
      labels[i*Self.numClasses + Int(data[dataStart])] = 1
      let imageStart = dataStart + 1
      for row in 0..<32 {
        let rowStart = dataSetStart + row * Self.inputDims * channels
        let rowDataStart = imageStart + row * Self.inputDims
        for column in 0..<32 {
          let columnStart = rowStart + column * channels
          let columnDataStart = rowDataStart + column
          for c in 0..<channels {
            // Plus 1 to account for label
            let pixelVal = data[columnDataStart + c*pixelsInColor]
            dataSet[columnStart + c] = Float(pixelVal)/255.0
          }
        }
      }
    }
    print("Time to prepare images \(CACurrentMediaTime() - start)")
  }

  func getInputData(tensor: MPSNDArray) {
    dataSet.withUnsafeMutableBufferPointer { buffer in
      let ptr = buffer.baseAddress!.advanced(by: currentBatch * batchSize * Self.inputDims * Self.inputDims * 3)
      tensor.writeBytes(UnsafeMutableRawPointer(ptr), strideBytes: nil)
    }
  }

  func getLabels(tensor: MPSNDArray) {
    labels.withUnsafeMutableBufferPointer { buffer in
      let ptr = buffer.baseAddress!.advanced(by: currentBatch * batchSize * Self.numClasses)
      tensor.writeBytes(UnsafeMutableRawPointer(ptr), strideBytes: nil)
    }
  }

  func advance() {
    currentBatch += 1
    if currentBatch >= stepsPerEpoch {
      currentBatch = 0
    }
  }
}
