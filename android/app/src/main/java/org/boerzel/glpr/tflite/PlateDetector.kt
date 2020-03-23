/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
package org.boerzel.glpr.tflite

import android.content.Context
import android.content.res.AssetManager
import android.graphics.Bitmap
import android.graphics.RectF
import org.opencv.android.Utils
import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import org.tensorflow.lite.Interpreter
import java.io.BufferedReader
import java.io.FileInputStream
import java.io.IOException
import java.io.InputStreamReader
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.util.*

/**
 * Detects license plates from an image of a car using the Tensorflow Object Detection API:
 * github.com/tensorflow/models/tree/master/research/object_detection
 */
class PlateDetector @Throws(IOException::class)
constructor(context: Context) {
    // TensorFlow Lite interpreter for running inference with the tflite model
    private lateinit var interpreter: Interpreter
    private var labels: Vector<String>
    private var isModelQuantized = false

    // Initialize TFLite interpreter
    init {
        // Load TFLite model
        val assetManager = context.assets
        val model = loadModelFile(assetManager)

        labels = loadLabels(assetManager)

        // Configure TFLite Interpreter options
        val options = Interpreter.Options()
        options.setNumThreads(NUM_THREADS)

        //val gpuDelegate = GpuDelegate()
        //options.addDelegate(gpuDelegate)
        //options.setAllowBufferHandleOutput(true)
        //options.setAllowFp16PrecisionForFp32(true)
        //options.setUseNNAPI(true)

        // Create & initialize TFLite interpreter
        try {
            interpreter = Interpreter(model, options)
        }
        catch (e: Exception)
        {
            print(e.message)
        }
    }

    /** Memory-map the model file in Assets.  */
    @Throws(IOException::class)
    private fun loadModelFile(assets: AssetManager): MappedByteBuffer {
        val fileDescriptor = assets.openFd(MODEL_PATH)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    @Throws(IOException::class)
    private fun loadLabels(assets: AssetManager): Vector<String> {
        val labels = Vector<String>()
        val inputStream = assets.open(LABELS_PATH)
        val br = BufferedReader(InputStreamReader(inputStream))
        br.useLines { lines -> lines.forEach { labels.add(it) }}
        br.close()
        return labels
    }

    /**
     * To detect a license plate in an image, follow these steps:
     * 1. pre-process the input image
     * 2. run inference with the license plate detection model
     *
     * @param bitmap
     * @return list containing the detections
     */
    fun detect_plates(bitmap: Bitmap): List<Detection> {
        // 1. Pre-processing
        val inputByteBuffer = preprocess(bitmap)
        val inputArray = arrayOf<Any?>(inputByteBuffer)

        val outputLocations = Array(1) { Array(NUM_DETECTIONS) { FloatArray(4) } }
        val outputClasses = Array(1) { FloatArray(NUM_DETECTIONS) }
        val outputScores = Array(1) { FloatArray(NUM_DETECTIONS) }
        val numDetections = FloatArray(1)
        val outputMap: MutableMap<Int, Any> = HashMap()
        outputMap[0] = outputLocations
        outputMap[1] = outputClasses
        outputMap[2] = outputScores
        outputMap[3] = numDetections

        interpreter.runForMultipleInputsOutputs(inputArray, outputMap)

        // Show the best detections.
        // after scaling them back to the input size.
        val detections = ArrayList<Detection>(NUM_DETECTIONS)
        for (i in 0 until NUM_DETECTIONS) {
            val detection = RectF(
                    outputLocations[0][i][1] * bitmap.width,
                    outputLocations[0][i][0] * bitmap.height,
                    outputLocations[0][i][3] * bitmap.width,
                    outputLocations[0][i][2] * bitmap.height)
            // SSD Mobilenet V2 Model assumes class 0 is background class
            // in label file and class labels start from 1 to number_of_classes+1,
            // while outputClasses correspond to class index from 0 to number_of_classes
            val labelOffset = 1
            detections.add(
                    Detection(
                            "" + i,
                            labels[outputClasses[0][i].toInt() + labelOffset],
                            outputScores[0][i],
                            detection))
        }

        return detections
    }

    fun close() {
        interpreter.close()
    }

    fun setNumThreads(num_threads: Int) {
        interpreter.setNumThreads(num_threads)
    }

    /**
     * Preprocess the bitmap:
     * 1. resize image to input size of the plate detection model
     * 2. convert image to ByteBuffer
     *
     * @param bitmap
     * @return preprocessed image as ByteBuffer
     */
    private fun preprocess(bitmap: Bitmap): ByteBuffer {
        val image = Mat()
        Utils.bitmapToMat(bitmap, image)

        val resized = Mat()
        val newSize = Size(DIM_INPUT_WIDTH.toDouble(), DIM_INPUT_HEIGHT.toDouble())
        Imgproc.resize(image, resized, newSize, 0.0, 0.0, Imgproc.INTER_AREA)

        return convertMatToTfLiteInput(resized)
    }

    private fun convertMatToTfLiteInput(image: Mat): ByteBuffer {
        val imgData = ByteBuffer.allocateDirect(DIM_BATCH_SIZE * DIM_INPUT_WIDTH * DIM_INPUT_HEIGHT * DIM_INPUT_DEPTH * FLOAT_TYPE_SIZE)
        imgData.order(ByteOrder.nativeOrder())
        //imgData.rewind()

        for (i in 0 until DIM_INPUT_WIDTH) {
            for (j in 0 until DIM_INPUT_HEIGHT) {
                val pixelValue: Int = image[i, j][0].toInt()
                if (isModelQuantized) {
                    // Quantized model
                    imgData.put((pixelValue shr 16 and 0xFF).toByte())
                    imgData.put((pixelValue shr 8 and 0xFF).toByte())
                    imgData.put((pixelValue and 0xFF).toByte())
                } else { // Float model
                    imgData.putFloat(((pixelValue shr 16 and 0xFF) - IMAGE_MEAN) / IMAGE_STD)
                    imgData.putFloat(((pixelValue shr 8 and 0xFF) - IMAGE_MEAN) / IMAGE_STD)
                    imgData.putFloat(((pixelValue and 0xFF) - IMAGE_MEAN) / IMAGE_STD)
                }
            }
        }

        return imgData
    }

    companion object {
        // Name of the model file (under /assets folder)
        private const val MODEL_PATH = "glpd-model.tflite"

        // Name of the labels file (under /assets folder)
        private const val LABELS_PATH = "labelmap.txt"

        // Only return this many results.
        private const val NUM_DETECTIONS = 1

        // Float model
        private const val IMAGE_MEAN = 128.0f
        private const val IMAGE_STD = 128.0f

        // Number of threads in the java app
        private const val NUM_THREADS = 1

        // Input size
        private const val DIM_BATCH_SIZE = 1      // batch size
        private const val DIM_INPUT_WIDTH = 300   // input image width
        private const val DIM_INPUT_HEIGHT = 300   // input image height
        private const val DIM_INPUT_DEPTH = 3     // 1 for gray scale & 3 for color images
        private const val FLOAT_TYPE_SIZE = 4
    }
}