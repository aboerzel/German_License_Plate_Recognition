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
import org.boerzel.glpr.ml.PlateDetectionModel
import org.boerzel.glpr.utils.Logger
import org.opencv.android.Utils
import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.image.BoundingBoxUtil
import org.tensorflow.lite.support.label.LabelUtil
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.BufferedReader
import java.io.IOException
import java.io.InputStreamReader
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.*


/**
 * Detects license plates from an image of a car using the Tensorflow Object Detection API:
 * github.com/tensorflow/models/tree/master/research/object_detection
 */
class PlateDetector @Throws(IOException::class)
constructor(context: Context) {
    private var labels: Vector<String>
    private var detectionModel: PlateDetectionModel

    // Initialize DetectionModel
    init {
        val assetManager = context.assets

        labels = loadLabels(assetManager)
        detectionModel = PlateDetectionModel.newInstance(context)
    }

    /** Load labels from label map file */
    @Throws(IOException::class)
    private fun loadLabels(assets: AssetManager): Vector<String> {
        val labels = Vector<String>()
        val inputStream = assets.open(LABELS_PATH)
        with(BufferedReader(InputStreamReader(inputStream))) {
            useLines { lines -> lines.forEach { labels.add(it) }}
            close()
        }
        return labels
    }

    /**
     * To detect a license plate in an image, follow these steps:
     * 1. pre-process the input image
     * 2. run inference with the license plate detection model
     *
     * @param bitmap
     * @return list containing the detections (bounding box, class, score)
     */
    fun detectPlates(bitmap: Bitmap): List<Detection> {
        // Pre-processing
        val inputByteBuffer = preprocess(bitmap)

        // Creates inputs for reference.
        val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 300, 300, 3), DataType.FLOAT32)
        inputFeature0.loadBuffer(inputByteBuffer)

        // Runs model inference and gets result.
        val outputs = detectionModel.process(inputFeature0)
        val outputLocations = outputs.outputFeature0AsTensorBuffer
        val outputClasses = outputs.outputFeature1AsTensorBuffer
        val outputScores = outputs.outputFeature2AsTensorBuffer
        val numDetections = outputs.outputFeature3AsTensorBuffer

        val locations = BoundingBoxUtil.convert(outputLocations, intArrayOf(0, 1, 2, 3), 2, BoundingBoxUtil.Type.BOUNDARIES, BoundingBoxUtil.CoordinateType.PIXEL, DIM_INPUT_HEIGHT, DIM_INPUT_WIDTH)

        // SSD Mobilenet V2 Model assumes class 0 is background class
        // in label file and class labels start from 1 to number_of_classes+1,
        // while outputClasses correspond to class index from 0 to number_of_classes
        val labelOffset = 1
        val labels = LabelUtil.mapValueToLabels(outputClasses, labels, labelOffset)

        // Label string
        //val labelIndex = outputClasses.getIntValue(0) + 1
        //var label = labels.elementAt(labelIndex);

        // Show the best detections after scaling them back to the input size.
        val detections = ArrayList<Detection>(numDetections.getIntValue(0))
        for (i in 0 until detections.count()) {
            val score = outputScores.getFloatValue(i)
            if (score < SCORE_THRESHOLD)
                continue

            val box = RectF(
                    locations[i].left * bitmap.width,
                    locations[i].top * bitmap.height,
                    locations[i].right * bitmap.width,
                    locations[i].bottom * bitmap.height)
            detections.add(Detection(i.toString(), labels[i], score, box))
        }

        return detections
    }

    /** Closes the interpreter */
    fun close() {
        detectionModel.close()
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

        // return the resized image as ByteBuffer
        return convertMatToTfLiteInput(resized)
    }

    /**
     * Convert plate image into byte buffer
     *
     * @image image
     * @result image as ByteBuffer
     * */
    private fun convertMatToTfLiteInput(image: Mat): ByteBuffer {
        val imgData = ByteBuffer.allocateDirect(DIM_BATCH_SIZE * DIM_INPUT_WIDTH * DIM_INPUT_HEIGHT * DIM_INPUT_DEPTH * NUM_BYTES_PER_CHANNEL)
        imgData.order(ByteOrder.nativeOrder())
        //imgData.rewind()

        for (i in 0 until DIM_INPUT_WIDTH) {
            for (j in 0 until DIM_INPUT_HEIGHT) {
                val pixelValue: Int = image[i, j][0].toInt()
                if (IS_QUANTIZED_MODEL) {
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
        private const val IS_QUANTIZED_MODEL = false

        // Name of the labels file (under /assets folder)
        private const val LABELS_PATH = "labelmap.txt"

        // Float model
        private const val IMAGE_MEAN = 128.0f
        private const val IMAGE_STD = 128.0f

        // Input size
        private const val DIM_BATCH_SIZE = 1        // batch size
        private const val DIM_INPUT_WIDTH = 300     // input image width
        private const val DIM_INPUT_HEIGHT = 300    // input image height
        private const val DIM_INPUT_DEPTH = 3       // 1 for gray scale & 3 for color images
        private val NUM_BYTES_PER_CHANNEL = if (IS_QUANTIZED_MODEL) 1 else 4 // Float model = 4 / Quantized model =  1

        private const val SCORE_THRESHOLD = 0.8

        private val LOGGER = Logger()
    }
}