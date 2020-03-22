/*
 * Copyright 2019 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.boerzel.glpr

import android.graphics.*
import android.media.ImageReader
import android.os.SystemClock
import android.util.Size
import android.view.Surface
import android.view.View
import org.boerzel.glpr.customview.OverlayView
import org.boerzel.glpr.env.Logger
import org.boerzel.glpr.tflite.LicenseRecognizer
import org.boerzel.glpr.tflite.PlateDetector
import java.io.IOException
import kotlin.math.roundToInt


class ClassifierActivity : CameraActivity(), ImageReader.OnImageAvailableListener {

    override val layoutId: Int
        get() = R.layout.camera_connection_fragment

    override val desiredPreviewFrameSize: Size?
        get() = Size(640, 480)

    private var plateDetector: PlateDetector? = null
    private var licenseRecognizer: LicenseRecognizer? = null

    private lateinit var rgbFrameBitmap: Bitmap
    private lateinit var trackingOverlay: OverlayView
    private lateinit var previewRoi: BoundingBox
    private lateinit var trackingOverlayRoi: RectF

    private val roiPaint = Paint()

    init {
        roiPaint.color = Color.GREEN
        roiPaint.alpha = 200
        roiPaint.style = Paint.Style.STROKE
        roiPaint.strokeWidth = 6.0f
    }

    public override fun onPreviewSizeChosen(size: Size, rotation: Int) {

        if (plateDetector == null) {
            try {
                LOGGER.d("Creating plateDetector")
                plateDetector = PlateDetector(this)
            } catch (e: IOException) {
                LOGGER.e(e, "Failed to create plateDetector.")
                throw e
            }
        }

        if (licenseRecognizer == null) {
            try {
                LOGGER.d("Creating licenseRecognizer")
                licenseRecognizer = LicenseRecognizer(this)
            } catch (e: IOException) {
                LOGGER.e(e, "Failed to create licenseRecognizer.")
                throw e
            }
        }

        previewWidth = size.width
        previewHeight = size.height
        LOGGER.i("Initializing at size %dx%d", previewWidth, previewHeight)
        rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Bitmap.Config.ARGB_8888)

        trackingOverlay = findViewById<View>(R.id.tracking_overlay) as OverlayView

        previewRoi = if (screenOrientationPortrait)
            getROI(previewHeight, previewWidth)
        else
            getROI(previewWidth, previewHeight)

        val roi = transformToTrackingOverlay(previewRoi)
        trackingOverlayRoi = RectF(roi.X.toFloat(), roi.Y.toFloat(), (roi.X + roi.WIDTH).toFloat(), (roi.Y + roi.HEIGHT).toFloat())

        //trackingOverlay.addCallback { canvas -> canvas.drawRoundRect(trackingOverlayRoi, cornerSize, cornerSize, roiPaint) }
    }

    private fun transformToTrackingOverlay(roi: BoundingBox) : BoundingBox {

        val scale = if (screenOrientationPortrait)
            trackingOverlay.width.toFloat() / previewHeight
        else
           trackingOverlay.width.toFloat() / previewWidth

        val w = (roi.WIDTH * scale).roundToInt()
        val x = ((trackingOverlay.width.toFloat() - w) / 2).roundToInt()
        val h = (w * licenseRecognizer!!.getInputAspectRatio()).roundToInt()
        val y = ((trackingOverlay.height.toFloat() - h) / 2).roundToInt()

        return BoundingBox(x, y, w, h)
    }

    override fun processImage() {
        rgbFrameBitmap.setPixels(getRgbBytes(), 0, previewWidth, 0, 0, previewWidth, previewHeight)

        //val plateBmp = cropROI(correctOrientation(rgbFrameBitmap), previewRoi)
        //trackingOverlay.postInvalidate()

        runInBackground(
                Runnable {
                    val startTime = SystemClock.uptimeMillis()

                    val detections = plateDetector!!.detect_plates(rgbFrameBitmap)
                    LOGGER.v("Detected license plates: %d", detections.count())

                    if (detections.count() > 0 && detections[0].confidence!! >= DETECTION_SCORE_THRESHOLD) {
                        LOGGER.v("Detected license plate: %s", detections[0].toString())
                        val location = detections[0].getLocation()
                        val plateBmp = cropLicensePlate(rgbFrameBitmap, location)
                        val license = licenseRecognizer!!.recognize(plateBmp)
                        LOGGER.v("Recognized license: %s", license)

                        runOnUiThread { showResult(license) }
                    }

                    val lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime
                    LOGGER.v("Processing time: %d ms", lastProcessingTimeMs)

                    readyForNextImage()
                })
    }

    private val screenOrientationPortrait: Boolean
        get() = when (screenOrientation) {
            Surface.ROTATION_180 -> true
            Surface.ROTATION_0 -> true
            else -> false
        }

    private val screenOrientationCorrectionAngle: Float
        get() = when (screenOrientation) {
            Surface.ROTATION_270 -> 180.0f
            Surface.ROTATION_180 -> -90.0f
            Surface.ROTATION_90 -> 0.0f
            Surface.ROTATION_0 -> 90.0f
            else -> 0.0f
        }

    private fun correctOrientation(bitmap: Bitmap): Bitmap {
        val matrix = Matrix()
        matrix.setRotate(screenOrientationCorrectionAngle)
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, false)
    }

    private fun cropLicensePlate(bitmap: Bitmap, rect: RectF) : Bitmap {
        return Bitmap.createBitmap(bitmap, rect.left.toInt(), rect.top.toInt(), rect.width().toInt(), rect.height().toInt())
    }

    private fun cropROI(bitmap: Bitmap, box: BoundingBox) : Bitmap {
        return Bitmap.createBitmap(bitmap, box.X, box.Y, box.WIDTH, box.HEIGHT,null, false)
    }

    private fun getROI(width: Int, height: Int) : BoundingBox
    {
        val x = 15
        val w = width - (2 * x)
        val h = (width * licenseRecognizer!!.getInputAspectRatio()).roundToInt()
        val y = (height - h) / 2

        return BoundingBox(x, y, w, h)
    }

    companion object {
        private val LOGGER = Logger()
        private const val DETECTION_SCORE_THRESHOLD = 0.0f //0.8f
        private const val cornerSize = 20.0f
    }
}