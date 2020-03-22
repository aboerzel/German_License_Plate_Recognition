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
import android.util.TypedValue
import android.view.Surface
import android.view.View
import org.boerzel.glpr.customview.OverlayView
import org.boerzel.glpr.env.BorderedText
import org.boerzel.glpr.env.Logger
import org.boerzel.glpr.tflite.Detection
import org.boerzel.glpr.tflite.LicenseRecognizer
import org.boerzel.glpr.tflite.PlateDetector
import java.io.IOException

class ClassifierActivity : CameraActivity(), ImageReader.OnImageAvailableListener {

    override val layoutId: Int
        get() = R.layout.camera_connection_fragment

    override val desiredPreviewFrameSize: Size?
        get() = Size(640, 480)

    private var plateDetector: PlateDetector? = null
    private var licenseRecognizer: LicenseRecognizer? = null
    private var detection: Detection? = null
    private var license: String? = null

    private lateinit var rgbFrameBitmap: Bitmap
    private lateinit var trackingOverlay: OverlayView

    private val trackingBoxPaint = Paint()
    private var trackingTitle: BorderedText? = null
    private var titleTextSizePx: Float = 0.0f

    init {
        trackingBoxPaint.color = Color.GREEN
        trackingBoxPaint.alpha = 200
        trackingBoxPaint.style = Paint.Style.STROKE
        trackingBoxPaint.strokeWidth = 6.0f
    }

    public override fun onPreviewSizeChosen(size: Size, rotation: Int) {

        titleTextSizePx = TypedValue.applyDimension(TypedValue.COMPLEX_UNIT_DIP, 20F, resources.displayMetrics);
        trackingTitle = BorderedText(trackingBoxPaint.color, Color.WHITE, titleTextSizePx)

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
        trackingOverlay.addCallback { canvas -> drawTrackingBox(canvas) }
    }

    private fun drawTrackingBox(canvas: Canvas) {
        if (detection == null)
            return

        val location = transformToTrackingOverlayLocation(detection!!.getLocation())
        canvas.drawRoundRect(transformToTrackingOverlayLocation(location), 0.0f, 0.0f, trackingBoxPaint)

        val title = if (detection!!.title.isNullOrEmpty())
            String.format("%.2f", 100 * detection!!.confidence!!)
        else
            String.format("%s %.2f", detection!!.title, 100 * detection!!.confidence!!)

        trackingTitle!!.drawText(canvas, location.left, location.top, "$license%")
    }

    private fun transformToTrackingOverlayLocation(roi: RectF) : RectF {
        //val scale = if (screenOrientationPortrait)
        //    trackingOverlay.width.toFloat() / previewHeight
        //else
        val scaleX = trackingOverlay.width.toFloat() / previewWidth
        val scaleY = trackingOverlay.height.toFloat() / previewHeight
        return RectF(roi.left * scaleX, roi.top * scaleY, roi.right * scaleX, roi.bottom * scaleY)
    }

    override fun processImage() {
        rgbFrameBitmap.setPixels(getRgbBytes(), 0, previewWidth, 0, 0, previewWidth, previewHeight)

        runInBackground(
                Runnable {
                    val startTime = SystemClock.uptimeMillis()

                    val detections = plateDetector!!.detect_plates(rgbFrameBitmap)
                    LOGGER.v("Detected license plates: %d", detections.count())

                    if (detections.count() > 0 && detections[0].confidence!! >= DETECTION_SCORE_THRESHOLD) {
                        LOGGER.v("Detected license plate: %s", detections[0].toString())
                        detection = detections[0]
                        val detectedPlateBmp = cropLicensePlate(rgbFrameBitmap, detection!!.getLocation())
                        license = licenseRecognizer!!.recognize(detectedPlateBmp)
                        LOGGER.v("Recognized license: %s", license)
                    }
                    else
                    {
                        detection = null
                        license = null
                    }

                    val lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime
                    LOGGER.v("Processing time: %d ms", lastProcessingTimeMs)

                    trackingOverlay.postInvalidate()

                    runOnUiThread { showResult(license?:"") }
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

    companion object {
        private val LOGGER = Logger()
        private const val DETECTION_SCORE_THRESHOLD = 0.8f
    }
}