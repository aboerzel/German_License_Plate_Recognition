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
import org.boerzel.glpr.tflite.Detection
import org.boerzel.glpr.tflite.LicenseRecognizer
import org.boerzel.glpr.tflite.PlateDetector
import org.boerzel.glpr.utils.BorderedText
import org.boerzel.glpr.utils.Logger
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

        titleTextSizePx = TypedValue.applyDimension(TypedValue.COMPLEX_UNIT_DIP, 36F, resources.displayMetrics);
        trackingTitle = BorderedText(trackingBoxPaint.color, Color.BLACK, titleTextSizePx)

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

        try {
            val location = transformToOverlayLocation(detection!!.getLocation())
            canvas.drawRoundRect(location, 0.0f, 0.0f, trackingBoxPaint)

            trackingTitle!!.drawTextBox(canvas, location.left, location.top, "$license")
        }
        catch (e: Exception) {
            print(e.message)
        }
    }

    private fun transformToOverlayLocation(rect: RectF) : RectF {
        //val scale = if (screenOrientationPortrait)
        //    trackingOverlay.width.toFloat() / previewHeight
        //else
        //    trackingOverlay.width.toFloat() / previewWidth

        val scaleX = trackingOverlay.width.toFloat() / previewWidth
        val scaleY = (trackingOverlay.height.toFloat() / previewHeight)
        return RectF(rect.left * scaleX, rect.top, rect.right * scaleX, rect.bottom * scaleY)
    }

    override fun processImage() {
        rgbFrameBitmap.setPixels(getRgbBytes(), 0, previewWidth, 0, 0, previewWidth, previewHeight)

        runInBackground(
                Runnable {
                    try {
                        val startTime = SystemClock.uptimeMillis()

                        val detections = plateDetector!!.detect_plates(rgbFrameBitmap)
                        LOGGER.v("Detected license plates: %d", detections.count())

                        if (detections.count() > 0 && detections[0].confidence!! >= DETECTION_SCORE_THRESHOLD && isValidRect(detections[0].getLocation())) {
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
                    }
                    catch (e: Exception) {
                        print(e.message)
                        detection = null
                        license = null
                    }

                    trackingOverlay.postInvalidate()

                    readyForNextImage()
                })
    }

    private fun isValidRect(rect: RectF) : Boolean {
        if (rect.left < 0)
            return false
        if (rect.top < 0)
            return false
        if (rect.right < 0)
            return false
        if (rect.bottom < 0)
            return false
        if (rect.right < rect.left)
            return false
        if (rect.bottom < rect.top)
            return false

        return true
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