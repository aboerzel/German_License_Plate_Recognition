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

/**
 *
 * */
class ClassifierActivity : CameraActivity(), ImageReader.OnImageAvailableListener {

    override val layoutId: Int
        get() = R.layout.camera_connection_fragment

    override val desiredPreviewFrameSize: Size?
        get() = Size(640, 480)

    private var plateDetector: PlateDetector? = null
    private var licenseRecognizer: LicenseRecognizer? = null
    private var lastDetection: Detection? = null

    private lateinit var rgbFrameBitmap: Bitmap
    private lateinit var trackingOverlay: OverlayView

    private val trackingBoxPaint = Paint()
    private var trackingTitle: BorderedText? = null
    private var titleTextSizePx: Float = 0.0f

    private var computingDetection = false

    init {
        trackingBoxPaint.color = Color.GREEN
        trackingBoxPaint.alpha = 200
        trackingBoxPaint.style = Paint.Style.STROKE
        trackingBoxPaint.strokeWidth = 6.0f
    }

    /**
     * Is called when the size of the preview changes, e.g. during initialization and when changing the device orientation
     *
     * @size Current preview size
     * @rotation Current device rotation
     * */
    public override fun onPreviewSizeChosen(size: Size, rotation: Int) {

        titleTextSizePx = TypedValue.applyDimension(TypedValue.COMPLEX_UNIT_DIP, 36F, resources.displayMetrics);
        trackingTitle = BorderedText(trackingBoxPaint.color, Color.BLACK, titleTextSizePx)

        if (plateDetector == null) {
            try {
                LOGGER.d("Creating plateDetector")
                plateDetector = PlateDetector(this)
            } catch (e: IOException) {
                LOGGER.e(e, "Failed to create plateDetector: %s", e.message)
                throw e
            }
        }

        if (licenseRecognizer == null) {
            try {
                LOGGER.d("Creating licenseRecognizer")
                licenseRecognizer = LicenseRecognizer(this)
            } catch (e: IOException) {
                LOGGER.e(e, "Failed to create licenseRecognizer: %s", e.message)
                throw e
            }
        }

        previewWidth = size.width
        previewHeight = size.height
        LOGGER.i("Initializing at size %dx%d", previewWidth, previewHeight)
        rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Bitmap.Config.ARGB_8888)

        trackingOverlay = findViewById<View>(R.id.tracking_overlay) as OverlayView
        trackingOverlay.addCallback { canvas -> drawDetection(canvas) }
    }

    /**
     * Draw detection (license text and bounding box of the license plate)
     *
     * @canvas Canvas on which to draw
     * */
    private fun drawDetection(canvas: Canvas) {
        if (lastDetection == null)
            return

        try {
            val location = transformToOverlayLocation(lastDetection!!.getLocation())
            canvas.drawRoundRect(location, 0.0f, 0.0f, trackingBoxPaint)

            trackingTitle!!.drawTextBox(canvas, location.left, location.top, lastDetection!!.getTitle())
        }
        catch (e: Exception) {
            LOGGER.e("Draw detection failed: %s", e.message)
            print(e.message)
        }
    }

    /**
     * Transforms the detected rectangle from preview coordinates to overlay coordinates
     *
     * @rect rectangle in preview coordinates to be transformed
     * @return transformed rectange in overlay coordinates
     * */
    private fun transformToOverlayLocation(rect: RectF) : RectF {
        val scaleX = if (screenOrientationPortrait) (trackingOverlay.width.toFloat() / previewHeight) else (trackingOverlay.width.toFloat() / previewWidth)
        val scaleY = if (screenOrientationPortrait) (trackingOverlay.height.toFloat() / previewWidth) else (trackingOverlay.height.toFloat() / previewHeight)
        return RectF(rect.left * scaleX, rect.top, rect.right * scaleX, rect.bottom * scaleY)
    }

    /**
     * Performs the license plate detection and license number recognition on the current camera image
     * */
    override fun processImage() {

        if (computingDetection) {
            readyForNextImage();
            return;
        }
        computingDetection = true;

        rgbFrameBitmap.setPixels(getRgbBytes(), 0, previewWidth, 0, 0, previewWidth, previewHeight)

        readyForNextImage()

        //rgbFrameBitmap = correctOrientation(rgbFrameBitmap)

        runInBackground(
                Runnable {
                    try {
                        // reset last detection
                        lastDetection = null

                        val startTime = SystemClock.uptimeMillis()

                        // detect license plate
                        val detections = plateDetector!!.detectPlates(rgbFrameBitmap)
                        LOGGER.i("Detected license plates: %d", detections.count())

                        if (detections.count() > 0 && detections[0].confidence >= DETECTION_SCORE_THRESHOLD && isValidRect(detections[0].getLocation())) {
                            LOGGER.i("Detected license plate: %s", detections[0].toString())
                            lastDetection = detections[0]

                            // crop license plate from image and recognize the license text
                            val detectedPlateBmp = cropLicensePlate(rgbFrameBitmap, lastDetection!!.getLocation())
                            lastDetection!!.setTitle(licenseRecognizer!!.recognize(detectedPlateBmp))
                            LOGGER.i("Recognized license: %s", lastDetection!!.getTitle())
                        }

                        LOGGER.i("Processing time: %d ms", SystemClock.uptimeMillis() - startTime)
                    } catch (e: Exception) {
                        LOGGER.e("Detection failed: %s", e.message)
                    }

                    // redraw overlay
                    trackingOverlay.postInvalidate()

                    computingDetection = false;
                    //readyForNextImage()
                })
    }

    /**
     * Checks if it is a valid rectangle
     *
     * @rect Rectangle to be checked
     * @return true, if the rectangle is valid, otherwise false
     * */
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

    /**
     * Returns the current device orientation
     */
    private val screenOrientationPortrait: Boolean
        get() = when (screenOrientation) {
            Surface.ROTATION_180 -> true
            Surface.ROTATION_0 -> true
            else -> false
        }

    /**
     * Returns the current correction angle with respect to the device orientation
     * */
    private val screenOrientationCorrectionAngle: Float
        get() = when (screenOrientation) {
            Surface.ROTATION_270 -> 180.0f
            Surface.ROTATION_180 -> -90.0f
            Surface.ROTATION_90 -> 0.0f
            Surface.ROTATION_0 -> 90.0f
            else -> 0.0f
        }

    /**
     * Rotates the current image by the correction angle to correct the device orientation
     *
     * @bitmap Bitmap to be corrected
     * @return corrected bitmap
     * */
    private fun correctOrientation(bitmap: Bitmap): Bitmap {
        val matrix = Matrix()
        matrix.setRotate(screenOrientationCorrectionAngle)
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, false)
    }

    /**
     * Crops a section of the image
     *
     * @image Original image
     * @rect rectangle to be cropped
     * @return Cropped image
     * */
    private fun cropLicensePlate(bitmap: Bitmap, rect: RectF) : Bitmap {
        return Bitmap.createBitmap(bitmap, rect.left.toInt(), rect.top.toInt(), rect.width().toInt(), rect.height().toInt())
    }

    companion object {
        private val LOGGER = Logger()
        private const val DETECTION_SCORE_THRESHOLD = 0.8f
    }
}