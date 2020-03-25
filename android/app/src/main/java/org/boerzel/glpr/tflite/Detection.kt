package org.boerzel.glpr.tflite

import android.graphics.RectF

/** An license detection result.  */
class Detection(
        /**
         * A unique identifier for what has been recognized.
         * Specific to the instance, not the class of the object.
         */
        private val id: String,

        /** Display name for the recognition.  */
        private var title: String,

        /**
         * A sortable score for how good the recognition is relative to others. Higher should be better.
         */
        val confidence: Float,

        /** Optional location within the source image for the location of the recognized object.  */
        private var location: RectF) {

    fun getLocation(): RectF {
        return RectF(location)
    }

    fun setLocation(location: RectF) {
        this.location = location
    }

    fun setTitle(title: String) {
        this.title = title
    }

    fun getTitle() : String {
        return this.title
    }

    override fun toString(): String {
        var resultString = ""
        resultString += "[$id] "
        resultString += "$title "
        resultString += String.format("(%.1f%%) ", confidence * 100.0f)
        resultString += location.toString() + " "
        return resultString.trim { it <= ' ' }
    }
}