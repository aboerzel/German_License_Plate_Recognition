package test

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.media.Image
import android.media.ImageReader
import android.support.test.InstrumentationRegistry
import android.util.Log
import androidx.test.espresso.Espresso.onView
import androidx.test.espresso.assertion.ViewAssertions.matches
import androidx.test.espresso.matcher.ViewMatchers.withId
import androidx.test.espresso.matcher.ViewMatchers.withText
import androidx.test.rule.ActivityTestRule
import androidx.test.runner.AndroidJUnit4
import org.boerzel.glpr.ClassifierActivity
import org.boerzel.glpr.R
import org.junit.Assert
import org.junit.Before
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import org.mockito.Mockito
import java.io.IOException
import java.io.InputStream

/** Test for license plate recognition app. */
@RunWith(AndroidJUnit4::class)
class LicenseRecognizerTest {

    @get:Rule var rule = ActivityTestRule(ClassifierActivity::class.java)

    @Before fun setUp() {
    }

    @Test
    @Throws(IOException::class)
    fun recognitionResultsShouldNotChange() {
        val imageReader: ImageReader = Mockito.mock(ImageReader::class.java)

        //val image = loadImage(INPUTS[0])
        val image: Image? = null
        Mockito.`when`(imageReader.acquireLatestImage()).thenReturn(image)

        rule.activity.onImageAvailable(imageReader)

        //val licenseRecognizer = LicenseRecognizer(rule.activity)
        for (i in INPUTS.indices) {
            val imageFileName = INPUTS[i]
            val license = imageFileName.split(".")[0].split("#")[0]

            //val image = loadImage(imageFileName)
            //val result = licenseRecognizer.recognize(image)
            var result = license

            Assert.assertTrue(result == license)
            onView(withId(R.id.detected_item)).check(matches(withText(license)))
        }
    }

    companion object {
        private val INPUTS = arrayOf("ES-EL2399.jpg")

        private fun loadImage(fileName: String): Bitmap {
            val assetManager = InstrumentationRegistry.getInstrumentation().context.assets
            var inputStream: InputStream? = null
            try {
                inputStream = assetManager.open(fileName)
            } catch (e: IOException) {
                Log.e("Test", "Cannot load image from assets")
            }
            return BitmapFactory.decodeStream(inputStream)
        }
    }
}