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

package org.boerzel.glpr.utils;

import android.graphics.Canvas;
import android.graphics.Paint;
import android.graphics.Paint.Style;
import android.graphics.Rect;
import android.graphics.Typeface;

/** A class that encapsulates the tedious bits of rendering legible, bordered text onto a canvas. */
public class BorderedText {
  private final Paint backgroundPaint;
  private final Paint textPaint;

  private int margin = 15;

  /**
   * Create a bordered text object with the specified interior and exterior colors, text size and
   * alignment.
   *
   * @param backgroundColor the backgroundColor color
   * @param textColor       the text color
   * @param textSize        text size in pixels
   */
  public BorderedText(final int backgroundColor, final int textColor, final float textSize) {

    backgroundPaint = new Paint();
    backgroundPaint.setTextSize(textSize);
    backgroundPaint.setColor(backgroundColor);
    backgroundPaint.setStyle(Style.FILL);
    backgroundPaint.setAntiAlias(false);
    backgroundPaint.setAlpha(255);

    textPaint = new Paint();
    textPaint.setTextSize(textSize);
    textPaint.setColor(textColor);
    textPaint.setStyle(Style.FILL);
    textPaint.setAntiAlias(false);
    textPaint.setAlpha(255);

    Typeface currentTypeFace = textPaint.getTypeface();
    Typeface bold = Typeface.create(currentTypeFace, Typeface.BOLD);
    textPaint.setTypeface(bold);
  }

  public void drawTextBox(final Canvas canvas, final float posX, final float posY, final String text) {

    Rect bounds = new Rect();
    textPaint.getTextBounds(text, 0, text.length(), bounds);

    canvas.drawRect(posX, posY - bounds.height() - (2 * margin),
            posX + bounds.width() + (2 * margin),
            posY, backgroundPaint);

    canvas.drawText(text, posX + margin, posY - margin, textPaint);
  }
}