/*
 * Copyright 2014 Google Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.google.android.apps.muzei.render

import android.graphics.Bitmap
import android.graphics.Color
import androidx.core.graphics.ColorUtils

/**
 * Mean WCAG relative luminance (0 = black .. 1 = white) of the bitmap. Unlike [darkness], each
 * pixel is gamma-corrected via [ColorUtils.calculateLuminance] — the same metric the framework's
 * WallpaperColors uses for its dark-text decision — so values are perceptually accurate.
 */
fun Bitmap.relativeLuminance(): Float {
    if (width == 0 || height == 0) {
        return 0f
    }
    val pixels = IntArray(width * height)
    getPixels(pixels, 0, width, 0, 0, width, height)
    var total = 0.0
    for (pixel in pixels) {
        total += ColorUtils.calculateLuminance(pixel)
    }
    return (total / pixels.size).toFloat()
}

fun Bitmap?.darkness(): Float {
    if (this == null || width == 0 || height == 0) {
        return 0f
    }

    // Read a row at a time with the bulk getPixels() rather than a per-pixel getPixel() (a JNI
    // call each), so this stays cheap even on larger bitmaps.
    val row = IntArray(width)
    var totalLum = 0L
    var y = 0
    var color: Int
    while (y < height) {
        getPixels(row, 0, width, 0, y, width, 1)
        var x = 0
        while (x < width) {
            color = row[x]
            totalLum += (0.21f * Color.red(color)
                    + 0.71f * Color.green(color)
                    + 0.07f * Color.blue(color)).toInt()
            x++
        }
        y++
    }

    return totalLum.toFloat() / (width * height) / 256f
}

fun Int.sampleSize(targetSize: Int): Int {
    var sampleSize = 1
    while (this / (sampleSize shl 1) > targetSize) {
        sampleSize = sampleSize shl 1
    }
    return sampleSize
}
