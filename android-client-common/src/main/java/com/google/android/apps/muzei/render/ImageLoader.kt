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

import android.content.ContentResolver
import android.content.res.AssetManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.media.MediaMetadataRetriever
import android.net.Uri
import android.os.Build
import android.util.Log
import androidx.exifinterface.media.ExifInterface
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import net.nurik.roman.muzei.androidclientcommon.BuildConfig
import java.io.FileNotFoundException
import java.io.IOException
import java.io.InputStream
import kotlin.math.max

fun InputStream.isValidImage(): Boolean {
    val options = BitmapFactory.Options().apply {
        inJustDecodeBounds = true
        inPreferredConfig = Bitmap.Config.ARGB_8888
    }
    BitmapFactory.decodeStream(this, null, options)
    return with(options) {
        outWidth != 0 && outHeight != 0 &&
                (Build.VERSION.SDK_INT < Build.VERSION_CODES.O ||
                        outConfig == Bitmap.Config.ARGB_8888)
    }
}

/**
 * Probes [uri] with [MediaMetadataRetriever] to determine whether it resolves to a playable
 * video (a container with a video track). Used to accept video artwork that would otherwise fail
 * the still-image validation in [isValidImage].
 *
 * Returns the resolved MIME type (e.g. `video/mp4`) when it is a video, or `null` when it is not
 * a video (or can't be read). The provider [uri]s Muzei validates return a cursor MIME from
 * [ContentResolver.getType], not the media type, so this reads the container itself rather than
 * trusting the reported type.
 */
fun ContentResolver.videoMimeType(uri: Uri): String? = try {
    openAssetFileDescriptor(uri, "r")?.use { afd ->
        val retriever = MediaMetadataRetriever()
        try {
            if (afd.declaredLength >= 0) {
                retriever.setDataSource(afd.fileDescriptor, afd.startOffset, afd.declaredLength)
            } else {
                retriever.setDataSource(afd.fileDescriptor)
            }
            if (retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_HAS_VIDEO) == "yes") {
                retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_MIMETYPE) ?: "video/*"
            } else {
                null
            }
        } finally {
            retriever.release()
        }
    }
} catch (_: Exception) {
    // Corrupt/unreadable media, an unsupported codec, or an unresolvable URI: treat as "not a
    // video" so the caller falls through to marking the artwork invalid.
    null
}

/**
 * Extracts a still poster frame from the video at [uri], scaled to roughly [width] x [height] when
 * both are positive (and the platform supports scaled extraction). Used to give video artwork a
 * thumbnail on surfaces that can't play it (DocumentsUI, widgets, notifications). Returns null if
 * the URI isn't a readable video.
 */
fun ContentResolver.videoFrame(uri: Uri, width: Int = 0, height: Int = 0): Bitmap? {
    val retriever = MediaMetadataRetriever()
    return try {
        val opened = openAssetFileDescriptor(uri, "r")?.use { afd ->
            if (afd.declaredLength >= 0) {
                retriever.setDataSource(afd.fileDescriptor, afd.startOffset, afd.declaredLength)
            } else {
                retriever.setDataSource(afd.fileDescriptor)
            }
            true
        } ?: false
        if (!opened) {
            null
        } else if (width > 0 && height > 0 &&
                Build.VERSION.SDK_INT >= Build.VERSION_CODES.O_MR1) {
            retriever.getScaledFrameAtTime(-1, MediaMetadataRetriever.OPTION_CLOSEST_SYNC,
                    width, height)
        } else {
            retriever.frameAtTime
        }
    } catch (_: Exception) {
        null
    } finally {
        retriever.release()
    }
}

/**
 * Base class for loading images with the correct rotation
 */
sealed class ImageLoader {

    companion object {
        private const val TAG = "ImageLoader"
        private const val UNKNOWN_ROTATION = -1

        suspend fun decode(
                contentResolver: ContentResolver,
                uri: Uri,
                targetWidth: Int = 0,
                targetHeight: Int = targetWidth
        ) = withContext(Dispatchers.IO) {
            ContentUriImageLoader(contentResolver, uri)
                    .decode(targetWidth, targetHeight)
        }
    }

    // Rotation and original (pre-rotation) bounds are immutable for a given source, so compute
    // each once and reuse it across the several getSize()/decode() calls made per artwork load
    // rather than re-opening the stream (an IPC + file read for content URIs) every time.
    private var cachedRotation: Int = UNKNOWN_ROTATION
    private var cachedOriginalBounds: Pair<Int, Int>? = null

    /** Original (pre-rotation) pixel dimensions, decoded once and cached. (0, 0) on failure. */
    private fun originalBounds(): Pair<Int, Int> {
        cachedOriginalBounds?.let { return it }
        val bounds = openInputStream()?.use { input ->
            val options = BitmapFactory.Options().apply {
                inJustDecodeBounds = true
            }
            BitmapFactory.decodeStream(input, null, options)
            options.outWidth to options.outHeight
        } ?: (0 to 0)
        return bounds.also { cachedOriginalBounds = it }
    }

    fun getSize(): Pair<Int, Int> {
        return try {
            val (originalWidth, originalHeight) = originalBounds()
            if (originalWidth == 0 || originalHeight == 0) return 0 to 0
            val rotation = getRotation()
            val width = if (rotation == 90 || rotation == 270) originalHeight else originalWidth
            val height = if (rotation == 90 || rotation == 270) originalWidth else originalHeight
            return width to height
        } catch (e: Exception) {
            if (BuildConfig.DEBUG) {
                Log.w(TAG, "Error decoding ${toString()}: ${e.message}")
            }
            0 to 0
        }
    }

    fun decode(
            targetWidth: Int = 0,
            targetHeight: Int = targetWidth
    ) : Bitmap? {
        return try {
            val (originalWidth, originalHeight) = originalBounds()
            if (originalWidth == 0 || originalHeight == 0) return null
            val rotation = getRotation()
            val width = if (rotation == 90 || rotation == 270) originalHeight else originalWidth
            val height = if (rotation == 90 || rotation == 270) originalWidth else originalHeight
            openInputStream()?.use { input ->
                BitmapFactory.decodeStream(input, null,
                        BitmapFactory.Options().apply {
                            inPreferredConfig = Bitmap.Config.ARGB_8888
                            if (targetWidth != 0) {
                                inSampleSize = max(
                                        width.sampleSize(targetWidth),
                                        height.sampleSize(targetHeight))
                            }
                        })
            }?.run {
                when (rotation) {
                    0 -> this
                    else -> {
                        // Post-rotation displayed dimensions of the decoded bitmap.
                        val rotated90 = rotation == 90 || rotation == 270
                        val displayWidth = if (rotated90) this.height else this.width
                        val displayHeight = if (rotated90) this.width else this.height
                        // inSampleSize is power-of-two coarse, so the decode lands at or above the
                        // target. Fold a downscale-to-target into the rotation matrix so we produce
                        // a target-sized rotated bitmap in a single pass instead of allocating and
                        // rotating the full-resolution one. It's a downscale (never an upscale), so
                        // there's no quality loss, and the rotation cost drops from being
                        // proportional to the full image to being proportional to the (small) output.
                        val scale = if (targetWidth != 0 && targetHeight != 0) {
                            max(targetWidth.toFloat() / displayWidth,
                                    targetHeight.toFloat() / displayHeight).coerceAtMost(1f)
                        } else {
                            1f
                        }
                        val matrix = Matrix().apply {
                            if (scale < 1f) {
                                postScale(scale, scale)
                            }
                            postRotate(rotation.toFloat())
                        }
                        Bitmap.createBitmap(
                                this, 0, 0,
                                this.width, this.height,
                                matrix, true).also { transformed ->
                            if (transformed != this) {
                                recycle()
                            }
                        }
                    }
                }
            }
        } catch (e: Exception) {
            if (BuildConfig.DEBUG) {
                Log.w(TAG, "Error decoding ${toString()}: ${e.message}")
            }
            null
        }
    }

    fun getRotation(): Int {
        cachedRotation.let { if (it != UNKNOWN_ROTATION) return it }
        return computeRotation().also { cachedRotation = it }
    }

    private fun computeRotation(): Int = try {
        openInputStream()?.use { input ->
            val exifInterface = ExifInterface(input)
            when (exifInterface.getAttributeInt(ExifInterface.TAG_ORIENTATION,
                    ExifInterface.ORIENTATION_NORMAL)) {
                ExifInterface.ORIENTATION_ROTATE_90 -> 90
                ExifInterface.ORIENTATION_ROTATE_180 -> 180
                ExifInterface.ORIENTATION_ROTATE_270 -> 270
                else -> 0
            }
        } ?: 0
    } catch (e: Exception) {
        if (BuildConfig.DEBUG) {
            Log.w(TAG, "Couldn't open EXIF interface for ${toString()}", e)
        }
        0
    }

    abstract fun openInputStream() : InputStream?
}

/**
 * An [ImageLoader] capable of loading images from a [ContentResolver]
 */
class ContentUriImageLoader(
        private val contentResolver: ContentResolver,
        private val uri: Uri
) : ImageLoader() {

    @Throws(FileNotFoundException::class)
    override fun openInputStream(): InputStream? =
            contentResolver.openInputStream(uri)

    override fun toString(): String {
        return uri.toString()
    }
}

/**
 * An [ImageLoader] capable of loading images from [AssetManager]
 */
class AssetImageLoader(
        private val assetManager: AssetManager,
        private val fileName: String
) : ImageLoader() {

    @Throws(IOException::class)
    override fun openInputStream(): InputStream =
            assetManager.open(fileName)

    override fun toString(): String {
        return fileName
    }
}