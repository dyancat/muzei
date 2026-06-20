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

import android.app.ActivityManager
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.RectF
import android.opengl.GLES20
import android.opengl.GLSurfaceView
import android.opengl.Matrix
import android.util.Log
import android.view.animation.AccelerateDecelerateInterpolator
import androidx.annotation.Keep
import androidx.core.graphics.scale
import com.google.android.apps.muzei.ArtDetailOpen
import com.google.android.apps.muzei.ArtDetailViewport
import com.google.android.apps.muzei.settings.Prefs
import com.google.android.apps.muzei.util.TickingFloatAnimator
import com.google.android.apps.muzei.util.constrain
import com.google.android.apps.muzei.util.floorEven
import com.google.android.apps.muzei.util.interpolate
import com.google.android.apps.muzei.util.roundMult4
import com.google.android.apps.muzei.util.uninterpolate
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.launch
import javax.microedition.khronos.egl.EGLConfig
import javax.microedition.khronos.opengles.GL10
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

sealed class SwitchingPhotos(val viewportId: Int)
data class SwitchingPhotosInProgress(private val currentId: Int) : SwitchingPhotos(currentId)
data class SwitchingPhotosDone(private val currentId: Int) : SwitchingPhotos(currentId)

val SwitchingPhotosStateFlow = MutableStateFlow<SwitchingPhotos?>(null)

data class ArtworkSize(val width: Int, val height: Int)

val ArtworkSizeStateFlow = MutableStateFlow<ArtworkSize?>(null)

class MuzeiBlurRenderer(
        private val context: Context,
        private val callbacks: Callbacks,
        private val demoMode: Boolean = false,
        private val preview: Boolean = false
) : GLSurfaceView.Renderer {

    companion object {
        private const val TAG = "MuzeiBlurRenderer"

        private const val CROSSFADE_ANIMATION_DURATION = 500
        private const val BLUR_ANIMATION_DURATION = 500

        const val DEFAULT_BLUR = 250 // max 500
        const val DEFAULT_GREY = 0 // max 500
        const val DEFAULT_MAX_DIM = 128 // technical max 255
        private const val DEMO_BLUR = 250
        private const val DEMO_DIM = 64
        private const val DEMO_GREY = 0
        private const val DIM_RANGE = 0.5f // percent of max dim

        // Technical maximum of the blur preference (see DEFAULT_BLUR). Used to size the downscaled
        // blur source for the strongest possible blur, so the source resolution stays fixed as the
        // runtime blur amount changes.
        private const val MAX_BLUR_AMOUNT = 500

        // Per-frame easing fraction for home<->lock (and settings) effect transitions.
        private const val PARAM_EASE_FACTOR = 0.15f

        // Prescaled blur radius (in source texels) over which the blurred overlay fades in. Below
        // this the full-res sharp picture still shows through, so a tiny radius doesn't abruptly
        // swap to the downscaled blur source; above it the overlay fully covers (blur hides the
        // downscaling).
        private const val BLUR_FADE_IN_PIXELS = 4f

        // The GPU blur source is a single (non-tiled) texture, so cap its dimensions to stay within
        // the guaranteed GL_MAX_TEXTURE_SIZE.
        private const val MAX_BLUR_SOURCE_DIM = 2048
    }

    private val blurKeyframes: Int
    // Blur-source downscale, read from the background decode thread (see decode()), so volatile.
    // Sized for MAX_BLUR_AMOUNT so it doesn't depend on the current blur amount.
    @Volatile private var blurredSampleSize: Int = 0
    // Effect-strength targets, set by recompute*() on the main thread and read on the GL thread.
    @Volatile private var targetPrescaledBlurPixels: Int = 0
    @Volatile private var targetDim: Int = 0
    @Volatile private var targetGrey: Int = 0
    // Effective effect strengths, eased toward the targets each frame (GL thread only) so a
    // home<->lock or settings change animates smoothly instead of needing a re-decode + crossfade.
    private var maxPrescaledBlurPixels = 0f
    private var maxDim = 0f
    private var maxGrey = 0f

    // Model and view matrices. Projection and MVP stored in picture set
    private val modelMatrix = FloatArray(16)
    private val viewMatrix = FloatArray(16)

    private var aspectRatio: Float = 0f
    @Volatile private var currentHeight: Int = 0

    private var currentGLPictureSet: GLPictureSet
    private var nextGLPictureSet: GLPictureSet
    private lateinit var colorOverlay: GLColorOverlay

    private var queuedNextImageLoader: ImageLoader? = null
    // The next artwork is decoded and blurred off the GL thread so rendering (e.g. scrolling)
    // stays smooth during a switch; only the texture upload runs on the GL thread. loadInProgress
    // serialises with the crossfade, and loadGeneration discards a decode that's been superseded.
    private val decodeScope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
    private var loadInProgress = false
    private var loadGeneration = 0

    private var surfaceCreated: Boolean = false

    @Volatile
    private var normalOffsetX: Float = 0f
    @Volatile
    private var zoomAmount: Float = 1f
    private val currentViewport = RectF() // [-1, -1] to [1, 1], flipped

    var isBlurred = true
        private set
    private var blurPreferenceName = Prefs.PREF_BLUR_AMOUNT
    private var dimPreferenceName = Prefs.PREF_DIM_AMOUNT
    private var greyPreferenceName = Prefs.PREF_GREY_AMOUNT
    private var blurRelatedToArtDetailMode = false
    private val blurInterpolator = AccelerateDecelerateInterpolator()
    private val blurAnimator = TickingFloatAnimator(BLUR_ANIMATION_DURATION * if (demoMode) 5 else 1)
    private val crossfadeAnimator = TickingFloatAnimator(CROSSFADE_ANIMATION_DURATION)

    init {
        val activityManager = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
        blurKeyframes = if (activityManager.isLowRamDevice) 1 else 2
        blurAnimator.currentValue = blurKeyframes.toFloat()

        currentGLPictureSet = GLPictureSet(0)
        nextGLPictureSet = GLPictureSet(1) // for transitioning to next pictures
        setNormalOffsetX(0f)
        setZoom(1f)
        recomputeMaxPrescaledBlurPixels()
        recomputeMaxDimAmount()
        recomputeGreyAmount()
        // Start at the targets rather than easing up from zero on first frame.
        snapEffectParams()
    }

    fun recomputeMaxPrescaledBlurPixels(
            newBlurPreferenceName: String = blurPreferenceName
    ) {
        blurPreferenceName = newBlurPreferenceName
        // Compute blur sizes
        val blurAmount = if (demoMode)
            DEMO_BLUR
        else
            Prefs.getSharedPreferences(context)
                    .getInt(blurPreferenceName, DEFAULT_BLUR)
        val dm = context.resources.displayMetrics
        // Size the downscaled blur source for the strongest possible blur, so its resolution is
        // independent of the current blur amount — only the runtime radius below changes. That lets
        // home<->lock and settings changes animate without re-decoding the source.
        val maxPossibleBlurPx = (dm.heightPixels * (MAX_BLUR_AMOUNT * 0.0001f)).toInt()
        blurredSampleSize = 4
        while (maxPossibleBlurPx / blurredSampleSize > GLBlur.MAX_RADIUS) {
            blurredSampleSize = blurredSampleSize shl 1
        }
        val maxBlurPx = (dm.heightPixels * (blurAmount * 0.0001f)).toInt()
        targetPrescaledBlurPixels = maxBlurPx / blurredSampleSize
    }

    fun recomputeMaxDimAmount(
            newDimPreferenceName: String = dimPreferenceName
    ) {
        dimPreferenceName = newDimPreferenceName
        targetDim = Prefs.getSharedPreferences(context).getInt(
                dimPreferenceName, DEFAULT_MAX_DIM)
    }

    fun recomputeGreyAmount(
            newGreyPreferenceName: String = greyPreferenceName
    ) {
        greyPreferenceName = newGreyPreferenceName
        targetGrey = if (demoMode)
            DEMO_GREY
        else
            Prefs.getSharedPreferences(context)
                    .getInt(greyPreferenceName, DEFAULT_GREY)
    }

    /** Snaps the effective effect strengths to their targets, skipping the transition animation. */
    private fun snapEffectParams() {
        maxPrescaledBlurPixels = targetPrescaledBlurPixels.toFloat()
        maxDim = targetDim.toFloat()
        maxGrey = targetGrey.toFloat()
    }

    /** Eases the effective effect strengths toward their targets. Returns true while still moving. */
    private fun easeEffectParams(): Boolean {
        maxPrescaledBlurPixels = ease(maxPrescaledBlurPixels, targetPrescaledBlurPixels.toFloat())
        maxDim = ease(maxDim, targetDim.toFloat())
        maxGrey = ease(maxGrey, targetGrey.toFloat())
        return maxPrescaledBlurPixels != targetPrescaledBlurPixels.toFloat() ||
                maxDim != targetDim.toFloat() ||
                maxGrey != targetGrey.toFloat()
    }

    private fun ease(current: Float, target: Float): Float {
        if (current == target) {
            return target
        }
        val next = current + (target - current) * PARAM_EASE_FACTOR
        return if (abs(next - target) < 1f) target else next
    }

    /** Dim alpha (0..255 scale) for an artwork of the given [darkness], at the current dim setting. */
    private fun dimAmountFor(darkness: Float): Float =
        if (demoMode)
            DEMO_DIM.toFloat()
        else
            maxDim * (1 - DIM_RANGE + DIM_RANGE * sqrt(darkness.toDouble()).toFloat())

    override fun onSurfaceCreated(unused: GL10, config: EGLConfig) {
        surfaceCreated = false
        GLES20.glEnable(GLES20.GL_BLEND)
        //        GLES20.glBlendFunc(GLES20.GL_SRC_ALPHA, GLES20.GL_ONE_MINUS_SRC_ALPHA);
        GLES20.glBlendFuncSeparate(GLES20.GL_SRC_ALPHA, GLES20.GL_ONE_MINUS_SRC_ALPHA,
                GLES20.GL_ONE, GLES20.GL_ONE)
        GLES20.glClearColor(0f, 0f, 0f, 0f)

        // Set the camera position (View matrix)
        Matrix.setLookAtM(viewMatrix, 0,
                0f, 0f, 1f,
                0f, 0f, -1f,
                0f, 1f, 0f)

        GLColorOverlay.initGl()
        GLPicture.initGl()
        GLBlur.initGl()

        colorOverlay = GLColorOverlay()

        surfaceCreated = true
        val loader = queuedNextImageLoader
        if (loader != null) {
            queuedNextImageLoader = null
            setAndConsumeImageLoader(loader)
        }
    }

    override fun onSurfaceChanged(unused: GL10, width: Int, height: Int) {
        GLES20.glViewport(0, 0, width, height)
        GLBlur.setScreenSize(width, height)
        hintViewportSize(width, height)
        if (!demoMode && !preview) {
            // Reset art detail viewports
            ArtDetailViewport.setViewport(0, 0f, 0f, 0f, 0f)
            ArtDetailViewport.setViewport(1, 0f, 0f, 0f, 0f)
        }
        currentGLPictureSet.recomputeTransformMatrices()
        nextGLPictureSet.recomputeTransformMatrices()
        recomputeMaxPrescaledBlurPixels()
        // A size change can change the target blur radius (it scales with screen height); apply it
        // immediately rather than easing from the pre-resize value over the next few frames.
        snapEffectParams()
    }

    fun hintViewportSize(width: Int, height: Int) {
        currentHeight = height
        aspectRatio = width * 1f / height
    }

    override fun onDrawFrame(unused: GL10) {
        GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT)

        Matrix.setIdentityM(modelMatrix, 0)

        val (stillCrossFadeAnimating, onCrossFadeEnd) = crossfadeAnimator.tick()
        val (stillBlurAnimating, onBlurEnd) = blurAnimator.tick()
        val stillParamsAnimating = easeEffectParams()
        val stillAnimating = stillCrossFadeAnimating or stillBlurAnimating or stillParamsAnimating

        if (blurRelatedToArtDetailMode) {
            currentGLPictureSet.recomputeTransformMatrices()
            nextGLPictureSet.recomputeTransformMatrices()
        }

        var dimAmount = dimAmountFor(currentGLPictureSet.darkness)
        currentGLPictureSet.drawFrame(1f)
        if (crossfadeAnimator.isRunning || onCrossFadeEnd != null) {
            dimAmount = interpolate(dimAmount, dimAmountFor(nextGLPictureSet.darkness),
                    crossfadeAnimator.currentValue)
            nextGLPictureSet.drawFrame(crossfadeAnimator.currentValue)
        }

        colorOverlay.color = Color.argb(
                (dimAmount * blurAnimator.currentValue / blurKeyframes).toInt().coerceIn(0, 255),
                0, 0, 0)
        // Skip the full-screen overlay draw when it would contribute nothing (focused, or dim off).
        if (Color.alpha(colorOverlay.color) != 0) {
            colorOverlay.draw(modelMatrix) // don't need any perspective or anything for color overlay
        }

        if (stillAnimating) {
            callbacks.requestRender()
        }

        // Run any animator end-callbacks only after the final frame has been drawn.
        // crossfadeAnimator's onEnd swaps the current/next picture sets, so invoking it
        // before drawing would render the swapped-in artwork a frame too early and flicker.
        onCrossFadeEnd?.invoke()
        onBlurEnd?.invoke()
    }

    @Keep
    fun setNormalOffsetX(x: Float) {
        normalOffsetX = x.constrain(0f, 1f)
        onViewportChanged()
    }

    fun setZoom(zoom: Float) {
        zoomAmount = interpolate(1f, 1.1f, 1 - zoom.constrain(0f, 1f))
        onViewportChanged()
    }

    private fun onViewportChanged() {
        currentGLPictureSet.recomputeTransformMatrices()
        nextGLPictureSet.recomputeTransformMatrices()
        if (surfaceCreated) {
            callbacks.requestRender()
        }
    }

    private fun blurRadiusAtFrame(f: Float): Float {
        return maxPrescaledBlurPixels * blurInterpolator.getInterpolation(f / blurKeyframes)
    }

    fun setAndConsumeImageLoader(imageLoader: ImageLoader, immediate: Boolean = false) {
        if (!surfaceCreated) {
            queuedNextImageLoader = imageLoader
            return
        }

        if ((loadInProgress || crossfadeAnimator.isRunning) && !immediate) {
            queuedNextImageLoader = imageLoader
            return
        }

        val generation = ++loadGeneration
        if (immediate) {
            // Decode synchronously so the switch is instant (e.g. lock-screen transitions).
            loadInProgress = false
            present(decode(imageLoader), immediate = true)
            return
        }

        // Decode and blur off the GL thread so rendering stays smooth during the switch; only the
        // texture upload (in present) runs on the GL thread.
        loadInProgress = true
        decodeScope.launch {
            val decoded = decode(imageLoader)
            callbacks.queueEventOnGlThread {
                if (generation != loadGeneration) {
                    // Superseded by a newer load; throw this one away.
                    decoded?.recycle()
                    return@queueEventOnGlThread
                }
                loadInProgress = false
                present(decoded, immediate = false)
            }
        }
    }

    private fun present(decoded: DecodedArtwork?, immediate: Boolean) {
        if (decoded == null) {
            return
        }
        if (!surfaceCreated) {
            // Surface went away while we were decoding.
            decoded.recycle()
            return
        }

        if (immediate) {
            // Stop any running cross fade if we're immediately switching to this new image
            crossfadeAnimator.finish()
        }

        if (!demoMode && !preview) {
            SwitchingPhotosStateFlow.value = SwitchingPhotosInProgress(nextGLPictureSet.id)
            ArtworkSizeStateFlow.value = ArtworkSize(decoded.width, decoded.height)
            ArtDetailViewport.setDefaultViewport(nextGLPictureSet.id,
                    decoded.width * 1f / decoded.height,
                    aspectRatio)
        }

        nextGLPictureSet.applyDecoded(decoded)

        crossfadeAnimator.start(if (immediate) 1 else 0, 1) {
            // swap current and next picturesets
            val oldGLPictureSet = currentGLPictureSet
            currentGLPictureSet = nextGLPictureSet
            nextGLPictureSet = GLPictureSet(oldGLPictureSet.id)
            callbacks.requestRender()
            oldGLPictureSet.destroyPictures()
            if (!demoMode) {
                SwitchingPhotosStateFlow.value = SwitchingPhotosDone(currentGLPictureSet.id)
            }
            val loader = queuedNextImageLoader
            if (loader != null) {
                queuedNextImageLoader = null
                setAndConsumeImageLoader(loader, immediate)
            }
        }
        callbacks.requestRender()
    }

    /**
     * Decodes and blurs [imageLoader] into bitmaps ready for upload. Does no GL work, so it can run
     * off the GL thread; the result is handed to [present] / [GLPictureSet.applyDecoded] which do
     * the texture upload on the GL thread. Returns null if the image can't be decoded.
     */
    private fun decode(imageLoader: ImageLoader): DecodedArtwork? {
        val (width, height) = imageLoader.getSize()
        if (width == 0 || height == 0) {
            return null
        }
        val bitmapAspectRatio = width * 1f / height

        // Image darkness drives the dim amount (computed live at draw time from this). It's measured
        // from the small scaled blur source below; if that decode fails we fall back to a tiny
        // dedicated decode.
        var darkness = 0f
        var darknessComputed = false

        // Decode the sharp picture, backing off the resolution if we run out of memory
        val targetHeight = currentHeight
        var sharp: Bitmap? = null
        var sampleSize = 1
        var attempted = false
        while (!attempted) {
            val attemptedWidth = (bitmapAspectRatio * targetHeight / sampleSize).toInt()
            val attemptedHeight = targetHeight / sampleSize
            try {
                sharp = imageLoader.decode(attemptedWidth, attemptedHeight)
                attempted = true
            } catch (_: OutOfMemoryError) {
                sampleSize = sampleSize shl 1
                Log.d(TAG, "Decoding image at ${attemptedWidth}x$attemptedHeight " +
                        "was too large, trying a sample size of $sampleSize")
            }
        }
        if (sharp == null) {
            return null
        }

        // The blur and desaturation happen on the GPU at draw time (see GLBlur); here we only decode
        // the source they operate on — a downscaled copy of the artwork. Its size is fixed by
        // blurredSampleSize (set for MAX_BLUR_AMOUNT), so it's the same for the home and lock
        // variants and survives a blur/grey/dim change without a re-decode.
        var scaledHeight = max(2, (targetHeight / blurredSampleSize).floorEven())
        var scaledWidth = max(4, (scaledHeight * bitmapAspectRatio).toInt().roundMult4())
        // Keep the source within a single GL texture (see MAX_BLUR_SOURCE_DIM). coerceAtMost guards
        // against roundMult4() nudging a dimension back over the cap (and so over the guaranteed
        // GL_MAX_TEXTURE_SIZE, which would yield an incomplete FBO / black blur).
        if (scaledWidth > MAX_BLUR_SOURCE_DIM || scaledHeight > MAX_BLUR_SOURCE_DIM) {
            val downscale = MAX_BLUR_SOURCE_DIM.toFloat() / max(scaledWidth, scaledHeight)
            scaledHeight = max(2, (scaledHeight * downscale).toInt().floorEven()).coerceAtMost(MAX_BLUR_SOURCE_DIM)
            scaledWidth = max(4, (scaledWidth * downscale).toInt().roundMult4()).coerceAtMost(MAX_BLUR_SOURCE_DIM)
        }
        // blurredSampleSize is >= 4, so the blur source is normally much smaller than the sharp
        // picture we already decoded above. Downscale that in-memory bitmap rather than opening the
        // stream and decompressing the file a second time (an IPC + decode for content URIs). The
        // result is Gaussian-blurred at draw time, so the slight quality difference vs a fresh decode
        // is irrelevant. Fall back to a fresh decode only if the sharp picture was shrunk past the
        // blur target by the OOM back-off above and so can't cover these dimensions.
        val tempBitmap = if (sharp.width >= scaledWidth && sharp.height >= scaledHeight) {
            sharp
        } else {
            imageLoader.decode(scaledWidth, scaledHeight)
        }
        val blurSource: Bitmap? = if (tempBitmap != null && tempBitmap.width != 0 && tempBitmap.height != 0) {
            // scale() returns the source untouched when it already matches the target size; copy in
            // that case so the blur source is always its own bitmap (sharp and blurSource are each
            // recycled independently, so they must never alias the same instance).
            val scaledBitmap = tempBitmap.scale(scaledWidth, scaledHeight).let { scaled ->
                if (scaled === sharp) scaled.copy(scaled.config ?: Bitmap.Config.ARGB_8888, false) else scaled
            }
            // Never recycle the shared sharp bitmap here; only a dedicated decode is ours to free.
            if (tempBitmap !== sharp && tempBitmap != scaledBitmap) {
                tempBitmap.recycle()
            }
            // Reuse this small, fully-decoded bitmap for the darkness calculation rather than
            // decoding the source again at 64px purely to measure brightness.
            darkness = scaledBitmap.darkness()
            darknessComputed = true
            scaledBitmap
        } else {
            Log.e(TAG, "ImageLoader failed to decode the blur source")
            null
        }

        // The blur source decode failed, so measure darkness from a small dedicated decode.
        if (!darknessComputed) {
            val darknessBitmap = imageLoader.decode(64)
            darkness = darknessBitmap.darkness()
            darknessBitmap?.recycle()
        }

        return DecodedArtwork(sharp, blurSource, darkness, bitmapAspectRatio, width, height)
    }

    /** Bitmaps decoded off the GL thread, awaiting texture upload (see [decode]/[present]). */
    private class DecodedArtwork(
            val sharp: Bitmap,
            // Downscaled source for the runtime GPU blur/desaturate; null only if its decode failed.
            val blurSource: Bitmap?,
            // Mean relative luminance of the artwork; the dim amount is derived from it live.
            val darkness: Float,
            val bitmapAspectRatio: Float,
            val width: Int,
            val height: Int
    ) {
        fun recycle() {
            sharp.recycle()
            blurSource?.recycle()
        }
    }

    private inner class GLPictureSet(val id: Int) {
        private val projectionMatrix = FloatArray(16)
        private val mvpMatrix = FloatArray(16)
        private var sharpPicture: GLPicture? = null
        // GPU blur of the downscaled source, drawn over the sharp picture as the artwork blurs.
        // Null only if the blur source failed to decode.
        private var blur: GLBlur? = null
        private var hasBitmap = false
        private var bitmapAspectRatio = 1f
        // Artwork luminance; the dim amount is computed from it live (see dimAmountFor).
        var darkness = 0f

        /**
         * Uploads the already-decoded [decoded] bitmaps as GL textures (the only part that must
         * run on the GL thread) and recycles them. The decode happened off-thread in [decode];
         * the blur itself happens at draw time on the GPU (see [GLBlur]).
         */
        fun applyDecoded(decoded: DecodedArtwork) {
            destroyPictures()

            hasBitmap = true
            bitmapAspectRatio = decoded.bitmapAspectRatio
            darkness = decoded.darkness

            sharpPicture = decoded.sharp.toGLPicture()
            blur = decoded.blurSource?.let { GLBlur().apply { setSource(it) } }

            decoded.recycle()
            recomputeTransformMatrices()
            callbacks.requestRender()
        }

        fun recomputeTransformMatrices() {
            // Nothing to transform until this set has an image. Avoids recomputing the "next"
            // (empty) picture set on every offset change while scrolling outside a crossfade.
            if (!hasBitmap) {
                return
            }
            val screenToBitmapAspectRatio = aspectRatio / bitmapAspectRatio
            if (screenToBitmapAspectRatio == 0f) {
                return
            }

            // Ensure the bitmap is as wide as the screen by applying zoom if necessary
            // ignoring any system wide zoom requests while the Art Detail screen is open
            val zoom = max(1f, screenToBitmapAspectRatio) *
                    (if (ArtDetailOpen.value) 1f else zoomAmount)

            // Total scale factors in both zoom and scale due to aspect ratio.
            val scaledBitmapToScreenAspectRatio = zoom / screenToBitmapAspectRatio

            // At most pan across 1.8 screenfuls (2 screenfuls + some parallax)
            // TODO: if we know the number of home screen pages, use that number here
            val maxPanScreenWidths = min(1.8f, scaledBitmapToScreenAspectRatio)

            currentViewport.apply {
                left = interpolate(-1f, 1f,
                        interpolate(
                                (1 - maxPanScreenWidths / scaledBitmapToScreenAspectRatio) / 2,
                                (1 + (maxPanScreenWidths - 2) / scaledBitmapToScreenAspectRatio) / 2,
                                normalOffsetX))
                right = left + 2f / scaledBitmapToScreenAspectRatio
                bottom = -1f / zoom
                top = 1f / zoom
            }

            val focusAmount = (blurKeyframes - blurAnimator.currentValue) / blurKeyframes
            if (blurRelatedToArtDetailMode && focusAmount > 0) {
                val artDetailViewport = ArtDetailViewport.getViewport(id)
                if (artDetailViewport.width() == 0f || artDetailViewport.height() == 0f) {
                    if (!demoMode && !preview) {
                        // reset art detail viewport
                        ArtDetailViewport.setViewport(id,
                                uninterpolate(-1f, 1f, currentViewport.left),
                                uninterpolate(1f, -1f, currentViewport.top),
                                uninterpolate(-1f, 1f, currentViewport.right),
                                uninterpolate(1f, -1f, currentViewport.bottom))
                    }
                } else {
                    // interpolate
                    currentViewport.apply {
                        left = interpolate(
                                left,
                                interpolate(-1f, 1f, artDetailViewport.left),
                                focusAmount)
                        top = interpolate(
                                top,
                                interpolate(1f, -1f, artDetailViewport.top),
                                focusAmount)
                        right = interpolate(
                                right,
                                interpolate(-1f, 1f, artDetailViewport.right),
                                focusAmount)
                        bottom = interpolate(
                                bottom,
                                interpolate(1f, -1f, artDetailViewport.bottom),
                                focusAmount)
                    }
                }
            }

            Matrix.orthoM(projectionMatrix, 0,
                    currentViewport.left, currentViewport.right,
                    currentViewport.bottom, currentViewport.top,
                    1f, 10f)
        }

        fun drawFrame(globalAlpha: Float) {
            if (!hasBitmap || globalAlpha <= 0f) {
                return
            }
            val sharp = sharpPicture ?: return

            Matrix.multiplyMM(mvpMatrix, 0, viewMatrix, 0, modelMatrix, 0)
            Matrix.multiplyMM(mvpMatrix, 0, projectionMatrix, 0, mvpMatrix, 0)

            // blurFraction goes 0 (focused) -> 1 (fully blurred); grey ramps with it like the
            // original. Desaturation is applied at full resolution (on the sharp picture, and on
            // the blur overlay's composite) so grey stays sharp even when blur is light or off.
            val blurFraction = blurAnimator.currentValue / blurKeyframes
            val grey = maxGrey / 500f * blurFraction
            // How much of the blurred overlay shows: the focus fraction, faded in with the blur
            // radius so a tiny radius doesn't abruptly swap the full-res sharp for the downscaled
            // source. 0 -> sharp only (incl. grey-without-blur); 1 -> blur fully covers the sharp.
            val blurMix = (maxPrescaledBlurPixels / BLUR_FADE_IN_PIXELS).coerceIn(0f, 1f)
            val blurWeight = blurFraction * blurMix

            val overlay = blur
            if (blurWeight <= 0f || overlay == null) {
                sharp.draw(mvpMatrix, globalAlpha, grey)
                return
            }

            // Composite lerp(sharp, blurred, blurWeight) onto the background at globalAlpha. The two
            // draws' alphas are recomposed so the visible result is exactly that single blend rather
            // than the blurred layer merely painted over the sharp one — without this, an artwork
            // crossfade (globalAlpha < 1) would show the incoming image partly sharp. When the blur
            // fully covers (blurWeight == 1) the sharp layer contributes nothing, so its full-res,
            // tiled draw is skipped entirely.
            if (blurWeight < 1f) {
                val sharpAlpha = globalAlpha * (1f - blurWeight) / (1f - globalAlpha * blurWeight)
                sharp.draw(mvpMatrix, sharpAlpha, grey)
            }
            overlay.drawBlurred(
                    mvpMatrix,
                    blurRadiusAtFrame(blurAnimator.currentValue),
                    globalAlpha * blurWeight,
                    grey)
        }

        fun destroyPictures() {
            sharpPicture?.destroy()
            sharpPicture = null
            blur?.destroy()
            blur = null
        }
    }

    fun destroy() {
        decodeScope.cancel()
        currentGLPictureSet.destroyPictures()
        nextGLPictureSet.destroyPictures()
    }

    fun setIsBlurred(isBlurred: Boolean, artDetailMode: Boolean) {
        if (artDetailMode && !isBlurred && !demoMode && !preview) {
            // Reset art detail viewport
            ArtDetailViewport.setViewport(0, 0f, 0f, 0f, 0f)
            ArtDetailViewport.setViewport(1, 0f, 0f, 0f, 0f)
        }

        blurRelatedToArtDetailMode = artDetailMode
        this.isBlurred = isBlurred
        blurAnimator.start(endValue = if (isBlurred) blurKeyframes else 0) {}
        callbacks.requestRender()
    }

    interface Callbacks {
        fun requestRender()

        /** Posts [event] to run on the GL thread (used to upload off-thread decode results). */
        fun queueEventOnGlThread(event: () -> Unit)
    }
}