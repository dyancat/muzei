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
import android.media.MediaMetadataRetriever
import android.net.Uri
import android.opengl.EGL14
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
        private val preview: Boolean = false,
        // Whether this renderer's video visibility is driven externally by a wallpaper engine's
        // onVisibilityChanged (true), or it is a standalone in-app view that is simply always visible
        // while rendering (false). Controls whether a video starts claiming the shared decoder (see
        // videoSurfaceVisible).
        private val videoVisibilityDrivenExternally: Boolean = false
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

        // At most this many screenfuls of a wide image/video are ever panned across (2 screenfuls of
        // home-screen travel minus some parallax); see recomputeTransformMatrices. A wider source is
        // only ever partially shown, so a video is cropped to this pannable extent before capture
        // (see GLPictureSet.applyVideo) to avoid storing columns that can never appear on screen.
        private const val MAX_PAN_SCREEN_WIDTHS = 1.8f
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

    private var queuedNextSource: RenderSource? = null
    // Whether this engine's surface is on screen. Only the on-screen engine claims the shared video
    // decoder's output (see SharedVideoPlayer / updateVideoBinding), so a video created while this
    // engine is hidden (e.g. artwork advancing on the lock engine while you're on home) doesn't grab
    // the decoder. A standalone in-app view is always visible while rendering; a wallpaper engine
    // starts hidden and is driven by onVisibilityChanged. GL thread.
    private var videoSurfaceVisible = !videoVisibilityDrivenExternally
    // The next artwork is decoded and blurred off the GL thread so rendering (e.g. scrolling)
    // stays smooth during a switch; only the texture upload runs on the GL thread. loadInProgress
    // serialises with the crossfade, and loadGeneration discards a decode that's been superseded.
    private val decodeScope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
    private var loadInProgress = false
    private var loadGeneration = 0

    private var surfaceCreated: Boolean = false

    // Centered until a scrolling launcher pushes a real offset (see onOffsetsChanged). onOffsetsChanged
    // is push-only — there's no API to query the current offset — and a non-scrolling launcher never
    // pushes one, so 0.5 frames the artwork centered rather than hard against the left edge.
    @Volatile
    private var normalOffsetX: Float = 0.5f
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
        setNormalOffsetX(0.5f)
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
        GLVideo.initGl()

        colorOverlay = GLColorOverlay()

        surfaceCreated = true
        val source = queuedNextSource
        if (source != null) {
            queuedNextSource = null
            setAndConsumeSource(source)
        }
    }

    override fun onSurfaceChanged(unused: GL10, width: Int, height: Int) {
        GLES20.glViewport(0, 0, width, height)
        GLBlur.setScreenSize(width, height)
        GLVideo.setScreenSize(width, height)
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

    fun setAndConsumeSource(source: RenderSource, immediate: Boolean = false) {
        if (!surfaceCreated) {
            queuedNextSource = source
            return
        }

        // Skip recreating the player when the same video is already the settled current artwork.
        // reloadCurrentArtwork() fires on every surface/size/lifecycle change and always builds a
        // fresh source, so without this each such event spins up another ExoPlayer for the video
        // already playing; several coexisting players (each buffering) exhaust the heap.
        if (source is RenderSource.Video
                && !loadInProgress && !crossfadeAnimator.isRunning && queuedNextSource == null
                && currentGLPictureSet.videoUri == source.uri) {
            return
        }

        if ((loadInProgress || crossfadeAnimator.isRunning) && !immediate) {
            queuedNextSource = source
            return
        }

        val generation = ++loadGeneration
        when (source) {
            is RenderSource.Image -> {
                val imageLoader = source.loader
                if (immediate) {
                    // Decode synchronously so the switch is instant (e.g. lock-screen transitions).
                    loadInProgress = false
                    present(decode(imageLoader), immediate = true)
                    return
                }
                // Decode and blur off the GL thread so rendering stays smooth during the switch;
                // only the texture upload (in present) runs on the GL thread.
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
            is RenderSource.Video -> {
                val uri = source.uri
                if (immediate) {
                    loadInProgress = false
                    presentVideo(decodeVideo(uri), immediate = true)
                    return
                }
                // Probe the video (size + poster darkness) off the GL thread; the GLVideo itself
                // (external texture + player) is created on the GL thread in presentVideo.
                loadInProgress = true
                decodeScope.launch {
                    val decodedVideo = decodeVideo(uri)
                    callbacks.queueEventOnGlThread {
                        if (generation != loadGeneration) {
                            return@queueEventOnGlThread
                        }
                        loadInProgress = false
                        presentVideo(decodedVideo, immediate = false)
                    }
                }
            }
        }
    }

    private fun present(decoded: DecodedArtwork?, immediate: Boolean) {
        if (decoded == null) {
            // This decode produced nothing (e.g. the initial content-URI reload tried to decode a
            // video as a still). Don't strand a source queued behind it — try that one now.
            consumeQueuedSource(immediate)
            return
        }
        if (!surfaceCreated || !hasGlContext()) {
            // Surface/context went away while we were decoding (e.g. the in-app preview being torn
            // down). Uploading a texture now would hit a dead context and crash; skip it.
            decoded.recycle()
            consumeQueuedSource(immediate)
            return
        }
        nextGLPictureSet.applyDecoded(decoded)
        startCrossfade(decoded.width, decoded.height, immediate)
    }

    private fun presentVideo(decoded: DecodedVideo?, immediate: Boolean) {
        if (decoded == null) {
            consumeQueuedSource(immediate)
            return
        }
        if (!surfaceCreated || !hasGlContext()) {
            // Surface/context went away while we were probing the video; don't build GL objects on
            // a dead context.
            consumeQueuedSource(immediate)
            return
        }
        // Build the player now, but hold the crossfade until its first frame is decoded. The player
        // is created and buffers asynchronously on the main thread, so starting the crossfade here
        // would fade the outgoing artwork into a blank (black) external texture until playback
        // catches up — the "goes black before the video appears" flash. onFirstFrame is delivered on
        // the main thread, so hop back to the GL thread to start the crossfade.
        val generation = loadGeneration
        nextGLPictureSet.applyVideo(decoded) {
            callbacks.queueEventOnGlThread {
                if (generation != loadGeneration) {
                    // Superseded by a newer load; that load's own first frame drives its crossfade.
                    return@queueEventOnGlThread
                }
                // Cross-fade using the cropped strip's dimensions (what's actually shown), so the
                // published aspect ratio matches the pan/zoom/art-detail transforms.
                startCrossfade(nextGLPictureSet.videoStripWidth, nextGLPictureSet.videoStripHeight,
                        immediate)
            }
        }
    }

    /**
     * Whether a live EGL context is current on this (GL) thread. A queued present can run after the
     * surface/context has been torn down — notably when the in-app preview's [GLTextureView] is
     * detached — and GL calls (glGenTextures etc.) would then fail; this lets callers skip that work.
     */
    private fun hasGlContext(): Boolean =
            EGL14.eglGetCurrentContext() != EGL14.EGL_NO_CONTEXT

    /**
     * Consumes any source queued behind an in-flight load (see [queuedNextSource]). Normally the
     * crossfade-end callback does this, but a load that ends without a crossfade (a null decode, or
     * a lost surface) must also drain the queue or a queued source is stranded — which stranded the
     * video behind the initial content-URI image reload on a cold start.
     */
    private fun consumeQueuedSource(immediate: Boolean) {
        val source = queuedNextSource
        if (source != null) {
            queuedNextSource = null
            setAndConsumeSource(source, immediate)
        }
    }

    /**
     * Starts the crossfade from the current picture set to [nextGLPictureSet] (already loaded with
     * an image or video), publishing the switch state and, on completion, swapping the sets,
     * destroying the outgoing one, and consuming any queued source. Shared by [present]/[presentVideo].
     */
    private fun startCrossfade(width: Int, height: Int, immediate: Boolean) {
        if (immediate) {
            // Stop any running cross fade if we're immediately switching to this new artwork
            crossfadeAnimator.finish()
        }

        if (!demoMode && !preview) {
            SwitchingPhotosStateFlow.value = SwitchingPhotosInProgress(nextGLPictureSet.id)
            ArtworkSizeStateFlow.value = ArtworkSize(width, height)
            ArtDetailViewport.setDefaultViewport(nextGLPictureSet.id,
                    width * 1f / height,
                    aspectRatio)
        }

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
            consumeQueuedSource(immediate)
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
            // Downscale in halving steps rather than one big createScaledBitmap: a single
            // large-factor bilinear scale only samples a 2x2 neighbourhood, so reducing by the
            // 4-16x blurredSampleSize skips source pixels and bakes in aliasing that a small blur
            // radius no longer hides. downscaleForBlur always returns its own bitmap (never sharp),
            // preserving the invariant that sharp and blurSource are independently recyclable.
            val scaledBitmap = downscaleForBlur(tempBitmap, scaledWidth, scaledHeight)
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

    /**
     * Downscales [source] to [targetWidth] x [targetHeight] by repeated halving so each step is at
     * most a 2x reduction, which bilinear filtering handles without skipping pixels. Doing the whole
     * 4-16x reduction in one createScaledBitmap would sample only a 2x2 neighbourhood per output
     * pixel and alias; halving averages the intermediate pixels (mip-style) so the blur source is
     * alias-free at any radius. Never recycles [source]; always returns a distinct new bitmap.
     */
    private fun downscaleForBlur(source: Bitmap, targetWidth: Int, targetHeight: Int): Bitmap {
        var current = source
        var currentOwned = false // whether `current` is an intermediate we may recycle
        while (current.width >= targetWidth * 2 && current.height >= targetHeight * 2) {
            val next = current.scale(current.width / 2, current.height / 2)
            if (currentOwned) current.recycle()
            current = next
            currentOwned = true
        }
        if (current.width == targetWidth && current.height == targetHeight) {
            // Already exact: hand back the owned intermediate, or a copy so we never return `source`.
            return if (currentOwned) current
            else current.copy(current.config ?: Bitmap.Config.ARGB_8888, false)
        }
        val scaled = current.scale(targetWidth, targetHeight)
        if (currentOwned) current.recycle()
        return scaled
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

    /**
     * Probes a video [uri] for its display size and a poster-frame darkness, off the GL thread.
     * Returns null if it can't be read (the caller then leaves the current artwork in place).
     */
    private fun decodeVideo(uri: Uri): DecodedVideo? {
        val retriever = MediaMetadataRetriever()
        try {
            val opened = context.contentResolver.openAssetFileDescriptor(uri, "r")?.use { afd ->
                if (afd.declaredLength >= 0) {
                    retriever.setDataSource(afd.fileDescriptor, afd.startOffset, afd.declaredLength)
                } else {
                    retriever.setDataSource(afd.fileDescriptor)
                }
                true
            } ?: false
            if (!opened) {
                return null
            }
            val rawWidth = retriever.extractMetadata(
                    MediaMetadataRetriever.METADATA_KEY_VIDEO_WIDTH)?.toIntOrNull() ?: 0
            val rawHeight = retriever.extractMetadata(
                    MediaMetadataRetriever.METADATA_KEY_VIDEO_HEIGHT)?.toIntOrNull() ?: 0
            if (rawWidth == 0 || rawHeight == 0) {
                return null
            }
            // Swap dimensions for portrait recordings so the display aspect ratio is correct.
            val rotation = retriever.extractMetadata(
                    MediaMetadataRetriever.METADATA_KEY_VIDEO_ROTATION)?.toIntOrNull() ?: 0
            val (width, height) = if (rotation == 90 || rotation == 270) {
                rawHeight to rawWidth
            } else {
                rawWidth to rawHeight
            }
            // A poster frame gives the dim amount its darkness, matching how images measure it.
            val poster = retriever.frameAtTime
            val darkness = poster.darkness()
            poster?.recycle()
            return DecodedVideo(uri, width, height, darkness)
        } catch (e: Exception) {
            Log.w(TAG, "Unable to read video $uri: ${e.message}")
            return null
        } finally {
            retriever.release()
        }
    }

    /** A probed video awaiting GLVideo creation on the GL thread (see [decodeVideo]/[presentVideo]). */
    private class DecodedVideo(
            val uri: Uri,
            val width: Int,
            val height: Int,
            val darkness: Float
    )

    private inner class GLPictureSet(val id: Int) {
        private val projectionMatrix = FloatArray(16)
        private val mvpMatrix = FloatArray(16)
        private var sharpPicture: GLPicture? = null
        // GPU blur of the downscaled source, drawn over the sharp picture as the artwork blurs.
        // Null only if the blur source failed to decode.
        private var blur: GLBlur? = null
        // Set instead of sharpPicture/blur when this set holds a video (see applyVideo). Drawn via the
        // same blur/composite machinery, fed each frame by the live video texture.
        private var video: GLVideo? = null
        // The URI of the video this set is currently playing (null for an image or empty set). Used
        // to skip recreating an identical player when a surface/size reload re-requests it.
        var videoUri: Uri? = null
            private set
        // Dimensions of the (centre-cropped) strip a video is actually shown as, so the crossfade
        // publishes the same aspect ratio the pan/zoom/art-detail transforms use (see applyVideo).
        // Zero for an image or empty set.
        var videoStripWidth = 0
            private set
        var videoStripHeight = 0
            private set
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

        /**
         * Sets this set up to play [decoded] as a video: creates the [GLVideo] (external texture +
         * player) sized for a screen-height sharp capture and a downscaled blur source (matching the
         * image blur source sizing), and points the drawing path at it. Runs on the GL thread.
         */
        fun applyVideo(decoded: DecodedVideo, onFirstFrame: () -> Unit) {
            destroyPictures()

            hasBitmap = true
            val fullAspectRatio = decoded.width * 1f / decoded.height
            darkness = decoded.darkness

            // A source wider than the pannable extent (see MAX_PAN_SCREEN_WIDTHS) is only ever shown
            // in part, so crop it to the centre strip that can actually be reached and treat that
            // strip as the artwork. captureFraction is the fraction of the width kept (1 = no crop);
            // the outer columns are dropped before capture (see GLVideo) so we never allocate sharp /
            // downsample / blur textures for pixels that can't appear on screen.
            val fullScreenWidths = if (aspectRatio > 0f) fullAspectRatio / aspectRatio else 1f
            val captureFraction = if (fullScreenWidths > MAX_PAN_SCREEN_WIDTHS)
                MAX_PAN_SCREEN_WIDTHS / fullScreenWidths else 1f
            bitmapAspectRatio = fullAspectRatio * captureFraction

            val targetHeight = if (currentHeight > 0) currentHeight else decoded.height
            // The video always decodes at full width (only the centre strip is captured), so size the
            // frame from the full aspect ratio at ~screen height, then apply the max-texture cap to
            // the whole frame.
            var fullHeight = min(decoded.height, targetHeight).coerceAtLeast(2)
            var fullWidth = max(2, (fullHeight * fullAspectRatio).toInt())
            // Only downscale if we'd exceed the GPU's max texture size (typically >= 4096). Using the
            // small blur-source cap here would shrink the sharp capture well below screen resolution
            // on high-res (e.g. QHD+) displays and make the video look soft.
            val maxTextureSize = IntArray(1)
            GLES20.glGetIntegerv(GLES20.GL_MAX_TEXTURE_SIZE, maxTextureSize, 0)
            val cap = maxTextureSize[0]
            if (cap > 0 && (fullWidth > cap || fullHeight > cap)) {
                val downscale = cap.toFloat() / max(fullWidth, fullHeight)
                fullWidth = max(2, (fullWidth * downscale).toInt())
                fullHeight = max(2, (fullHeight * downscale).toInt())
            }
            // Sharp capture is just the kept strip of the (capped) full frame.
            val sharpHeight = fullHeight
            val sharpWidth = max(2, (fullWidth * captureFraction).toInt())
            // Downscaled blur source, sized like the image blur source so the blur radius matches.
            val blurHeight = max(2, (targetHeight / blurredSampleSize).floorEven())
            val blurWidth = max(4, (blurHeight * bitmapAspectRatio).toInt().roundMult4())

            video = GLVideo(context, decoded.uri, sharpWidth, sharpHeight, blurWidth, blurHeight,
                    captureFraction,
                    requestRender = { callbacks.requestRender() },
                    onFirstFrame = onFirstFrame)
            videoUri = decoded.uri
            videoStripWidth = sharpWidth
            videoStripHeight = sharpHeight
            // Claim the shared decoder's output for this video only if this engine is the one on
            // screen; otherwise it stays a passive texture until its engine becomes visible (e.g. a
            // video that advanced on the lock engine while you're on the home screen).
            if (videoSurfaceVisible) {
                video?.bind()
            }
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
            val maxPanScreenWidths = min(MAX_PAN_SCREEN_WIDTHS, scaledBitmapToScreenAspectRatio)

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

            // Video path: pull the latest frame and draw it through the same blur/composite passes.
            val vid = video
            if (vid != null) {
                vid.updateFrame()
                // blurMix > 0 means blur is enabled for the current screen, so the video keeps (or
                // allocates) its effect textures; when it's 0 (blur turned off) they are reclaimed.
                vid.draw(mvpMatrix, globalAlpha, blurWeight,
                        blurRadiusAtFrame(blurAnimator.currentValue), grey,
                        effectsActive = blurMix > 0f)
                return
            }

            val sharp = sharpPicture ?: return
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
            video?.release()
            video = null
            videoUri = null
        }

        /** Claims the shared decoder's output for this set's video (if any) — this engine on screen. */
        fun bindVideo() {
            video?.bind()
        }

        /** Releases this set's claim on the shared decoder's output (if any). */
        fun unbindVideo() {
            video?.unbind()
        }

        /** Releases this set's video player (if any). Idempotent; safe off the GL thread. */
        fun releaseVideoPlayer() {
            video?.release()
        }
    }

    fun destroy() {
        decodeScope.cancel()
        currentGLPictureSet.destroyPictures()
        nextGLPictureSet.destroyPictures()
    }

    /**
     * Records whether this engine's surface is on screen and (re)points the shared video decoder's
     * output accordingly: the on-screen engine's foreground video claims it, hidden engines release
     * their claim (a no-op for image artwork). Screen on/off is handled globally by
     * [SharedVideoPlayer.setScreenOn], so this is purely about which engine owns the output. Must run
     * on the GL thread (it touches the picture sets).
     */
    fun setVideoSurfaceVisible(visible: Boolean) {
        videoSurfaceVisible = visible
        updateVideoBinding()
    }

    /**
     * Binds the foreground video (the incoming one during a load/crossfade, else the current) to the
     * shared decoder while this engine is on screen, and releases every claim while it isn't. Binding
     * the incoming video replaces any previous claim, so the outgoing video simply freezes on its
     * last frame through the crossfade — acceptable for a wallpaper and the price of one shared
     * decoder.
     */
    private fun updateVideoBinding() {
        if (videoSurfaceVisible) {
            // The incoming set (during a load/crossfade) is the foreground video; else the current.
            val foreground = if (nextGLPictureSet.videoUri != null) nextGLPictureSet
                    else currentGLPictureSet
            foreground.bindVideo()
        } else {
            currentGLPictureSet.unbindVideo()
            nextGLPictureSet.unbindVideo()
        }
    }

    /**
     * Releases any video players immediately. Safe to call from the main thread (e.g. the in-app
     * preview being destroyed) so the codec is freed even if the GL thread exits before running the
     * queued [destroy] — see GLVideo.release. Idempotent.
     */
    fun releaseVideoPlayers() {
        currentGLPictureSet.releaseVideoPlayer()
        nextGLPictureSet.releaseVideoPlayer()
    }

    fun setIsBlurred(isBlurred: Boolean, artDetailMode: Boolean) {
        if (artDetailMode && !isBlurred && !demoMode && !preview) {
            // Clear the art-detail framing on open so recomputeTransformMatrices re-seeds it from the
            // current launcher crop. Opening art detail therefore lands on the launcher position: if a
            // programmatic pan back to the launcher (a close that's still animating) is in flight, this
            // jumps straight to its final position instead of reversing toward the abandoned pan.
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