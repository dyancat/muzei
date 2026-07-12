/*
 * Copyright 2024 Google Inc.
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

import android.content.Context
import android.graphics.SurfaceTexture
import android.net.Uri
import android.opengl.EGL14
import android.opengl.GLES11Ext
import android.opengl.GLES20
import android.os.Handler
import android.os.Looper
import android.view.Surface
import kotlin.math.roundToInt

/**
 * Plays a video as artwork, mirroring how [GLBlur] renders still images: each frame the live video
 * frame (an external `GL_TEXTURE_EXTERNAL_OES` fed by a video decoder through a [SurfaceTexture]) is
 * captured into an ordinary 2D FBO — the "sharp" texture — which is then drawn and blurred by the
 * same [GLBlur] composite/blur passes the image path uses. So blur / grey / dim / pan-zoom /
 * crossfade all apply to video exactly as they do to images.
 *
 * The actual decoding is done by the process-wide [SharedVideoPlayer], not a per-GLVideo player:
 * every engine (home, lock, preview) has its own [GLVideo] with its own [SurfaceTexture], but they
 * share one hardware decoder whose output surface is pointed at whichever engine is on screen (see
 * [bind]/[unbind]). Only that engine receives frames.
 *
 * Threading: the GL objects (external texture, [SurfaceTexture], FBO, [GLBlur]) are created and used
 * on the GL thread; the shared player lives on the main thread ([bind]/[unbind]/[release] hop there
 * inside [SharedVideoPlayer]). New frames arrive via [SurfaceTexture.setOnFrameAvailableListener],
 * which simply asks the renderer to draw again (the renderer is `RENDERMODE_WHEN_DIRTY`), so an
 * unbound video produces no frames and therefore no redraws — the battery win on AOD / when hidden.
 */
internal class GLVideo(
        context: Context,
        private val uri: Uri,
        sharpWidth: Int,
        sharpHeight: Int,
        blurWidth: Int,
        blurHeight: Int,
        // Fraction of the source video's width to keep, centred. Wide videos are cropped to the
        // pannable extent (see MuzeiBlurRenderer.MAX_PAN_SCREEN_WIDTHS); the never-rendered outer
        // columns are discarded before capture so the sharp/downsample/blur textures only ever hold
        // the strip that can actually be shown. 1f keeps the whole width.
        captureFraction: Float,
        private val requestRender: () -> Unit,
        // Invoked once, on the main thread, when the first decoded frame becomes available in the
        // SurfaceTexture. The renderer holds the crossfade until then so it never fades the outgoing
        // artwork into this video's still-blank (black) external texture before the player has
        // produced a frame.
        private val onFirstFrame: () -> Unit
) {
    companion object {
        private const val TAG = "GLVideo"

        // Renders the external video texture into a 2D FBO. The SurfaceTexture transform matrix
        // (uSTMatrix) maps the [0,1] quad coordinates onto the correct region/orientation of the
        // external image.
        private const val OES_VERTEX_SHADER = "" +
                "attribute vec4 aPosition;" +
                "attribute vec4 aTexCoords;" +
                "uniform mat4 uSTMatrix;" +
                "uniform vec2 uCrop;" +   // (scale, offset): keep only the centre uCrop.x of the width
                "varying vec2 vTexCoords;" +
                "void main(){" +
                "  gl_Position = aPosition;" +
                "  vec4 tc = aTexCoords;" +
                "  tc.x = tc.x * uCrop.x + uCrop.y;" +
                "  vTexCoords = (uSTMatrix * tc).xy;" +
                "}"

        private const val OES_FRAGMENT_SHADER = "" +
                "#extension GL_OES_EGL_image_external : require\n" +
                "precision mediump float;" +
                "uniform samplerExternalOES uTexture;" +
                "varying vec2 vTexCoords;" +
                "void main(){" +
                "  gl_FragColor = texture2D(uTexture, vTexCoords);" +
                "}"

        // Draws the external video texture straight to the bound framebuffer (the screen) through
        // the MVP, applying alpha and desaturation — the sharp path. This avoids the extra
        // external->FBO->screen resample the blur path needs, so an unblurred video is a single
        // upscale (as sharp as a normal video player) rather than two bilinear passes.
        private const val DIRECT_VERTEX_SHADER = "" +
                "uniform mat4 uMVPMatrix;" +
                "uniform mat4 uSTMatrix;" +
                "uniform vec2 uCrop;" +   // (scale, offset): keep only the centre uCrop.x of the width
                "attribute vec4 aPosition;" +
                "attribute vec4 aTexCoords;" +
                "varying vec2 vTexCoords;" +
                "void main(){" +
                "  gl_Position = uMVPMatrix * aPosition;" +
                "  vec4 tc = aTexCoords;" +
                "  tc.x = tc.x * uCrop.x + uCrop.y;" +
                "  vTexCoords = (uSTMatrix * tc).xy;" +
                "}"

        private const val DIRECT_FRAGMENT_SHADER = "" +
                "#extension GL_OES_EGL_image_external : require\n" +
                "precision mediump float;" +
                "uniform samplerExternalOES uTexture;" +
                "uniform float uAlpha;" +
                "uniform float uGrey;" +
                "varying vec2 vTexCoords;" +
                "void main(){" +
                "  vec4 c = texture2D(uTexture, vTexCoords);" +
                "  float lum = dot(c.rgb, vec3(0.299, 0.587, 0.114));" +
                "  c.rgb = mix(c.rgb, vec3(lum), clamp(uGrey, 0.0, 1.0));" +
                "  c.a = uAlpha;" +
                "  gl_FragColor = c;" +
                "}"

        // 2x box downsample used by the progressive chain (see renderDownsampleChain). Averages an
        // explicit 2x2 of source texels via four taps offset by half a source texel, computed in
        // the vertex shader at highp so the offsets stay exact on large textures. A single
        // GL_LINEAR tap only box-averages when the reduction lands exactly on 2x; for odd
        // dimensions it leaves residual aliasing that crawls once the (video) source is in motion.
        private const val COPY_VERTEX_SHADER = "" +
                "attribute vec4 aPosition;" +
                "attribute vec2 aTexCoords;" +
                "uniform vec2 uHalfTexel;" +   // half a source texel: (0.5/srcW, 0.5/srcH)
                "varying vec2 vT0;" +
                "varying vec2 vT1;" +
                "varying vec2 vT2;" +
                "varying vec2 vT3;" +
                "void main(){" +
                "  vT0 = aTexCoords + vec2(-uHalfTexel.x, -uHalfTexel.y);" +
                "  vT1 = aTexCoords + vec2( uHalfTexel.x, -uHalfTexel.y);" +
                "  vT2 = aTexCoords + vec2(-uHalfTexel.x,  uHalfTexel.y);" +
                "  vT3 = aTexCoords + vec2( uHalfTexel.x,  uHalfTexel.y);" +
                "  gl_Position = aPosition;" +
                "}"

        private const val COPY_FRAGMENT_SHADER = "" +
                "precision highp float;" +
                "uniform sampler2D uTexture;" +
                "varying vec2 vT0;" +
                "varying vec2 vT1;" +
                "varying vec2 vT2;" +
                "varying vec2 vT3;" +
                "void main(){" +
                "  gl_FragColor = 0.25 * (texture2D(uTexture, vT0) + texture2D(uTexture, vT1)" +
                "                       + texture2D(uTexture, vT2) + texture2D(uTexture, vT3));" +
                "}"

        // Full-screen quad (NDC), TL, BL, BR, TL, BR, TR.
        private val QUAD_POSITIONS = floatArrayOf(
                -1f, 1f, 0f,   -1f, -1f, 0f,   1f, -1f, 0f,
                -1f, 1f, 0f,    1f, -1f, 0f,   1f, 1f, 0f)
        // Identity texcoords ((pos + 1) / 2) for the downsample copy: a straight 2D->2D scale that
        // preserves orientation, so the chain's output feeds GLBlur exactly as sharpTexture did.
        private val COPY_TEXCOORDS = floatArrayOf(
                0f, 1f,   0f, 0f,   1f, 0f,
                0f, 1f,   1f, 0f,   1f, 1f)
        // Base texture coordinates (as vec4 s,t,0,1) for the ST-matrix multiply, with the vertical
        // (t) axis flipped relative to QUAD_POSITIONS so the frame captured into the FBO comes out
        // upright for GLBlur's composite pass (which samples top-left origin, see COMPOSITE_TEXCOORDS
        // there). Rendering into an FBO flips vertically, so the source t is inverted to compensate.
        private val OES_TEXCOORDS = floatArrayOf(
                0f, 0f, 0f, 1f,   0f, 1f, 0f, 1f,   1f, 1f, 0f, 1f,
                0f, 0f, 0f, 1f,   1f, 1f, 0f, 1f,   1f, 0f, 0f, 1f)
        // Texcoords for the direct-to-screen draw. Same content as OES_TEXCOORDS but without the FBO
        // vertical flip (one fewer render target), so the frame is upright drawn straight to screen.
        private val DIRECT_TEXCOORDS = floatArrayOf(
                0f, 1f, 0f, 1f,   0f, 0f, 0f, 1f,   1f, 0f, 0f, 1f,
                0f, 1f, 0f, 1f,   1f, 0f, 0f, 1f,   1f, 1f, 0f, 1f)
        private const val VERTICES = 6

        private var program = 0
        private var positionHandle = 0
        private var texCoordsHandle = 0
        private var textureHandle = 0
        private var stMatrixHandle = 0
        private var cropHandle = 0

        private var directProgram = 0
        private var directPositionHandle = 0
        private var directTexCoordsHandle = 0
        private var directTextureHandle = 0
        private var directStMatrixHandle = 0
        private var directMvpHandle = 0
        private var directAlphaHandle = 0
        private var directGreyHandle = 0
        private var directCropHandle = 0

        private var copyProgram = 0
        private var copyPositionHandle = 0
        private var copyTexCoordsHandle = 0
        private var copyTextureHandle = 0
        private var copyHalfTexelHandle = 0

        private var screenWidth = 0
        private var screenHeight = 0

        fun initGl() {
            program = GLUtil.createAndLinkProgram(
                    GLUtil.loadShader(GLES20.GL_VERTEX_SHADER, OES_VERTEX_SHADER),
                    GLUtil.loadShader(GLES20.GL_FRAGMENT_SHADER, OES_FRAGMENT_SHADER), null)
            positionHandle = GLES20.glGetAttribLocation(program, "aPosition")
            texCoordsHandle = GLES20.glGetAttribLocation(program, "aTexCoords")
            textureHandle = GLES20.glGetUniformLocation(program, "uTexture")
            stMatrixHandle = GLES20.glGetUniformLocation(program, "uSTMatrix")
            cropHandle = GLES20.glGetUniformLocation(program, "uCrop")

            directProgram = GLUtil.createAndLinkProgram(
                    GLUtil.loadShader(GLES20.GL_VERTEX_SHADER, DIRECT_VERTEX_SHADER),
                    GLUtil.loadShader(GLES20.GL_FRAGMENT_SHADER, DIRECT_FRAGMENT_SHADER), null)
            directPositionHandle = GLES20.glGetAttribLocation(directProgram, "aPosition")
            directTexCoordsHandle = GLES20.glGetAttribLocation(directProgram, "aTexCoords")
            directTextureHandle = GLES20.glGetUniformLocation(directProgram, "uTexture")
            directStMatrixHandle = GLES20.glGetUniformLocation(directProgram, "uSTMatrix")
            directMvpHandle = GLES20.glGetUniformLocation(directProgram, "uMVPMatrix")
            directAlphaHandle = GLES20.glGetUniformLocation(directProgram, "uAlpha")
            directGreyHandle = GLES20.glGetUniformLocation(directProgram, "uGrey")
            directCropHandle = GLES20.glGetUniformLocation(directProgram, "uCrop")

            copyProgram = GLUtil.createAndLinkProgram(
                    GLUtil.loadShader(GLES20.GL_VERTEX_SHADER, COPY_VERTEX_SHADER),
                    GLUtil.loadShader(GLES20.GL_FRAGMENT_SHADER, COPY_FRAGMENT_SHADER), null)
            copyPositionHandle = GLES20.glGetAttribLocation(copyProgram, "aPosition")
            copyTexCoordsHandle = GLES20.glGetAttribLocation(copyProgram, "aTexCoords")
            copyTextureHandle = GLES20.glGetUniformLocation(copyProgram, "uTexture")
            copyHalfTexelHandle = GLES20.glGetUniformLocation(copyProgram, "uHalfTexel")
        }

        /** Records the surface size so a capture pass can restore the viewport (see [GLBlur]). */
        fun setScreenSize(width: Int, height: Int) {
            screenWidth = width
            screenHeight = height
        }
    }

    private val mainHandler = Handler(Looper.getMainLooper())

    // GL objects (GL thread).
    private val externalTexture: Int
    private val surfaceTexture: SurfaceTexture
    private val surface: Surface
    private val stMatrix = FloatArray(16)
    private val sharpFbo = IntArray(1)
    private val sharpTexture = IntArray(1)
    private val sharpWidth = sharpWidth
    private val sharpHeight = sharpHeight
    // Centre-crop mapping for the external video texture: keep the middle [captureFraction] of the
    // width, so tc.x' = tc.x * cropScale + cropOffset.
    private val cropScale = captureFraction
    private val cropOffset = (1f - captureFraction) / 2f
    // The video must decode at full width so the cropped-away columns exist to sample from; only the
    // centre strip (sharpWidth) is then captured into the FBO/downsample/blur textures.
    private val decodeWidth =
            if (captureFraction < 1f) (sharpWidth / captureFraction).roundToInt() else sharpWidth
    private val blurWidth = blurWidth
    private val blurHeight = blurHeight
    // Reuses the image blur machinery: its ping-pong FBOs are sized to the downscaled blur source,
    // so pass 1 downsamples the (higher-res) sharp texture into them (see GLBlur.setExternalSource).
    private val blur = GLBlur()
    // The blur-effect GL objects (sharp FBO, downsample chain, GLBlur FBOs) are allocated on demand
    // and freed whenever blur is turned off (e.g. a lock screen with blur disabled), so a video shown
    // without effects holds only its external video texture. Toggled at draw time; see [draw],
    // [allocateEffects], [freeEffects].
    private var effectsAllocated = false
    private val quadPositions = GLUtil.asFloatBuffer(QUAD_POSITIONS)
    private val texCoords = GLUtil.asFloatBuffer(OES_TEXCOORDS)
    private val directTexCoords = GLUtil.asFloatBuffer(DIRECT_TEXCOORDS)
    private val copyTexCoords = GLUtil.asFloatBuffer(COPY_TEXCOORDS)
    // Sizes of the progressive-downsample chain from the sharp capture down to the blur source,
    // each at most a 2x reduction of the previous; the last is the blur source size.
    private val downsampleSizes: List<Pair<Int, Int>> = buildList {
        var w = sharpWidth
        var h = sharpHeight
        while (w >= blurWidth * 2 && h >= blurHeight * 2) {
            w /= 2
            h /= 2
            add(w to h)
        }
        if (isEmpty() || last() != (blurWidth to blurHeight)) {
            add(blurWidth to blurHeight)
        }
    }
    private val downsampleFbos = IntArray(downsampleSizes.size)
    private val downsampleTextures = IntArray(downsampleSizes.size)
    private var released = false
    // Whether onFirstFrame has already fired; the callback is one-shot (only the first frame gates
    // the crossfade).
    private var firstFrameSignaled = false

    private val appContext = context.applicationContext

    init {
        // External OES texture backing the SurfaceTexture.
        val tex = IntArray(1)
        GLES20.glGenTextures(1, tex, 0)
        externalTexture = tex[0]
        GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, externalTexture)
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES,
                GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_LINEAR)
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES,
                GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR)
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES,
                GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE)
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES,
                GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE)
        GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, 0)

        surfaceTexture = SurfaceTexture(externalTexture)
        // Off-screen SurfaceTextures need an explicit buffer size or some devices/codecs deliver no
        // frames to the surface (a common cause of a black video).
        surfaceTexture.setDefaultBufferSize(decodeWidth, sharpHeight)
        // One render per decoded frame, so the wallpaper renders at exactly the video's source frame
        // rate (never faster). Delivered on the main thread; it just kicks a render. The very first
        // frame also signals the renderer to begin the crossfade (see onFirstFrame).
        surfaceTexture.setOnFrameAvailableListener({
            requestRender()
            if (!firstFrameSignaled) {
                firstFrameSignaled = true
                onFirstFrame()
            }
        }, mainHandler)
        surface = Surface(surfaceTexture)

        // The sharp-capture FBO, downsample chain and GLBlur FBOs are allocated lazily the first time
        // blur is actually needed (see allocateEffects), so a blur-off video never pays for them.

        // The video is decoded by the process-wide SharedVideoPlayer, not a per-GLVideo ExoPlayer, so
        // the (up to three) engines share a single hardware decoder. Register as a user of it now; the
        // engine that is actually on screen claims its output surface via bind().
        SharedVideoPlayer.retain()
    }

    /**
     * Pulls the latest video frame into [sharpTexture] and marks the blur stale. Must run on the GL
     * thread at the start of a frame that will [draw] this video. A no-op once [released].
     */
    fun updateFrame() {
        if (released) {
            return
        }
        surfaceTexture.updateTexImage()
        surfaceTexture.getTransformMatrix(stMatrix)
    }

    /**
     * Allocates the blur-effect GL objects (sharp capture FBO, progressive-downsample chain, and the
     * GLBlur ping-pong FBOs) if not already present. Idempotent; call on the GL thread before a
     * blurred draw. Freed again by [freeEffects] when blur is turned off.
     */
    private fun allocateEffects() {
        if (effectsAllocated) {
            return
        }

        // Sharp capture FBO (ordinary 2D texture the blur/composite passes read).
        GLES20.glGenFramebuffers(1, sharpFbo, 0)
        GLES20.glGenTextures(1, sharpTexture, 0)
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, sharpTexture[0])
        GLES20.glTexImage2D(GLES20.GL_TEXTURE_2D, 0, GLES20.GL_RGBA, sharpWidth, sharpHeight, 0,
                GLES20.GL_RGBA, GLES20.GL_UNSIGNED_BYTE, null)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_LINEAR)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE)
        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, sharpFbo[0])
        GLES20.glFramebufferTexture2D(GLES20.GL_FRAMEBUFFER, GLES20.GL_COLOR_ATTACHMENT0,
                GLES20.GL_TEXTURE_2D, sharpTexture[0], 0)
        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, 0)
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, 0)

        // Progressive-downsample chain (sharp -> ... -> blur source), each an FBO + 2D texture.
        GLES20.glGenFramebuffers(downsampleFbos.size, downsampleFbos, 0)
        GLES20.glGenTextures(downsampleTextures.size, downsampleTextures, 0)
        for (i in downsampleSizes.indices) {
            val (w, h) = downsampleSizes[i]
            GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, downsampleTextures[i])
            GLES20.glTexImage2D(GLES20.GL_TEXTURE_2D, 0, GLES20.GL_RGBA, w, h, 0,
                    GLES20.GL_RGBA, GLES20.GL_UNSIGNED_BYTE, null)
            GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_LINEAR)
            GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR)
            GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE)
            GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE)
            GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, downsampleFbos[i])
            GLES20.glFramebufferTexture2D(GLES20.GL_FRAMEBUFFER, GLES20.GL_COLOR_ATTACHMENT0,
                    GLES20.GL_TEXTURE_2D, downsampleTextures[i], 0)
        }
        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, 0)
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, 0)

        // GLBlur reads the fully-downsampled (prefiltered) source, so its own passes only blur —
        // they no longer downsample, which is what aliased a small-radius blur.
        blur.setExternalSource(downsampleTextures.last(), blurWidth, blurHeight)

        effectsAllocated = true
    }

    /** Frees everything [allocateEffects] created, reclaiming the effect textures. Idempotent. */
    private fun freeEffects() {
        if (!effectsAllocated) {
            return
        }
        blur.destroy()
        GLES20.glDeleteFramebuffers(1, sharpFbo, 0)
        GLES20.glDeleteTextures(1, sharpTexture, 0)
        if (downsampleFbos.isNotEmpty()) {
            GLES20.glDeleteFramebuffers(downsampleFbos.size, downsampleFbos, 0)
            GLES20.glDeleteTextures(downsampleTextures.size, downsampleTextures, 0)
        }
        effectsAllocated = false
    }

    /**
     * Captures the current external frame into [sharpTexture] (a 2D FBO) so the blur passes can read
     * it, and marks the cached blur stale. Only needed when the frame will actually be blurred — the
     * sharp path draws the external texture straight to screen (see [draw]) and skips this. Assumes
     * the effect objects are allocated ([allocateEffects]).
     */
    private fun captureToFbo() {
        GLES20.glDisable(GLES20.GL_BLEND)
        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, sharpFbo[0])
        GLES20.glViewport(0, 0, sharpWidth, sharpHeight)
        GLES20.glUseProgram(program)
        GLES20.glEnableVertexAttribArray(positionHandle)
        GLES20.glVertexAttribPointer(positionHandle, 3, GLES20.GL_FLOAT, false, 0, quadPositions)
        GLES20.glEnableVertexAttribArray(texCoordsHandle)
        GLES20.glVertexAttribPointer(texCoordsHandle, 4, GLES20.GL_FLOAT, false, 0, texCoords)
        GLES20.glUniformMatrix4fv(stMatrixHandle, 1, false, stMatrix, 0)
        GLES20.glUniform2f(cropHandle, cropScale, cropOffset)
        GLES20.glActiveTexture(GLES20.GL_TEXTURE0)
        GLES20.glUniform1i(textureHandle, 0)
        GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, externalTexture)
        GLES20.glDrawArrays(GLES20.GL_TRIANGLES, 0, VERTICES)
        GLES20.glDisableVertexAttribArray(positionHandle)
        GLES20.glDisableVertexAttribArray(texCoordsHandle)
        GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, 0)

        // Reduce the sharp frame to the blur source in <=2x steps so GLBlur gets a prefiltered
        // source (blending stays off; these copies fully overwrite their targets).
        renderDownsampleChain()

        // Restore the default framebuffer / full-surface viewport / blending, matching the state
        // GLBlur leaves and the renderer expects for the rest of the frame.
        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, 0)
        GLES20.glViewport(0, 0, screenWidth, screenHeight)
        GLES20.glEnable(GLES20.GL_BLEND)

        // The sharp texture content changed, so any cached blur is stale.
        blur.invalidate()
    }

    /**
     * Downsamples [sharpTexture] through the [downsampleSizes] chain into [downsampleTextures], each
     * step at most a 2x GL_LINEAR reduction (a 2x2 average). The last texture — the blur source
     * GLBlur reads — is thus mip-prefiltered, so a small blur radius no longer reveals the aliasing
     * that a single large-factor reduction would leave. Assumes blending is already disabled.
     */
    private fun renderDownsampleChain() {
        GLES20.glUseProgram(copyProgram)
        GLES20.glEnableVertexAttribArray(copyPositionHandle)
        GLES20.glVertexAttribPointer(copyPositionHandle, 3, GLES20.GL_FLOAT, false, 0, quadPositions)
        GLES20.glEnableVertexAttribArray(copyTexCoordsHandle)
        GLES20.glVertexAttribPointer(copyTexCoordsHandle, 2, GLES20.GL_FLOAT, false, 0, copyTexCoords)
        GLES20.glActiveTexture(GLES20.GL_TEXTURE0)
        GLES20.glUniform1i(copyTextureHandle, 0)
        var srcTexture = sharpTexture[0]
        var srcWidth = sharpWidth
        var srcHeight = sharpHeight
        for (i in downsampleSizes.indices) {
            val (w, h) = downsampleSizes[i]
            // Half a source texel, so the four taps hit the centres of the 2x2 source block.
            GLES20.glUniform2f(copyHalfTexelHandle, 0.5f / srcWidth, 0.5f / srcHeight)
            GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, downsampleFbos[i])
            GLES20.glViewport(0, 0, w, h)
            GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, srcTexture)
            GLES20.glDrawArrays(GLES20.GL_TRIANGLES, 0, VERTICES)
            srcTexture = downsampleTextures[i]
            srcWidth = w
            srcHeight = h
        }
        GLES20.glDisableVertexAttribArray(copyPositionHandle)
        GLES20.glDisableVertexAttribArray(copyTexCoordsHandle)
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, 0)
    }

    /** Draws the external video texture straight to the bound framebuffer (the sharp path). */
    private fun drawExternalDirect(mvpMatrix: FloatArray, alpha: Float, grey: Float) {
        GLES20.glUseProgram(directProgram)
        GLES20.glEnableVertexAttribArray(directPositionHandle)
        GLES20.glVertexAttribPointer(directPositionHandle, 3, GLES20.GL_FLOAT, false, 0,
                quadPositions)
        GLES20.glEnableVertexAttribArray(directTexCoordsHandle)
        GLES20.glVertexAttribPointer(directTexCoordsHandle, 4, GLES20.GL_FLOAT, false, 0,
                directTexCoords)
        GLES20.glUniformMatrix4fv(directMvpHandle, 1, false, mvpMatrix, 0)
        GLES20.glUniformMatrix4fv(directStMatrixHandle, 1, false, stMatrix, 0)
        GLES20.glUniform2f(directCropHandle, cropScale, cropOffset)
        GLES20.glUniform1f(directAlphaHandle, alpha)
        GLES20.glUniform1f(directGreyHandle, grey)
        GLES20.glActiveTexture(GLES20.GL_TEXTURE0)
        GLES20.glUniform1i(directTextureHandle, 0)
        GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, externalTexture)
        GLES20.glDrawArrays(GLES20.GL_TRIANGLES, 0, VERTICES)
        GLES20.glDisableVertexAttribArray(directPositionHandle)
        GLES20.glDisableVertexAttribArray(directTexCoordsHandle)
        GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, 0)
    }

    /**
     * Draws the current frame through [mvpMatrix], compositing sharp and blurred layers like the
     * image path: [blurWeight] 0 -> sharp only, 1 -> blur fully covers; [grey] desaturates;
     * [blurRadiusPx] is the current blur radius. Alphas are recomposed so a crossfade
     * ([globalAlpha] < 1) blends the single lerp(sharp, blurred) result, not two stacked layers.
     *
     * When fully sharp the external texture is drawn straight to screen (one upscale, as crisp as a
     * player); only when blurring do we capture into the FBO the blur reads from.
     *
     * [effectsActive] tells whether blur is enabled at all for the current screen (independent of the
     * momentary [blurWeight], which is also 0 while focused in art-detail). The effect textures are
     * allocated while it is true and freed while it is false, so a blur-off screen (e.g. a lock
     * screen with blur disabled) reclaims them and holds only the external video texture.
     */
    fun draw(mvpMatrix: FloatArray, globalAlpha: Float, blurWeight: Float, blurRadiusPx: Float,
             grey: Float, effectsActive: Boolean) {
        if (released || globalAlpha <= 0f) {
            return
        }
        // Keep the effect textures allocated only while blur is enabled for this screen; reclaim them
        // as soon as it is turned off. Driven by the setting, not blurWeight, so briefly focusing in
        // art-detail (blurWeight 0 with blur still enabled) doesn't churn the allocation.
        if (effectsActive) {
            allocateEffects()
        } else {
            freeEffects()
        }
        if (blurWeight <= 0f || !effectsAllocated) {
            drawExternalDirect(mvpMatrix, globalAlpha, grey)
            return
        }
        // Blurring: capture the frame so GLBlur can read it, then composite sharp + blurred.
        captureToFbo()
        if (blurWeight < 1f) {
            val sharpAlpha = globalAlpha * (1f - blurWeight) / (1f - globalAlpha * blurWeight)
            blur.drawTexture(mvpMatrix, sharpTexture[0], sharpAlpha, grey)
        }
        blur.drawBlurred(mvpMatrix, blurRadiusPx, globalAlpha * blurWeight, grey)
    }

    /**
     * Claims the shared decoder's output for this video's engine: the [SharedVideoPlayer] loads this
     * video's [uri] (if not already) and renders into this [surface]. Called when this video's engine
     * becomes the on-screen one (see MuzeiBlurRenderer.updateVideoBinding). A no-op once [released].
     */
    fun bind() {
        if (released) {
            return
        }
        SharedVideoPlayer.bind(this, uri, surface, appContext)
    }

    /** Releases this video's claim on the shared decoder's output (its engine going off screen). */
    fun unbind() {
        SharedVideoPlayer.unbind(this)
    }

    /**
     * Releases the surface (on the main thread), this video's use of the shared player, and the GL
     * objects. Idempotent and safe to call from any thread: the surface release and the shared-player
     * release are posted/handled off the GL context, while the GL deletes run only if a GL context is
     * current (i.e. we're on the GL thread). When called during teardown from the main thread there's
     * no context, so the GL objects are left for the dying context to reclaim.
     */
    fun release() {
        if (released) {
            return
        }
        released = true
        // Give up the shared decoder's output (if we held it) and drop our reference so it can be
        // freed once no engine is using a video any more.
        unbind()
        SharedVideoPlayer.release()
        mainHandler.post {
            surface.release()
            surfaceTexture.release()
        }
        if (EGL14.eglGetCurrentContext() != EGL14.EGL_NO_CONTEXT) {
            freeEffects()
            GLES20.glDeleteTextures(1, intArrayOf(externalTexture), 0)
        }
    }
}
