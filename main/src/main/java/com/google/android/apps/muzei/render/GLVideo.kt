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
import android.util.Log
import android.view.Surface
import androidx.media3.common.MediaItem
import androidx.media3.common.PlaybackException
import androidx.media3.common.Player
import androidx.media3.exoplayer.ExoPlayer

/**
 * Plays a video as artwork, mirroring how [GLBlur] renders still images: each frame the live video
 * frame (an external `GL_TEXTURE_EXTERNAL_OES` fed by an [ExoPlayer] through a [SurfaceTexture]) is
 * captured into an ordinary 2D FBO — the "sharp" texture — which is then drawn and blurred by the
 * same [GLBlur] composite/blur passes the image path uses. So blur / grey / dim / pan-zoom /
 * crossfade all apply to video exactly as they do to images.
 *
 * Threading: the GL objects (external texture, [SurfaceTexture], FBO, [GLBlur]) are created and used
 * on the GL thread. [ExoPlayer] lives on the main thread (it is not thread-safe); [pause]/[resume]/
 * [release] hop there. New frames arrive via [SurfaceTexture.setOnFrameAvailableListener], which
 * simply asks the renderer to draw again (the renderer is `RENDERMODE_WHEN_DIRTY`), so a paused
 * player produces no frames and therefore no redraws — the battery win on AOD / when hidden.
 */
internal class GLVideo(
        context: Context,
        private val uri: Uri,
        sharpWidth: Int,
        sharpHeight: Int,
        blurWidth: Int,
        blurHeight: Int,
        private val requestRender: () -> Unit
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
                "varying vec2 vTexCoords;" +
                "void main(){" +
                "  gl_Position = aPosition;" +
                "  vTexCoords = (uSTMatrix * aTexCoords).xy;" +
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
                "attribute vec4 aPosition;" +
                "attribute vec4 aTexCoords;" +
                "varying vec2 vTexCoords;" +
                "void main(){" +
                "  gl_Position = uMVPMatrix * aPosition;" +
                "  vTexCoords = (uSTMatrix * aTexCoords).xy;" +
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

        // Full-screen quad (NDC), TL, BL, BR, TL, BR, TR.
        private val QUAD_POSITIONS = floatArrayOf(
                -1f, 1f, 0f,   -1f, -1f, 0f,   1f, -1f, 0f,
                -1f, 1f, 0f,    1f, -1f, 0f,   1f, 1f, 0f)
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

        private var directProgram = 0
        private var directPositionHandle = 0
        private var directTexCoordsHandle = 0
        private var directTextureHandle = 0
        private var directStMatrixHandle = 0
        private var directMvpHandle = 0
        private var directAlphaHandle = 0
        private var directGreyHandle = 0

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
    // Reuses the image blur machinery: its ping-pong FBOs are sized to the downscaled blur source,
    // so pass 1 downsamples the (higher-res) sharp texture into them (see GLBlur.setExternalSource).
    private val blur = GLBlur()
    private val quadPositions = GLUtil.asFloatBuffer(QUAD_POSITIONS)
    private val texCoords = GLUtil.asFloatBuffer(OES_TEXCOORDS)
    private val directTexCoords = GLUtil.asFloatBuffer(DIRECT_TEXCOORDS)
    private var released = false

    // ExoPlayer (main thread).
    @Volatile private var player: ExoPlayer? = null
    private var playWhenReady = true

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
        surfaceTexture.setDefaultBufferSize(sharpWidth, sharpHeight)
        // One render per decoded frame, so the wallpaper renders at exactly the video's source frame
        // rate (never faster). Delivered on the main thread; it just kicks a render.
        surfaceTexture.setOnFrameAvailableListener({ requestRender() }, mainHandler)
        surface = Surface(surfaceTexture)

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

        // GLBlur reads the sharp texture as its source, downscaling into its blur FBOs.
        blur.setExternalSource(sharpTexture[0], blurWidth, blurHeight)

        // Create and start the player on the main thread.
        val appContext = context.applicationContext
        mainHandler.post {
            if (released) {
                return@post
            }
            val exo = ExoPlayer.Builder(appContext).build().apply {
                setVideoSurface(surface)
                repeatMode = Player.REPEAT_MODE_ONE // seamless loop
                volume = 0f // wallpapers are silent
                addListener(object : Player.Listener {
                    override fun onPlayerError(error: PlaybackException) {
                        Log.e(TAG, "Error playing video $uri: ${error.errorCodeName}", error)
                    }
                })
                setMediaItem(MediaItem.fromUri(uri))
                prepare()
                this.playWhenReady = this@GLVideo.playWhenReady
            }
            player = exo
        }
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
     * Captures the current external frame into [sharpTexture] (a 2D FBO) so the blur passes can read
     * it, and marks the cached blur stale. Only needed when the frame will actually be blurred — the
     * sharp path draws the external texture straight to screen (see [draw]) and skips this.
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
        GLES20.glActiveTexture(GLES20.GL_TEXTURE0)
        GLES20.glUniform1i(textureHandle, 0)
        GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, externalTexture)
        GLES20.glDrawArrays(GLES20.GL_TRIANGLES, 0, VERTICES)
        GLES20.glDisableVertexAttribArray(positionHandle)
        GLES20.glDisableVertexAttribArray(texCoordsHandle)
        GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, 0)

        // Restore the default framebuffer / full-surface viewport / blending, matching the state
        // GLBlur leaves and the renderer expects for the rest of the frame.
        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, 0)
        GLES20.glViewport(0, 0, screenWidth, screenHeight)
        GLES20.glEnable(GLES20.GL_BLEND)

        // The sharp texture content changed, so any cached blur is stale.
        blur.invalidate()
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
     */
    fun draw(mvpMatrix: FloatArray, globalAlpha: Float, blurWeight: Float, blurRadiusPx: Float,
             grey: Float) {
        if (released || globalAlpha <= 0f) {
            return
        }
        if (blurWeight <= 0f) {
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

    fun pause() {
        mainHandler.post { player?.playWhenReady = false }
        playWhenReady = false
    }

    fun resume() {
        mainHandler.post { player?.playWhenReady = true }
        playWhenReady = true
    }

    /**
     * Releases the player + surface (on the main thread) and the GL objects. Idempotent and safe to
     * call from any thread: the player release is always posted to the main thread (freeing the
     * codec — the critical resource), while the GL deletes run only if a GL context is current (i.e.
     * we're on the GL thread). When called during teardown from the main thread there's no context,
     * so the GL objects are left for the dying context to reclaim.
     */
    fun release() {
        if (released) {
            return
        }
        released = true
        val exo = player
        player = null
        mainHandler.post {
            exo?.release()
            surface.release()
            surfaceTexture.release()
        }
        if (EGL14.eglGetCurrentContext() != EGL14.EGL_NO_CONTEXT) {
            blur.destroy()
            GLES20.glDeleteFramebuffers(1, sharpFbo, 0)
            GLES20.glDeleteTextures(1, sharpTexture, 0)
            GLES20.glDeleteTextures(1, intArrayOf(externalTexture), 0)
        }
    }
}
