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

import android.graphics.Bitmap
import android.opengl.GLES20
import java.nio.FloatBuffer
import kotlin.math.exp
import kotlin.math.max

/**
 * Real-time two-pass separable Gaussian blur on the GPU. Holds a single (non-tiled) source
 * texture — the downscaled artwork — plus a ping-pong pair of framebuffer objects sized to it.
 * [drawBlurred] blurs the source at a runtime radius into the FBOs, then composites the result onto
 * the currently bound framebuffer applying an alpha and desaturation. This replaces the previous
 * approach of pre-blurring a fixed set of keyframe bitmaps on the CPU (via RenderScript).
 */
internal class GLBlur {

    companion object {
        // Upper bound on the blur radius (in source texels); the shader's loop is bounded by it.
        // MuzeiBlurRenderer sizes the blur source so the prescaled radius stays within this, so a
        // larger bound means a less-downscaled (higher quality) source. Unlike RenderScript (which
        // capped at 25 and forced the source to be downscaled further), the GPU handles the extra
        // taps cheaply.
        internal const val MAX_RADIUS = 32

        // Each shader iteration covers a pair of texels with one bilinear fetch per side (see
        // BLUR_FRAGMENT_SHADER), so the maximum number of tap-pairs is half the radius bound.
        private const val MAX_TAPS = MAX_RADIUS / 2

        // Separable blur pass: samples 2*radius+1 taps along uStep, weighted by a Gaussian.
        private const val BLUR_VERTEX_SHADER = "" +
                "attribute vec4 aPosition;" +
                "attribute vec2 aTexCoords;" +
                "varying vec2 vTexCoords;" +
                "void main(){" +
                "  vTexCoords = aTexCoords;" +
                "  gl_Position = aPosition;" +
                "}"

        private val BLUR_FRAGMENT_SHADER = "" +
                "precision mediump float;" +
                "uniform sampler2D uTexture;" +
                "uniform vec2 uStep;" +     // texel step along the blur axis (1/size, 0) or (0, 1/size)
                "uniform int uTapCount;" +  // number of active tap-pairs (<= MAX_TAPS)
                "uniform float uCenterWeight;" +              // normalized weight of the center tap
                "uniform float uWeights[" + MAX_TAPS + "];" + // normalized weight per tap-pair
                "uniform float uOffsets[" + MAX_TAPS + "];" + // centroid distance (in texels) per pair
                "varying vec2 vTexCoords;" +
                "void main(){" +
                // Linear-sampling Gaussian: each iteration covers a pair of taps with a single
                // bilinear fetch per side, placed at the weight-centroid between the two texels so
                // the hardware interpolation returns the exact pair sum. This halves the texture
                // fetches. The Gaussian weights and centroid offsets depend only on the radius (a
                // per-pass constant), so they're precomputed on the CPU (see uploadWeights) and
                // arrive already normalized — the shader does only fetches and multiply-adds, with
                // no per-fragment exp() or divide.
                "  vec4 sum = texture2D(uTexture, vTexCoords) * uCenterWeight;" +
                "  for (int k = 0; k < " + MAX_TAPS + "; k++) {" +
                "    if (k >= uTapCount) break;" +
                "    vec2 off = uStep * uOffsets[k];" +
                "    sum += (texture2D(uTexture, vTexCoords + off)" +
                "          + texture2D(uTexture, vTexCoords - off)) * uWeights[k];" +
                "  }" +
                "  gl_FragColor = sum;" +
                "}"

        // Composite pass: draws the blurred texture through the MVP, applying alpha and desaturation.
        private const val COMPOSITE_VERTEX_SHADER = "" +
                "uniform mat4 uMVPMatrix;" +
                "attribute vec4 aPosition;" +
                "attribute vec2 aTexCoords;" +
                "varying vec2 vTexCoords;" +
                "void main(){" +
                "  vTexCoords = aTexCoords;" +
                "  gl_Position = uMVPMatrix * aPosition;" +
                "}"

        private const val COMPOSITE_FRAGMENT_SHADER = "" +
                "precision mediump float;" +
                "uniform sampler2D uTexture;" +
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

        // Full-screen quad (NDC), vertex order TL, BL, BR, TL, BR, TR.
        private val QUAD_POSITIONS = floatArrayOf(
                -1f, 1f, 0f,   -1f, -1f, 0f,   1f, -1f, 0f,
                -1f, 1f, 0f,    1f, -1f, 0f,   1f, 1f, 0f)
        // Identity texcoords ((pos + 1) / 2) for the blur passes. Mapping the source coordinate
        // straight to the framebuffer position keeps the FBO result in the same orientation as the
        // source texture (no net vertical flip across the two passes).
        private val BLUR_TEXCOORDS = floatArrayOf(
                0f, 1f,   0f, 0f,   1f, 0f,
                0f, 1f,   1f, 0f,   1f, 1f)
        // Composite samples the blurred FBO with the same convention GLPicture uses
        // (texcoord (0,0) == top-left), so it draws upright through the MVP.
        private val COMPOSITE_TEXCOORDS = floatArrayOf(
                0f, 0f,   0f, 1f,   1f, 1f,
                0f, 0f,   1f, 1f,   1f, 0f)
        private const val VERTICES = 6

        private var blurProgram = 0
        private var blurPositionHandle = 0
        private var blurTexCoordsHandle = 0
        private var blurTextureHandle = 0
        private var blurStepHandle = 0
        private var blurTapCountHandle = 0
        private var blurCenterWeightHandle = 0
        private var blurWeightsHandle = 0
        private var blurOffsetsHandle = 0

        // Scratch buffers for the CPU-side Gaussian weights/offsets, reused across reblurs. The
        // blur only ever runs on the GL thread, so a single shared pair is safe.
        private val weights = FloatArray(MAX_TAPS)
        private val offsets = FloatArray(MAX_TAPS)

        private var compositeProgram = 0
        private var compositePositionHandle = 0
        private var compositeTexCoordsHandle = 0
        private var compositeTextureHandle = 0
        private var compositeMvpHandle = 0
        private var compositeAlphaHandle = 0
        private var compositeGreyHandle = 0

        private lateinit var quadPositions: FloatBuffer
        private lateinit var blurTexCoords: FloatBuffer
        private lateinit var compositeTexCoords: FloatBuffer

        fun initGl() {
            blurProgram = GLUtil.createAndLinkProgram(
                    GLUtil.loadShader(GLES20.GL_VERTEX_SHADER, BLUR_VERTEX_SHADER),
                    GLUtil.loadShader(GLES20.GL_FRAGMENT_SHADER, BLUR_FRAGMENT_SHADER), null)
            blurPositionHandle = GLES20.glGetAttribLocation(blurProgram, "aPosition")
            blurTexCoordsHandle = GLES20.glGetAttribLocation(blurProgram, "aTexCoords")
            blurTextureHandle = GLES20.glGetUniformLocation(blurProgram, "uTexture")
            blurStepHandle = GLES20.glGetUniformLocation(blurProgram, "uStep")
            blurTapCountHandle = GLES20.glGetUniformLocation(blurProgram, "uTapCount")
            blurCenterWeightHandle = GLES20.glGetUniformLocation(blurProgram, "uCenterWeight")
            blurWeightsHandle = GLES20.glGetUniformLocation(blurProgram, "uWeights")
            blurOffsetsHandle = GLES20.glGetUniformLocation(blurProgram, "uOffsets")

            compositeProgram = GLUtil.createAndLinkProgram(
                    GLUtil.loadShader(GLES20.GL_VERTEX_SHADER, COMPOSITE_VERTEX_SHADER),
                    GLUtil.loadShader(GLES20.GL_FRAGMENT_SHADER, COMPOSITE_FRAGMENT_SHADER), null)
            compositePositionHandle = GLES20.glGetAttribLocation(compositeProgram, "aPosition")
            compositeTexCoordsHandle = GLES20.glGetAttribLocation(compositeProgram, "aTexCoords")
            compositeTextureHandle = GLES20.glGetUniformLocation(compositeProgram, "uTexture")
            compositeMvpHandle = GLES20.glGetUniformLocation(compositeProgram, "uMVPMatrix")
            compositeAlphaHandle = GLES20.glGetUniformLocation(compositeProgram, "uAlpha")
            compositeGreyHandle = GLES20.glGetUniformLocation(compositeProgram, "uGrey")

            quadPositions = GLUtil.asFloatBuffer(QUAD_POSITIONS)
            blurTexCoords = GLUtil.asFloatBuffer(BLUR_TEXCOORDS)
            compositeTexCoords = GLUtil.asFloatBuffer(COMPOSITE_TEXCOORDS)
        }

        // Surface dimensions, so runBlurPasses can restore the viewport after its off-screen passes
        // without a glGetIntegerv round-trip. The blur is only ever invoked while the renderer is
        // drawing to the full-surface default framebuffer, so this is always the viewport to restore.
        private var screenWidth = 0
        private var screenHeight = 0

        /** Records the surface size (call from the renderer's onSurfaceChanged). */
        fun setScreenSize(width: Int, height: Int) {
            screenWidth = width
            screenHeight = height
        }

        /**
         * Computes the linear-sampling Gaussian weights and centroid offsets for [radiusPx] into the
         * shared [weights]/[offsets] scratch buffers and uploads them (plus the tap count and center
         * weight) to the currently bound blur program. All values are normalized so the shader needs
         * no final divide. Returns nothing; call once per reblur before the two passes.
         *
         * This mirrors the Gaussian the fragment shader used to evaluate per-fragment, but since the
         * weights depend only on the radius (constant across the pass) it's hoisted here to the CPU,
         * eliminating ~MAX_TAPS exp() calls and a divide for every fragment of both passes.
         */
        private fun uploadWeights(radiusPx: Float) {
            val s = max(radiusPx, 0.0001f) * 0.5f
            val twoSigmaSq = 2f * s * s
            var wsum = 1f // center tap, pre-normalization weight 1
            var tapCount = 0
            // Each iteration pairs taps i1=2k-1 and i2=2k. i1 <= radiusPx keeps w1 >= exp(-2) > 0, so
            // the centroid divide below is always safe.
            var k = 1
            while (k <= MAX_TAPS) {
                val i1 = (2 * k - 1).toFloat()
                if (i1 > radiusPx) break
                val i2 = (2 * k).toFloat()
                val w1 = exp(-i1 * i1 / twoSigmaSq)
                val w2 = if (i2 > radiusPx) 0f else exp(-i2 * i2 / twoSigmaSq)
                val cw = w1 + w2
                weights[tapCount] = cw
                offsets[tapCount] = (i1 * w1 + i2 * w2) / cw
                wsum += 2f * cw
                tapCount++
                k++
            }
            // Normalize so the shader's weighted sum already integrates to 1.
            val invWsum = 1f / wsum
            for (j in 0 until tapCount) {
                weights[j] *= invWsum
            }
            GLES20.glUniform1i(blurTapCountHandle, tapCount)
            GLES20.glUniform1f(blurCenterWeightHandle, invWsum)
            if (tapCount > 0) {
                GLES20.glUniform1fv(blurWeightsHandle, tapCount, weights, 0)
                GLES20.glUniform1fv(blurOffsetsHandle, tapCount, offsets, 0)
            }
        }
    }

    private var sourceTexture = 0
    private var width = 0
    private var height = 0
    private val fbos = IntArray(2)
    private val fboTextures = IntArray(2)
    private var ready = false
    // Radius the cached fbo[1] was last blurred at; -1 forces a (re)blur on the next draw.
    private var lastBlurRadius = -1f

    /** Uploads [bitmap] as the blur source and (re)allocates the ping-pong FBOs sized to it. */
    fun setSource(bitmap: Bitmap) {
        destroy()
        lastBlurRadius = -1f // new source: the cached blur (if any) is stale
        width = bitmap.width
        height = bitmap.height
        if (width == 0 || height == 0) {
            return
        }
        sourceTexture = GLUtil.loadTexture(bitmap)

        GLES20.glGenFramebuffers(2, fbos, 0)
        GLES20.glGenTextures(2, fboTextures, 0)
        for (i in 0..1) {
            GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, fboTextures[i])
            GLES20.glTexImage2D(GLES20.GL_TEXTURE_2D, 0, GLES20.GL_RGBA, width, height, 0,
                    GLES20.GL_RGBA, GLES20.GL_UNSIGNED_BYTE, null)
            GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_S,
                    GLES20.GL_CLAMP_TO_EDGE)
            GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_T,
                    GLES20.GL_CLAMP_TO_EDGE)
            GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MIN_FILTER,
                    GLES20.GL_LINEAR)
            GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MAG_FILTER,
                    GLES20.GL_LINEAR)
            GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, fbos[i])
            GLES20.glFramebufferTexture2D(GLES20.GL_FRAMEBUFFER, GLES20.GL_COLOR_ATTACHMENT0,
                    GLES20.GL_TEXTURE_2D, fboTextures[i], 0)
        }
        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, 0)
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, 0)
        ready = true
    }

    /**
     * Blurs the source at [radiusPx] and draws the result through [mvpMatrix] onto the bound
     * framebuffer, blended in at [alpha] and desaturated by [grey] (0..1). A [radiusPx] of 0 is a
     * straight (optionally desaturated) copy of the source.
     *
     * The blur itself only re-runs when [radiusPx] changes, so panning the home screen (which only
     * moves the MVP) just re-composites the cached blurred texture rather than blurring every frame.
     */
    fun drawBlurred(mvpMatrix: FloatArray, radiusPx: Float, alpha: Float, grey: Float) {
        if (!ready || alpha <= 0f) {
            return
        }

        if (radiusPx != lastBlurRadius) {
            runBlurPasses(radiusPx)
            lastBlurRadius = radiusPx
        }

        // Composite the cached blurred result (fbo[1]) onto the bound framebuffer.
        GLES20.glUseProgram(compositeProgram)
        GLES20.glEnableVertexAttribArray(compositePositionHandle)
        GLES20.glVertexAttribPointer(compositePositionHandle, 3, GLES20.GL_FLOAT, false, 0,
                quadPositions)
        GLES20.glEnableVertexAttribArray(compositeTexCoordsHandle)
        GLES20.glVertexAttribPointer(compositeTexCoordsHandle, 2, GLES20.GL_FLOAT, false, 0,
                compositeTexCoords)
        GLES20.glUniformMatrix4fv(compositeMvpHandle, 1, false, mvpMatrix, 0)
        GLES20.glUniform1f(compositeAlphaHandle, alpha)
        GLES20.glUniform1f(compositeGreyHandle, grey)
        GLES20.glActiveTexture(GLES20.GL_TEXTURE0)
        GLES20.glUniform1i(compositeTextureHandle, 0)
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, fboTextures[1])
        GLES20.glDrawArrays(GLES20.GL_TRIANGLES, 0, VERTICES)

        GLES20.glDisableVertexAttribArray(compositePositionHandle)
        GLES20.glDisableVertexAttribArray(compositeTexCoordsHandle)
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, 0)
    }

    /** Runs the two separable blur passes (source -> fbo[0] -> fbo[1]) into the cached FBOs. */
    private fun runBlurPasses(radiusPx: Float) {
        // The blur passes fully overwrite their FBO, so blending is off for them.
        GLES20.glDisable(GLES20.GL_BLEND)
        GLES20.glUseProgram(blurProgram)
        GLES20.glEnableVertexAttribArray(blurPositionHandle)
        GLES20.glVertexAttribPointer(blurPositionHandle, 3, GLES20.GL_FLOAT, false, 0, quadPositions)
        GLES20.glEnableVertexAttribArray(blurTexCoordsHandle)
        GLES20.glVertexAttribPointer(blurTexCoordsHandle, 2, GLES20.GL_FLOAT, false, 0, blurTexCoords)
        // The Gaussian weights/offsets are the same for both passes (only uStep differs), so compute
        // and upload them once here rather than per-fragment in the shader.
        uploadWeights(radiusPx)
        GLES20.glActiveTexture(GLES20.GL_TEXTURE0)
        GLES20.glUniform1i(blurTextureHandle, 0)
        GLES20.glViewport(0, 0, width, height)

        // Pass 1: horizontal, source -> fbo[0]
        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, fbos[0])
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, sourceTexture)
        GLES20.glUniform2f(blurStepHandle, 1f / width, 0f)
        GLES20.glDrawArrays(GLES20.GL_TRIANGLES, 0, VERTICES)

        // Pass 2: vertical, fbo[0] -> fbo[1]
        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, fbos[1])
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, fboTextures[0])
        GLES20.glUniform2f(blurStepHandle, 0f, 1f / height)
        GLES20.glDrawArrays(GLES20.GL_TRIANGLES, 0, VERTICES)

        GLES20.glDisableVertexAttribArray(blurPositionHandle)
        GLES20.glDisableVertexAttribArray(blurTexCoordsHandle)

        // Restore the default framebuffer and full-surface viewport, and re-enable blending. The
        // blur is only ever invoked while the renderer draws to the default framebuffer at the full
        // surface size (see setScreenSize), so these are the values to restore — no glGetIntegerv
        // round-trip needed. drawBlurred() requires blending for the composite pass that follows,
        // and the renderer keeps GL_BLEND enabled for the rest of the frame, so this is the correct
        // state to leave it in (callers must invoke the blur with blending enabled).
        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, 0)
        GLES20.glViewport(0, 0, screenWidth, screenHeight)
        GLES20.glEnable(GLES20.GL_BLEND)
    }

    fun destroy() {
        if (sourceTexture != 0) {
            GLES20.glDeleteTextures(1, intArrayOf(sourceTexture), 0)
            sourceTexture = 0
        }
        if (ready) {
            GLES20.glDeleteFramebuffers(2, fbos, 0)
            GLES20.glDeleteTextures(2, fboTextures, 0)
            ready = false
        }
    }
}
