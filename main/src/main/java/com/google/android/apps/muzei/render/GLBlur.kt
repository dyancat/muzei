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

/**
 * GPU blur of a downscaled artwork source. The two-pass separable Gaussian blur is run once per
 * distinct radius into a cached result texture; the home- and lock-screen blur levels are
 * pre-blurred up front in [setSource], so transitions between them (and the focus animation) are
 * pure texture crossfades with no per-frame blur work — important on the always-on display, where
 * the device is in a low-power state. Lazily blurs any other radius on demand (e.g. a settings
 * change) and caches it.
 */
internal class GLBlur {

    companion object {
        // Upper bound on the blur radius (in source texels); the shader's loop is bounded by it.
        // MuzeiBlurRenderer sizes the blur source so the prescaled radius stays within this, so a
        // larger bound means a less-downscaled (higher quality) source. Unlike RenderScript (which
        // capped at 25 and forced the source to be downscaled further), the GPU handles the extra
        // taps cheaply.
        internal const val MAX_RADIUS = 32

        // Result textures retained at once. Steady state needs the home + lock levels; a couple of
        // spare slots absorb a settings change without immediately dropping a level still in use.
        private const val MAX_CACHED_LEVELS = 4

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
                "uniform float uRadius;" +  // blur radius in texels
                "varying vec2 vTexCoords;" +
                "void main(){" +
                "  float sigma = max(uRadius, 0.0001) * 0.5;" +
                "  vec4 sum = texture2D(uTexture, vTexCoords);" +
                "  float wsum = 1.0;" +
                "  for (int i = 1; i <= " + MAX_RADIUS + "; i++) {" +
                "    if (float(i) > uRadius) break;" +
                "    float w = exp(-float(i) * float(i) / (2.0 * sigma * sigma));" +
                "    vec2 off = uStep * float(i);" +
                "    sum += texture2D(uTexture, vTexCoords + off) * w;" +
                "    sum += texture2D(uTexture, vTexCoords - off) * w;" +
                "    wsum += 2.0 * w;" +
                "  }" +
                "  gl_FragColor = sum / wsum;" +
                "}"

        // Composite pass: draws a blurred texture through the MVP, applying alpha and desaturation.
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
        // Composite samples the blurred texture with the same convention GLPicture uses
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
        private var blurRadiusHandle = 0

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
            blurRadiusHandle = GLES20.glGetUniformLocation(blurProgram, "uRadius")

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
    }

    private var sourceTexture = 0
    private var width = 0
    private var height = 0
    private val scratchFbo = IntArray(1)      // H-pass render target
    private val scratchTexture = IntArray(1)
    private val resultFbo = IntArray(1)       // reused; the result texture is attached per blur
    private val savedViewport = IntArray(4)
    private val savedFramebuffer = IntArray(1)
    private var ready = false
    // Blurred result textures keyed by radius (parallel arrays, scanned linearly — only a handful).
    private val cachedRadius = FloatArray(MAX_CACHED_LEVELS)
    private val cachedTexture = IntArray(MAX_CACHED_LEVELS)
    private var cachedCount = 0

    /** Uploads [bitmap] as the blur source and pre-blurs it at each of [radii] (e.g. home + lock). */
    fun setSource(bitmap: Bitmap, vararg radii: Float) {
        destroy()
        width = bitmap.width
        height = bitmap.height
        if (width == 0 || height == 0) {
            return
        }
        sourceTexture = GLUtil.loadTexture(bitmap)

        GLES20.glGenFramebuffers(1, scratchFbo, 0)
        GLES20.glGenFramebuffers(1, resultFbo, 0)
        GLES20.glGenTextures(1, scratchTexture, 0)
        allocateColorTexture(scratchTexture[0])
        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, scratchFbo[0])
        GLES20.glFramebufferTexture2D(GLES20.GL_FRAMEBUFFER, GLES20.GL_COLOR_ATTACHMENT0,
                GLES20.GL_TEXTURE_2D, scratchTexture[0], 0)
        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, 0)
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, 0)
        ready = true

        // Pre-blur the requested levels so later transitions are pure crossfades.
        for (radius in radii) {
            textureForRadius(radius)
        }
    }

    /**
     * Composites a crossfade between the [fromRadius] and [toRadius] blur levels (by [fraction],
     * 0 -> from, 1 -> to) onto the bound framebuffer at [alpha], desaturated by [grey]. Both levels
     * are normally already pre-blurred, so this is just one or two textured-quad draws.
     */
    fun crossfade(mvpMatrix: FloatArray, fromRadius: Float, toRadius: Float,
                  fraction: Float, alpha: Float, grey: Float) {
        if (!ready || alpha <= 0f) {
            return
        }
        val lf = fraction.coerceIn(0f, 1f)
        // Recompose the two layers' alphas so the visible result is alpha * lerp(from, to, lf) over
        // whatever is below, rather than the "to" layer merely painted over the "from" layer.
        if (lf < 1f) {
            val fromAlpha = alpha * (1f - lf) / (1f - alpha * lf)
            compositeTexture(mvpMatrix, textureForRadius(fromRadius), fromAlpha, grey)
        }
        if (lf > 0f) {
            compositeTexture(mvpMatrix, textureForRadius(toRadius), alpha * lf, grey)
        }
    }

    private fun compositeTexture(mvpMatrix: FloatArray, texture: Int, alpha: Float, grey: Float) {
        if (alpha <= 0f) {
            return
        }
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
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, texture)
        GLES20.glDrawArrays(GLES20.GL_TRIANGLES, 0, VERTICES)
        GLES20.glDisableVertexAttribArray(compositePositionHandle)
        GLES20.glDisableVertexAttribArray(compositeTexCoordsHandle)
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, 0)
    }

    /** Returns the cached blurred texture for [radius], blurring (and caching) it on a miss. */
    private fun textureForRadius(radius: Float): Int {
        for (i in 0 until cachedCount) {
            if (cachedRadius[i] == radius) {
                return cachedTexture[i]
            }
        }
        val result = IntArray(1)
        GLES20.glGenTextures(1, result, 0)
        allocateColorTexture(result[0])
        runBlurPasses(radius, result[0])
        if (cachedCount == MAX_CACHED_LEVELS) {
            // Drop the oldest level.
            GLES20.glDeleteTextures(1, cachedTexture, 0)
            for (i in 1 until cachedCount) {
                cachedRadius[i - 1] = cachedRadius[i]
                cachedTexture[i - 1] = cachedTexture[i]
            }
            cachedCount--
        }
        cachedRadius[cachedCount] = radius
        cachedTexture[cachedCount] = result[0]
        cachedCount++
        return result[0]
    }

    /** Runs the two separable blur passes (source -> scratch -> [resultTexture]). */
    private fun runBlurPasses(radius: Float, resultTexture: Int) {
        // Remember the framebuffer/viewport so we can restore them after the off-screen passes.
        GLES20.glGetIntegerv(GLES20.GL_FRAMEBUFFER_BINDING, savedFramebuffer, 0)
        GLES20.glGetIntegerv(GLES20.GL_VIEWPORT, savedViewport, 0)

        // The blur passes fully overwrite their target, so blending is off for them.
        GLES20.glDisable(GLES20.GL_BLEND)
        GLES20.glUseProgram(blurProgram)
        GLES20.glEnableVertexAttribArray(blurPositionHandle)
        GLES20.glVertexAttribPointer(blurPositionHandle, 3, GLES20.GL_FLOAT, false, 0, quadPositions)
        GLES20.glEnableVertexAttribArray(blurTexCoordsHandle)
        GLES20.glVertexAttribPointer(blurTexCoordsHandle, 2, GLES20.GL_FLOAT, false, 0, blurTexCoords)
        GLES20.glUniform1f(blurRadiusHandle, radius)
        GLES20.glActiveTexture(GLES20.GL_TEXTURE0)
        GLES20.glUniform1i(blurTextureHandle, 0)
        GLES20.glViewport(0, 0, width, height)

        // Pass 1: horizontal, source -> scratch
        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, scratchFbo[0])
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, sourceTexture)
        GLES20.glUniform2f(blurStepHandle, 1f / width, 0f)
        GLES20.glDrawArrays(GLES20.GL_TRIANGLES, 0, VERTICES)

        // Pass 2: vertical, scratch -> resultTexture
        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, resultFbo[0])
        GLES20.glFramebufferTexture2D(GLES20.GL_FRAMEBUFFER, GLES20.GL_COLOR_ATTACHMENT0,
                GLES20.GL_TEXTURE_2D, resultTexture, 0)
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, scratchTexture[0])
        GLES20.glUniform2f(blurStepHandle, 0f, 1f / height)
        GLES20.glDrawArrays(GLES20.GL_TRIANGLES, 0, VERTICES)

        GLES20.glDisableVertexAttribArray(blurPositionHandle)
        GLES20.glDisableVertexAttribArray(blurTexCoordsHandle)

        // Restore the framebuffer/viewport and re-enable blending for the rest of the frame.
        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, savedFramebuffer[0])
        GLES20.glViewport(savedViewport[0], savedViewport[1], savedViewport[2], savedViewport[3])
        GLES20.glEnable(GLES20.GL_BLEND)
    }

    private fun allocateColorTexture(texture: Int) {
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, texture)
        GLES20.glTexImage2D(GLES20.GL_TEXTURE_2D, 0, GLES20.GL_RGBA, width, height, 0,
                GLES20.GL_RGBA, GLES20.GL_UNSIGNED_BYTE, null)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_LINEAR)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR)
    }

    fun destroy() {
        if (sourceTexture != 0) {
            GLES20.glDeleteTextures(1, intArrayOf(sourceTexture), 0)
            sourceTexture = 0
        }
        if (cachedCount > 0) {
            GLES20.glDeleteTextures(cachedCount, cachedTexture, 0)
            cachedCount = 0
        }
        if (ready) {
            GLES20.glDeleteTextures(1, scratchTexture, 0)
            GLES20.glDeleteFramebuffers(1, scratchFbo, 0)
            GLES20.glDeleteFramebuffers(1, resultFbo, 0)
            ready = false
        }
    }
}
