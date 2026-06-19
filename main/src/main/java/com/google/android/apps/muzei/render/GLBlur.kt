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
    private val fbos = IntArray(2)
    private val fboTextures = IntArray(2)
    private val savedViewport = IntArray(4)
    private val savedFramebuffer = IntArray(1)
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
        // Remember the framebuffer/viewport so we can restore them after the off-screen passes.
        GLES20.glGetIntegerv(GLES20.GL_FRAMEBUFFER_BINDING, savedFramebuffer, 0)
        GLES20.glGetIntegerv(GLES20.GL_VIEWPORT, savedViewport, 0)

        // The blur passes fully overwrite their FBO, so blending is off for them.
        GLES20.glDisable(GLES20.GL_BLEND)
        GLES20.glUseProgram(blurProgram)
        GLES20.glEnableVertexAttribArray(blurPositionHandle)
        GLES20.glVertexAttribPointer(blurPositionHandle, 3, GLES20.GL_FLOAT, false, 0, quadPositions)
        GLES20.glEnableVertexAttribArray(blurTexCoordsHandle)
        GLES20.glVertexAttribPointer(blurTexCoordsHandle, 2, GLES20.GL_FLOAT, false, 0, blurTexCoords)
        GLES20.glUniform1f(blurRadiusHandle, radiusPx)
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

        // Restore the framebuffer/viewport and re-enable blending. drawBlurred() requires blending
        // for the composite pass that follows, and the renderer keeps GL_BLEND enabled for the rest
        // of the frame, so this is the correct state to leave it in (callers must invoke the blur
        // with blending enabled).
        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, savedFramebuffer[0])
        GLES20.glViewport(savedViewport[0], savedViewport[1], savedViewport[2], savedViewport[3])
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
