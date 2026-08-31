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

import android.content.Context
import android.content.SharedPreferences
import androidx.lifecycle.DefaultLifecycleObserver
import androidx.lifecycle.LifecycleOwner
import androidx.lifecycle.lifecycleScope
import com.google.android.apps.muzei.settings.Prefs
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch

sealed class ReloadType
data object ReloadWhenVisible : ReloadType()
data object ReloadDespiteInvisible : ReloadType()
data object ReloadImmediate : ReloadType()

abstract class RenderController(
        protected var context: Context,
        protected var renderer: MuzeiBlurRenderer,
        private var callbacks: Callbacks
) : DefaultLifecycleObserver {

    var visible: Boolean = false
        set(value) {
            field = value
            if (value) {
                callbacks.queueEventOnGlThread {
                    val source = queuedSource
                    if (source != null) {
                        queuedSource = null
                        renderer.setAndConsumeSource(source)
                    }
                }
                callbacks.requestRender()
            }
        }
    var onLockScreen: Boolean = false
        set(value) {
            if (field != value) {
                field = value
                renderer.recomputeMaxPrescaledBlurPixels(
                        if (value) Prefs.PREF_LOCK_BLUR_AMOUNT else Prefs.PREF_BLUR_AMOUNT)
                renderer.recomputeMaxDimAmount(
                        if (value) Prefs.PREF_LOCK_DIM_AMOUNT else Prefs.PREF_DIM_AMOUNT)
                renderer.recomputeGreyAmount(
                        if (value) Prefs.PREF_LOCK_GREY_AMOUNT else Prefs.PREF_GREY_AMOUNT)
                // The GPU blur reads the blur/dim/grey amounts live and onDrawFrame eases the
                // effective values toward the new targets, so the home<->lock transition animates
                // without re-decoding the artwork. Just kick a render to start the easing; it keeps
                // running in the background (see MuzeiWallpaperEngine.onVisibilityChanged).
                callbacks.requestRender()
            }
        }
    private lateinit var coroutineScope: CoroutineScope
    private var destroyed = false
    private var queuedSource: RenderSource? = null
    private val sharedPreferenceChangeListener = SharedPreferences.OnSharedPreferenceChangeListener { _, key ->
        if (onLockScreen) {
            when (key) {
                Prefs.PREF_LOCK_BLUR_AMOUNT -> {
                    renderer.recomputeMaxPrescaledBlurPixels()
                    callbacks.requestRender()
                }
                Prefs.PREF_LOCK_DIM_AMOUNT -> {
                    renderer.recomputeMaxDimAmount()
                    callbacks.requestRender()
                }
                Prefs.PREF_LOCK_GREY_AMOUNT -> {
                    renderer.recomputeGreyAmount()
                    callbacks.requestRender()
                }
            }
        } else {
            when (key) {
                Prefs.PREF_BLUR_AMOUNT -> {
                    renderer.recomputeMaxPrescaledBlurPixels()
                    callbacks.requestRender()
                }
                Prefs.PREF_DIM_AMOUNT -> {
                    renderer.recomputeMaxDimAmount()
                    callbacks.requestRender()
                }
                Prefs.PREF_GREY_AMOUNT -> {
                    renderer.recomputeGreyAmount()
                    callbacks.requestRender()
                }
            }
        }
    }

    override fun onCreate(owner: LifecycleOwner) {
        coroutineScope = owner.lifecycleScope
        Prefs.getSharedPreferences(context)
                .registerOnSharedPreferenceChangeListener(sharedPreferenceChangeListener)
    }

    override fun onDestroy(owner: LifecycleOwner) {
        queuedSource = null
        Prefs.getSharedPreferences(context)
                .unregisterOnSharedPreferenceChangeListener(sharedPreferenceChangeListener)
        destroyed = true
    }

    /**
     * Records whether this engine's surface is on screen (on the GL thread). Only the on-screen
     * engine claims the shared video decoder's output, so a hidden engine's video freezes and stops
     * redrawing (a no-op for image artwork). Screen on/off is separate (see [setVideoScreenOn]).
     */
    fun setVideoSurfaceVisible(visible: Boolean) {
        callbacks.queueEventOnGlThread { renderer.setVideoSurfaceVisible(visible) }
    }

    /**
     * Sets whether the screen is on. The single shared video decoder pauses the instant the screen
     * goes off (across every engine), independent of which engine owns its output.
     */
    fun setVideoScreenOn(on: Boolean) {
        SharedVideoPlayer.setScreenOn(on)
    }

    protected abstract suspend fun openDownloadedCurrentArtwork(): RenderSource

    fun reloadCurrentArtwork(reloadType: ReloadType = ReloadWhenVisible) {
        if (destroyed) {
            // Don't reload artwork for destroyed RenderControllers
            return
        }
        coroutineScope.launch(Dispatchers.Main) {
            val source = openDownloadedCurrentArtwork()

            callbacks.queueEventOnGlThread {
                if (visible || reloadType != ReloadWhenVisible) {
                    renderer.setAndConsumeSource(source, reloadType == ReloadImmediate || !visible)
                } else {
                    queuedSource = source
                }
            }
        }
    }

    interface Callbacks {
        fun queueEventOnGlThread(event: () -> Unit)
        fun requestRender()
    }
}