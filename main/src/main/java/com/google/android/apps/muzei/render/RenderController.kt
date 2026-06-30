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

abstract class RenderController(
        protected var context: Context,
        protected var renderer: MuzeiBlurRenderer,
        private var callbacks: Callbacks
) : DefaultLifecycleObserver {

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
        Prefs.getSharedPreferences(context)
                .unregisterOnSharedPreferenceChangeListener(sharedPreferenceChangeListener)
        destroyed = true
    }

    protected abstract suspend fun openDownloadedCurrentArtwork(): ImageLoader

    fun reloadCurrentArtwork() {
        if (destroyed) {
            // Don't reload artwork for destroyed RenderControllers
            return
        }
        coroutineScope.launch(Dispatchers.Main) {
            val imageLoader = openDownloadedCurrentArtwork()
            callbacks.queueEventOnGlThread {
                // The renderer buffers this itself if its surface isn't ready yet.
                renderer.setAndConsumeImageLoader(imageLoader)
            }
        }
    }

    interface Callbacks {
        fun queueEventOnGlThread(event: () -> Unit)
        fun requestRender()
    }
}