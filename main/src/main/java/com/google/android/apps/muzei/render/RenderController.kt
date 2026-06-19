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
                    val loader = queuedImageLoader
                    if (loader != null) {
                        queuedImageLoader = null
                        renderer.setAndConsumeImageLoader(loader)
                    }
                }
                callbacks.requestRender()
            }
        }
    var onLockScreen: Boolean = false
        set(value) {
            if (field != value) {
                field = value
                // Both blur levels are pre-blurred per artwork, so the home<->lock transition is
                // just a time-based crossfade of those textures plus dim/grey interpolation — no
                // re-decode and no blur work. It keeps running in the background to completion
                // (see MuzeiWallpaperEngine.onVisibilityChanged).
                renderer.setOnLockScreen(value)
            }
        }
    private lateinit var coroutineScope: CoroutineScope
    private var destroyed = false
    private var queuedImageLoader: ImageLoader? = null
    private val sharedPreferenceChangeListener = SharedPreferences.OnSharedPreferenceChangeListener { _, key ->
        // The renderer reads all six home/lock blur/dim/grey amounts, so any of them recomputes the
        // same way. The blur levels re-blur lazily on the next frame if a radius changed.
        when (key) {
            Prefs.PREF_BLUR_AMOUNT, Prefs.PREF_DIM_AMOUNT, Prefs.PREF_GREY_AMOUNT,
            Prefs.PREF_LOCK_BLUR_AMOUNT, Prefs.PREF_LOCK_DIM_AMOUNT, Prefs.PREF_LOCK_GREY_AMOUNT -> {
                renderer.recomputeEffects()
                callbacks.requestRender()
            }
        }
    }

    override fun onCreate(owner: LifecycleOwner) {
        coroutineScope = owner.lifecycleScope
        Prefs.getSharedPreferences(context)
                .registerOnSharedPreferenceChangeListener(sharedPreferenceChangeListener)
    }

    override fun onDestroy(owner: LifecycleOwner) {
        queuedImageLoader = null
        Prefs.getSharedPreferences(context)
                .unregisterOnSharedPreferenceChangeListener(sharedPreferenceChangeListener)
        destroyed = true
    }

    protected abstract suspend fun openDownloadedCurrentArtwork(): ImageLoader

    fun reloadCurrentArtwork(reloadType: ReloadType = ReloadWhenVisible) {
        if (destroyed) {
            // Don't reload artwork for destroyed RenderControllers
            return
        }
        coroutineScope.launch(Dispatchers.Main) {
            val imageLoader = openDownloadedCurrentArtwork()

            callbacks.queueEventOnGlThread {
                if (visible || reloadType != ReloadWhenVisible) {
                    renderer.setAndConsumeImageLoader(imageLoader, reloadType == ReloadImmediate || !visible)
                } else {
                    queuedImageLoader = imageLoader
                }
            }
        }
    }

    interface Callbacks {
        fun queueEventOnGlThread(event: () -> Unit)
        fun requestRender()
    }
}