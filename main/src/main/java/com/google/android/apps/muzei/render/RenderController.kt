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
import com.google.android.apps.muzei.room.Screen
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
    /**
     * Which screen's artwork the wallpaper is currently rendering. This is driven
     * by the real keyguard state (see MuzeiWallpaperEngine.lockScreenVisibleChanged),
     * not by the in-app effects preview. Changing it lets a subclass crossfade to
     * that screen's provider artwork; when the lock screen is linked to home it
     * resolves to the same artwork and is a no-op.
     */
    var activeScreen: Screen = Screen.HOME
        set(value) {
            if (field != value) {
                field = value
                onActiveScreenChanged(value)
            }
        }
    private lateinit var coroutineScope: CoroutineScope
    private var destroyed = false
    private var queuedImageLoader: ImageLoader? = null
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
        queuedImageLoader = null
        Prefs.getSharedPreferences(context)
                .unregisterOnSharedPreferenceChangeListener(sharedPreferenceChangeListener)
        destroyed = true
    }

    protected abstract suspend fun openDownloadedCurrentArtwork(): ImageLoader

    /**
     * Called when [activeScreen] changes so subclasses can swap the rendered
     * artwork to that screen's provider (crossfading from the current artwork).
     */
    protected open fun onActiveScreenChanged(screen: Screen) {}

    /**
     * Hold the outgoing screen's effects steady ahead of a crossfade to a different
     * screen's artwork, so the incoming screen's effects don't bleed onto the
     * previous screen as it fades out. Call immediately before [reloadCurrentArtwork].
     */
    protected fun holdEffectsForScreenSwitch() {
        callbacks.queueEventOnGlThread { renderer.holdEffectsForScreenSwitch() }
    }

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